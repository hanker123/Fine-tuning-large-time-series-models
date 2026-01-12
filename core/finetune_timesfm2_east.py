"""
Times 2.0 微调脚本 - 适配 16GB 显存
使用 LoRA 方法微调 east_settlement 数据
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加 v1 路径到 Python path
script_dir = Path(__file__).parent.absolute()
v1_path = script_dir.parent / "v1" / "src"
sys.path.insert(0, str(v1_path))

# 导入 TimesFM 相关模块
try:
    from finetuning.finetuning_torch import FinetuningConfig, TimesFMFinetuner
    from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
    from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder, TimesFMConfig
except ImportError as e:
    print(f"Error importing TimesFM modules: {e}")
    print(f"Please ensure you have installed TimesFM v1 dependencies")
    sys.exit(1)


class EastSettlementDataset(Dataset):
    """East Settlement 数据集"""

    def __init__(self, samples_path, context_length=512, horizon_length=128, freq_type=0):
        """
        Args:
            samples_path: numpy 样本文件路径
            context_length: 上下文长度
            horizon_length: 预测长度
            freq_type: 频率类型 (0=高频, 1=中频, 2=低频)
        """
        self.samples = np.load(samples_path, allow_pickle=True)
        self.context_length = context_length
        self.horizon_length = horizon_length
        self.freq_type = freq_type

        print(f"Loaded {len(self.samples)} samples from {samples_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 获取上下文和目标
        x_context = np.array(sample['context'], dtype=np.float32)
        x_future = np.array(sample['target'], dtype=np.float32)

        # 转换为 tensor
        x_context = torch.tensor(x_context, dtype=torch.float32)
        x_future = torch.tensor(x_future, dtype=torch.float32)

        # 创建 padding mask (全零表示无 padding)
        input_padding = torch.zeros_like(x_context)

        # 频率类型
        freq = torch.tensor([self.freq_type], dtype=torch.long)

        return x_context, input_padding, freq, x_future


def load_model(local_model_path=None):
    """加载 TimesFM 2.0 模型"""
    print("\n" + "="*80)
    print("LOADING TIMESFM 2.0 MODEL")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 配置模型参数 (TimesFM 2.0-500m)
    hparams = TimesFmHparams(
        backend=device,
        per_core_batch_size=8,  # 16GB 显存适中的 batch size
        horizon_len=128,  # 使用 128 匹配 checkpoint
        num_layers=50,  # TimesFM 2.0 使用 50 层
        use_positional_embedding=False,  # 2.0 不使用位置编码
        context_len=512,
    )

    print(f"\nModel configuration:")
    print(f"  - num_layers: {hparams.num_layers}")
    print(f"  - context_len: {hparams.context_len}")
    print(f"  - horizon_len: {hparams.horizon_len}")
    print(f"  - batch_size: {hparams.per_core_batch_size}")

    # 创建模型 - 使用 TimesFM 2.0-500m 配置（50层）
    tfm_config = TimesFMConfig(
        num_layers=50,  # 重要：设置为 50 层匹配 checkpoint
        hidden_size=1280,
        num_heads=16,
        horizon_len=128,  # 使用 128 匹配 checkpoint（预测 5.3 天）
        use_positional_embedding=False,  # TimesFM 2.0 不使用位置编码
    )
    model = PatchedTimeSeriesDecoder(tfm_config)

    # 验证模型层数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total parameters: {total_params:,} (~{total_params * 4 / 1e9:.2f} GB)")
    print(f"  - Model layers: {tfm_config.num_layers}")

    # 加载预训练权重
    if local_model_path:
        # 检查是否是目录或文件
        if os.path.isdir(local_model_path):
            # 优先使用 torch_model.ckpt (PyTorch native format)
            ckpt_path = os.path.join(local_model_path, "torch_model.ckpt")
            safetensors_path = os.path.join(local_model_path, "model.safetensors")

            if os.path.exists(ckpt_path):
                print(f"\n✓ Found local checkpoint: {ckpt_path}")
                print(f"  Size: {os.path.getsize(ckpt_path) / 1e9:.2f} GB")

                # 加载 checkpoint (不使用 weights_only 以支持旧格式)
                loaded_checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

                # 检查 checkpoint 格式
                if isinstance(loaded_checkpoint, dict):
                    # 可能的键: 'model_state_dict', 'state_dict', 'model', 或直接是 state_dict
                    if 'model_state_dict' in loaded_checkpoint:
                        state_dict = loaded_checkpoint['model_state_dict']
                        print("  → Using 'model_state_dict' from checkpoint")
                    elif 'state_dict' in loaded_checkpoint:
                        state_dict = loaded_checkpoint['state_dict']
                        print("  → Using 'state_dict' from checkpoint")
                    elif 'model' in loaded_checkpoint:
                        state_dict = loaded_checkpoint['model']
                        print("  → Using 'model' from checkpoint")
                    else:
                        # 直接就是 state_dict
                        state_dict = loaded_checkpoint
                        print("  → Checkpoint is direct state_dict")
                else:
                    state_dict = loaded_checkpoint

                # 加载权重（horizon_len=128 完美匹配 checkpoint）
                try:
                    model.load_state_dict(state_dict, strict=True)
                    print("✓ Model loaded from local torch checkpoint (strict=True)")
                    print("  → All pretrained weights loaded successfully!")
                except RuntimeError as e:
                    print(f"  Warning: Strict loading failed, trying with strict=False")
                    print(f"  Error was: {str(e)[:200]}...")
                    result = model.load_state_dict(state_dict, strict=False)
                    print("✓ Model loaded from local torch checkpoint (strict=False)")
                    if result.missing_keys:
                        print(f"  ⚠ Missing keys: {len(result.missing_keys)}")
                    if result.unexpected_keys:
                        print(f"  ⚠ Unexpected keys: {len(result.unexpected_keys)}")
            elif os.path.exists(safetensors_path):
                print(f"\n✓ Found local safetensors: {safetensors_path}")
                print(f"  Size: {os.path.getsize(safetensors_path) / 1e9:.2f} GB")
                from safetensors.torch import load_file
                state_dict = load_file(safetensors_path, device=str(device))

                # 加载权重（horizon_len=128 完美匹配 checkpoint）
                try:
                    model.load_state_dict(state_dict, strict=True)
                    print("✓ Model loaded from local safetensors (strict=True)")
                    print("  → All pretrained weights loaded successfully!")
                except RuntimeError as e:
                    print(f"  Warning: Strict loading failed, trying with strict=False")
                    print(f"  Error was: {str(e)[:200]}...")
                    result = model.load_state_dict(state_dict, strict=False)
                    print("✓ Model loaded from local safetensors (strict=False)")
                    if result.missing_keys:
                        print(f"  ⚠ Missing keys: {len(result.missing_keys)}")
                    if result.unexpected_keys:
                        print(f"  ⚠ Unexpected keys: {len(result.unexpected_keys)}")
            else:
                raise FileNotFoundError(
                    f"No model weights found in {local_model_path}\n"
                    f"Expected: 'torch_model.ckpt' or 'model.safetensors'"
                )
        else:
            # 直接是文件路径
            print(f"\n✓ Loading from: {local_model_path}")
            if local_model_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(local_model_path, device=str(device))
            else:
                loaded_checkpoint = torch.load(local_model_path, map_location=device, weights_only=False)
                # 尝试提取 state_dict
                if isinstance(loaded_checkpoint, dict):
                    if 'model_state_dict' in loaded_checkpoint:
                        state_dict = loaded_checkpoint['model_state_dict']
                    elif 'state_dict' in loaded_checkpoint:
                        state_dict = loaded_checkpoint['state_dict']
                    elif 'model' in loaded_checkpoint:
                        state_dict = loaded_checkpoint['model']
                    else:
                        state_dict = loaded_checkpoint
                else:
                    state_dict = loaded_checkpoint

            # 加载权重（horizon_len=128 完美匹配 checkpoint）
            try:
                model.load_state_dict(state_dict, strict=True)
                print("✓ Model loaded from local file (strict=True)")
                print("  → All pretrained weights loaded successfully!")
            except RuntimeError as e:
                print(f"  Warning: Strict loading failed, trying with strict=False")
                print(f"  Error was: {str(e)[:200]}...")
                result = model.load_state_dict(state_dict, strict=False)
                print("✓ Model loaded from local file (strict=False)")
                if result.missing_keys:
                    print(f"  ⚠ Missing keys: {len(result.missing_keys)}")
                if result.unexpected_keys:
                    print(f"  ⚠ Unexpected keys: {len(result.unexpected_keys)}")
    else:
        # 从 Hugging Face 下载
        print("\nDownloading from Hugging Face: google/timesfm-2.0-500m-pytorch")
        repo_id = "google/timesfm-2.0-500m-pytorch"

        try:
            tfm = TimesFm(
                hparams=hparams,
                checkpoint=TimesFmCheckpoint(huggingface_repo_id=repo_id)
            )
            tfm_config = tfm._model_config
            model = PatchedTimeSeriesDecoder(tfm_config)

            from huggingface_hub import snapshot_download
            checkpoint_path = os.path.join(snapshot_download(repo_id), "torch_model.ckpt")
            loaded_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(loaded_checkpoint)
            print("✓ Model downloaded and loaded from Hugging Face")
        except Exception as e:
            print(f"\n✗ Error downloading model: {e}")
            print("\nTo use a local model, set the path:")
            print("  local_model_dir = '/path/to/fm_500_models'")
            raise

    return model, hparams, tfm_config


def plot_training_history(history, save_path):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss 曲线
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 学习率曲线 (如果有)
    if 'learning_rate' in history:
        ax2.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history plot saved to: {save_path}")
    plt.close()


def main():
    print("\n" + "="*80)
    print("TIMESFM 2.0 FINETUNING - EAST SETTLEMENT DATA")
    print("="*80)

    # ========== 配置 ==========
    script_dir = Path(__file__).parent.absolute()
    data_dir = script_dir / "finetune_data"
    output_dir = script_dir / "finetuning_output"
    output_dir.mkdir(exist_ok=True)

    # 数据路径
    train_samples = data_dir / "train_samples.npy"
    val_samples = data_dir / "val_samples.npy"

    # 检查数据是否存在
    if not train_samples.exists() or not val_samples.exists():
        print("\n✗ Error: Training data not found!")
        print(f"Please run prepare_data_for_finetuning.py first")
        print(f"Expected files:")
        print(f"  - {train_samples}")
        print(f"  - {val_samples}")
        return

    # 加载数据统计
    stats_path = data_dir / "data_stats.json"
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            data_stats = json.load(f)
        print("\nData statistics:")
        print(f"  - Train samples: {data_stats['train_samples']}")
        print(f"  - Val samples: {data_stats['val_samples']}")
        print(f"  - Context length: {data_stats['context_len']}")
        print(f"  - Horizon length: {data_stats['horizon_len']}")

    # ========== 加载模型 ==========
    # 使用本地 TimesFM 2.0-500m 模型
    local_model_dir = script_dir / "fm_500_models"
    model, hparams, tfm_config = load_model(local_model_path=str(local_model_dir))

    # ========== 创建数据集 ==========
    print("\n" + "="*80)
    print("CREATING DATASETS")
    print("="*80)

    train_dataset = EastSettlementDataset(
        samples_path=str(train_samples),
        context_length=512,
        horizon_length=128,  # 使用 128 匹配 checkpoint
        freq_type=0  # 小时数据，高频
    )

    val_dataset = EastSettlementDataset(
        samples_path=str(val_samples),
        context_length=512,
        horizon_length=128,  # 使用 128 匹配 checkpoint
        freq_type=0
    )

    # ========== 配置微调 ==========
    print("\n" + "="*80)
    print("CONFIGURING FINETUNING")
    print("="*80)

    config = FinetuningConfig(
        # 基础配置
        batch_size=8,  # 16GB 显存的安全值
        num_epochs=1,  # 快速测试 - 只训练1个epoch
        learning_rate=3e-5,  # 配合梯度裁剪使用
        weight_decay=0.01,
        freq_type=0,

        # 损失函数
        use_quantile_loss=True,  # 使用分位数损失
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],

        # 设备配置
        device="cuda" if torch.cuda.is_available() else "cpu",
        distributed=False,

        # 日志配置
        use_wandb=False,  # 设为 True 如果想使用 W&B
        wandb_project="timesfm2-east-settlement",
        log_every_n_steps=50,
        val_check_interval=0.5,  # 每半个 epoch 验证一次
    )

    print(f"Finetuning configuration:")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Epochs: {config.num_epochs}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Use quantile loss: {config.use_quantile_loss}")
    print(f"  - Device: {config.device}")

    # ========== 开始微调 ==========
    print("\n" + "="*80)
    print("STARTING FINETUNING")
    print("="*80)

    finetuner = TimesFMFinetuner(model, config)

    try:
        results = finetuner.finetune(
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )

        print("\n" + "="*80)
        print("FINETUNING COMPLETED!")
        print("="*80)

        # 保存结果
        history = results['history']
        print(f"\nFinal results:")
        print(f"  - Final train loss: {history['train_loss'][-1]:.6f}")
        print(f"  - Final val loss: {history['val_loss'][-1]:.6f}")
        print(f"  - Best val loss: {min(history['val_loss']):.6f}")

        # 绘制训练历史
        plot_path = output_dir / "training_history.png"
        plot_training_history(history, plot_path)

        # 保存模型
        model_save_path = output_dir / "finetuned_model.pt"
        torch.save(model.state_dict(), model_save_path)
        print(f"✓ Model saved to: {model_save_path}")

        # 保存训练历史
        history_path = output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"✓ Training history saved to: {history_path}")

        print("\n" + "="*80)
        print("ALL DONE!")
        print("="*80)
        print(f"\nOutput directory: {output_dir}")
        print("Generated files:")
        print("  - finetuned_model.pt")
        print("  - training_history.json")
        print("  - training_history.png")

    except Exception as e:
        print(f"\n✗ Error during finetuning: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
