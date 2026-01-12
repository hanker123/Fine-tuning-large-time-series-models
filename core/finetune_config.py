"""
TimesFM 2.0 微调配置文件
可根据硬件和需求调整参数
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FinetuneConfiguration:
    """微调配置"""

    # ========== 路径配置 ==========
    data_csv: str = "east_settlement.csv"
    local_model_dir: str = "fm_500_models"
    output_dir: str = "finetuning_output"
    data_dir: str = "finetune_data"

    # ========== 数据分割 ==========
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # ========== 模型配置 ==========
    context_length: int = 512  # 上下文长度
    horizon_length: int = 96   # 预测长度
    freq_type: int = 0         # 频率类型: 0=高频(小时), 1=中频(日/周), 2=低频(月/季)

    # ========== 训练配置 ==========
    # 16GB 显存推荐配置
    batch_size: int = 8        # 批次大小 (8-12 适合16GB)
    num_epochs: int = 30       # 训练轮数
    learning_rate: float = 5e-5  # 学习率
    weight_decay: float = 0.01   # 权重衰减

    # ========== 损失函数 ==========
    use_quantile_loss: bool = True  # 使用分位数损失
    quantiles: list = None  # 分位数列表

    # ========== 验证配置 ==========
    val_check_interval: float = 0.5  # 验证间隔 (0.5 = 每半个epoch验证一次)
    log_every_n_steps: int = 50      # 每N步记录一次日志

    # ========== W&B 日志 ==========
    use_wandb: bool = False  # 使用 Weights & Biases
    wandb_project: str = "timesfm2-east-settlement"
    wandb_entity: Optional[str] = None  # W&B 用户名/组织

    # ========== 设备配置 ==========
    # 自动检测，通常不需要修改
    device: str = "auto"  # "auto", "cuda", "cpu"
    distributed: bool = False  # 分布式训练
    gpu_ids: list = None  # GPU IDs for multi-GPU

    def __post_init__(self):
        """初始化后处理"""
        if self.quantiles is None:
            self.quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        if self.gpu_ids is None:
            self.gpu_ids = [0]


# ========== 预定义配置 ==========

def get_config_16gb():
    """16GB 显存配置（推荐）"""
    return FinetuneConfiguration(
        batch_size=8,
        num_epochs=30,
        learning_rate=5e-5,
        context_length=512,
        horizon_length=96,
    )


def get_config_12gb():
    """12GB 显存配置（较小显存）"""
    return FinetuneConfiguration(
        batch_size=4,
        num_epochs=30,
        learning_rate=5e-5,
        context_length=384,  # 减小上下文
        horizon_length=96,
    )


def get_config_24gb():
    """24GB 显存配置（更大显存）"""
    return FinetuneConfiguration(
        batch_size=16,
        num_epochs=50,
        learning_rate=1e-4,
        context_length=512,
        horizon_length=128,
    )


def get_config_fast():
    """快速测试配置"""
    return FinetuneConfiguration(
        batch_size=8,
        num_epochs=5,  # 少量epoch用于测试
        learning_rate=1e-4,
        context_length=256,
        horizon_length=96,
        val_check_interval=1.0,  # 每个epoch结束验证
    )


def get_config_high_precision():
    """高精度配置（更长训练时间）"""
    return FinetuneConfiguration(
        batch_size=8,
        num_epochs=100,  # 更多epoch
        learning_rate=1e-5,  # 更小学习率
        context_length=512,
        horizon_length=96,
        val_check_interval=0.25,  # 更频繁验证
    )


def get_config_cpu():
    """CPU 训练配置"""
    return FinetuneConfiguration(
        batch_size=4,
        num_epochs=10,
        learning_rate=1e-4,
        context_length=256,
        horizon_length=96,
        device="cpu",
    )


# ========== 配置说明 ==========

CONFIG_DESCRIPTIONS = {
    "16gb": "推荐配置，适合 16GB 显存 (RTX 4070/3090等)",
    "12gb": "小显存配置，适合 12GB 显存 (RTX 3060/4060等)",
    "24gb": "大显存配置，适合 24GB 显存 (RTX 4090/A5000等)",
    "fast": "快速测试配置，5 epochs 快速验证",
    "high_precision": "高精度配置，100 epochs 获得最佳效果",
    "cpu": "CPU 训练配置（非常慢，不推荐）",
}


def print_config(config: FinetuneConfiguration):
    """打印配置信息"""
    print("="*80)
    print("FINETUNING CONFIGURATION")
    print("="*80)
    print(f" Paths:")
    print(f"  - Data CSV: {config.data_csv}")
    print(f"  - Model Dir: {config.local_model_dir}")
    print(f"  - Output Dir: {config.output_dir}")

    print(f" Data:")
    print(f"  - Train ratio: {config.train_ratio:.1%}")
    print(f"  - Val ratio: {config.val_ratio:.1%}")
    print(f"  - Test ratio: {config.test_ratio:.1%}")

    print(f" Model:")
    print(f"  - Context length: {config.context_length}")
    print(f"  - Horizon length: {config.horizon_length}")
    print(f"  - Frequency type: {config.freq_type}")

    print(f"  Training:")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Epochs: {config.num_epochs}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Weight decay: {config.weight_decay}")

    print(f" Validation:")
    print(f"  - Val check interval: {config.val_check_interval}")
    print(f"  - Log every N steps: {config.log_every_n_steps}")

    print(f" Device:")
    print(f"  - Device: {config.device}")
    print(f"  - Distributed: {config.distributed}")

    if config.use_wandb:
        print(f" W&B:")
        print(f"  - Project: {config.wandb_project}")
        print(f"  - Entity: {config.wandb_entity or 'default'}")

    print("="*80)


def list_configs():
    """列出所有预定义配置"""
    print("\n可用配置:")
    print("-" * 80)
    for name, desc in CONFIG_DESCRIPTIONS.items():
        print(f"  {name:15s} - {desc}")
    print("-" * 80)


if __name__ == "__main__":
    # 示例用法
    import argparse

    parser = argparse.ArgumentParser(description="查看微调配置")
    parser.add_argument("--config", type=str, default="16gb",
                       choices=CONFIG_DESCRIPTIONS.keys(),
                       help="选择配置")
    args = parser.parse_args()

    # 获取配置
    config_func = {
        "16gb": get_config_16gb,
        "12gb": get_config_12gb,
        "24gb": get_config_24gb,
        "fast": get_config_fast,
        "high_precision": get_config_high_precision,
        "cpu": get_config_cpu,
    }

    config = config_func[args.config]()

    # 打印配置
    list_configs()
    print(f"\n当前选择: {args.config}")
    print_config(config)
