# Fine-tuning-large-time-series-models
# Times 2.0 微调 - 快速启动

## 一键运行 (服务器)

```bash
cd /path/to/timesfm-master/src
chmod +x run_finetuning.sh
./run_finetuning.sh
```

## 前置要求

 Python 3.11+
 PyTorch + CUDA
 16GB 显存
 本地模型: `fm_500_models/`
 数据文件: `east_settlement.csv`

## 文件清单

| 文件 | 说明 |
|------|------|
| `run_finetuning.sh` | 一键运行脚本  |
| `prepare_data_for_finetuning.py` | 数据准备 |
| `finetune_timesfm2_east.py` | 微调主程序 |
| `finetune_config.py` | 配置管理 |

## 默认配置 (16GB显存)

```python
batch_size = 8
num_epochs = 30
context_len = 512
horizon_len = 96
learning_rate = 5e-5
```

**预计时间**: 3-6 小时
**预计显存**: 12-14 GB

## 手动运行

```bash
# 步骤 1: 准备数据
python3 prepare_data_for_finetuning.py

# 步骤 2: 开始微调
python3 finetune_timesfm2_east.py
```

## 使用 screen/tmux

```bash
# 创建会话
screen -S finetune
# 或
tmux new -s finetune

# 运行
./run_finetuning.sh

# 分离: Ctrl+A D (screen) 或 Ctrl+B D (tmux)
# 重连: screen -r finetune 或 tmux attach -t finetune
```

## 调整配置

**如果显存不足:**
```python
# 编辑 finetune_timesfm2_east.py
batch_size = 4  # 改为 4
context_len = 384  # 改为 384
```

**如果想快速测试:**
```python
num_epochs = 5  # 改为 5
```

## 输出文件

```
finetuning_output/
├── finetuned_model.pt          # 微调后的模型
├── training_history.json       # 训练历史
└── training_history.png        # 训练曲线
```

## 检查进度

```bash
# 查看显存使用
nvidia-smi

# 查看输出日志
tail -f finetuning.log  # 如果有日志文件

# 实时监控
watch -n 1 nvidia-smi
```

## 常见命令

```bash
# 查看 GPU
nvidia-smi

# 查看配置
python3 finetune_config.py --config 16gb

# 测试 CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# 查看数据
head east_settlement.csv
```

详细文档: `FINETUNE_README.md`

---
**运行命令**

```bash
./run_finetuning.sh
```

