#!/bin/bash

################################################################################
# TimesFM 2.0 微调脚本 - East Settlement Data
# 适配 16GB 显存，使用本地 fm_500_models
################################################################################

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 打印分隔线
print_separator() {
    echo "================================================================================"
}

print_separator
echo "TimesFM 2.0 Finetuning - East Settlement Data"
echo "Using Local Model: fm_500_models"
print_separator
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

print_info "Working directory: $SCRIPT_DIR"
echo ""

################################################################################
# Step 1: 检查依赖
################################################################################
print_separator
echo "Step 1/4: Checking Dependencies"
print_separator

# 检查 Python
if ! command -v python3 &> /dev/null; then
    print_error "Python3 not found! Please install Python 3.11+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_success "Python version: $PYTHON_VERSION"

# 检查 PyTorch
if ! python3 -c "import torch" 2>/dev/null; then
    print_error "PyTorch not installed!"
    echo "  Install with: pip install torch torchvision torchaudio"
    exit 1
fi

print_success "PyTorch installed"

# 检查 CUDA
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEMORY=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}')")
    print_success "GPU detected: $GPU_NAME"
    print_success "GPU Memory: ${GPU_MEMORY} GB"
else
    print_warning "CUDA not available! Training will use CPU (very slow)"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""

################################################################################
# Step 2: 检查文件
################################################################################
print_separator
echo "Step 2/4: Checking Files"
print_separator

# 检查数据文件
if [ ! -f "east_settlement.csv" ]; then
    print_error "east_settlement.csv not found!"
    echo "  Please place the data file in: $SCRIPT_DIR"
    exit 1
fi

print_success "Data file found: east_settlement.csv"
DATA_SIZE=$(du -h east_settlement.csv | cut -f1)
print_info "  Size: $DATA_SIZE"

# 检查本地模型
if [ ! -d "fm_500_models" ]; then
    print_error "Local model directory not found: fm_500_models"
    echo "  Please ensure fm_500_models directory exists in: $SCRIPT_DIR"
    exit 1
fi

if [ -f "fm_500_models/torch_model.ckpt" ]; then
    MODEL_FILE="fm_500_models/torch_model.ckpt"
    print_success "Model checkpoint found: $MODEL_FILE"
elif [ -f "fm_500_models/model.safetensors" ]; then
    MODEL_FILE="fm_500_models/model.safetensors"
    print_success "Model safetensors found: $MODEL_FILE"
else
    print_error "No model weights found in fm_500_models/"
    echo "  Expected: torch_model.ckpt or model.safetensors"
    exit 1
fi

MODEL_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
print_info "  Size: $MODEL_SIZE"

# 检查 Python 脚本
for script in "prepare_data_for_finetuning.py" "finetune_timesfm2_east.py"; do
    if [ ! -f "$script" ]; then
        print_error "Script not found: $script"
        exit 1
    fi
    print_success "Found: $script"
done

echo ""

################################################################################
# Step 3: 准备数据
################################################################################
print_separator
echo "Step 3/4: Preparing Training Data"
print_separator

if [ -f "finetune_data/train_samples.npy" ]; then
    # 检查现有数据的 horizon_len 是否匹配
    if [ -f "finetune_data/data_stats.json" ]; then
        EXISTING_HORIZON=$(python3 -c "import json; f=open('finetune_data/data_stats.json'); stats=json.load(f); print(stats.get('horizon_len', 0)); f.close()")

        if [ "$EXISTING_HORIZON" != "128" ]; then
            print_warning "Existing data has horizon_len=$EXISTING_HORIZON, but need 128"
            print_info "Regenerating data with correct horizon_len..."
            python3 prepare_data_for_finetuning.py
            if [ $? -ne 0 ]; then
                print_error "Data preparation failed!"
                exit 1
            fi
            print_success "Data preparation completed"
        else
            print_success "Existing data matches configuration (horizon_len=128)"
            print_warning "Skip data preparation?"
            read -p "Skip? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                print_info "Using existing data"
            else
                print_info "Regenerating data..."
                python3 prepare_data_for_finetuning.py
                if [ $? -ne 0 ]; then
                    print_error "Data preparation failed!"
                    exit 1
                fi
                print_success "Data preparation completed"
            fi
        fi
    else
        print_warning "Cannot verify existing data. Regenerating..."
        python3 prepare_data_for_finetuning.py
        if [ $? -ne 0 ]; then
            print_error "Data preparation failed!"
            exit 1
        fi
        print_success "Data preparation completed"
    fi
else
    print_info "Preparing data..."
    python3 prepare_data_for_finetuning.py
    if [ $? -ne 0 ]; then
        print_error "Data preparation failed!"
        exit 1
    fi
    print_success "Data preparation completed"
fi

# 显示数据统计
if [ -f "finetune_data/data_stats.json" ]; then
    print_info "Data statistics:"
    python3 -c "
import json
with open('finetune_data/data_stats.json', 'r') as f:
    stats = json.load(f)
print(f\"  - Train samples: {stats['train_samples']}\")
print(f\"  - Val samples: {stats['val_samples']}\")
print(f\"  - Test samples: {stats['test_samples']}\")
print(f\"  - Context length: {stats['context_len']}\")
print(f\"  - Horizon length: {stats['horizon_len']}\")
"
fi

echo ""

################################################################################
# Step 4: 开始微调
################################################################################
print_separator
echo "Step 4/4: Starting Finetuning"
print_separator

echo ""
print_info "Training Configuration:"
echo "  - Model: TimesFM 2.0-500m (local)"
echo "  - Batch size: 8"
echo "  - Epochs: 30"
echo "  - Context length: 512"
echo "  - Horizon length: 128"
echo "  - Learning rate: 5e-5"
echo "  - Device: $(python3 -c 'import torch; print("CUDA" if torch.cuda.is_available() else "CPU")')"
echo ""
print_warning "Estimated training time: 3-6 hours (on GPU)"
print_warning "Estimated GPU memory usage: 12-14 GB"
echo ""

read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Training cancelled"
    exit 0
fi

print_info "Starting training..."
echo ""

# 运行微调
python3 finetune_timesfm2_east.py

if [ $? -eq 0 ]; then
    echo ""
    print_separator
    print_success "FINETUNING COMPLETED!"
    print_separator
    echo ""

    # 显示输出文件
    if [ -d "finetuning_output" ]; then
        print_info "Output files:"
        ls -lh finetuning_output/
        echo ""
        print_info "Output directory: $SCRIPT_DIR/finetuning_output"
    fi

    print_success "All done!"
else
    echo ""
    print_error "Finetuning failed!"
    print_info "Please check the error messages above"
    exit 1
fi
