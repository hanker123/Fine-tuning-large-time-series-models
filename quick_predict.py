"""
快速预测脚本 - 预测单个列并可视化
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# 添加 v1 路径
script_dir = Path(__file__).parent.absolute()
v1_path = script_dir.parent / "v1" / "src"
sys.path.insert(0, str(v1_path))

from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder, TimesFMConfig


def load_model(model_path, device="cuda"):
    """加载微调模型"""
    tfm_config = TimesFMConfig(
        num_layers=50,
        hidden_size=1280,
        num_heads=16,
        horizon_len=128,
        use_positional_embedding=False,
    )

    model = PatchedTimeSeriesDecoder(tfm_config)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, tfm_config


def predict_column(model, data, context_length=512, device="cuda"):
    """预测单个列"""
    # 准备输入
    x_context = torch.tensor(data[-context_length:], dtype=torch.float32).unsqueeze(0)
    input_padding = torch.zeros_like(x_context)
    freq = torch.tensor([0], dtype=torch.long)

    # 移到设备
    x_context = x_context.to(device)
    input_padding = input_padding.to(device)
    freq = freq.to(device)

    # 预测
    with torch.no_grad():
        predictions = model(x_context, input_padding.float(), freq)

    # 提取预测
    last_patch_pred = predictions[0, -1, :, :]
    point_forecast = last_patch_pred[:, 0].cpu().numpy()
    quantile_forecasts = last_patch_pred[:, 1:].cpu().numpy()

    return point_forecast, quantile_forecasts


def plot_forecast(historical_data, forecast, quantiles, title="预测结果", save_path=None):
    """绘制预测结果"""
    fig, ax = plt.subplots(figsize=(15, 6))

    # 历史数据（显示最后 512 个点）
    hist_len = min(512, len(historical_data))
    hist_x = range(-hist_len, 0)
    hist_y = historical_data[-hist_len:]

    # 预测数据
    forecast_x = range(0, len(forecast))
    forecast_y = forecast

    # 绘制历史数据
    ax.plot(hist_x, hist_y, 'b-', linewidth=2, label='历史数据')

    # 绘制预测
    ax.plot(forecast_x, forecast_y, 'r-', linewidth=2, label='预测值')

    # 绘制置信区间（使用 10% 和 90% 分位数）
    if quantiles is not None:
        q10 = quantiles[:, 0]  # 10% 分位数
        q90 = quantiles[:, 8]  # 90% 分位数
        ax.fill_between(forecast_x, q10, q90, alpha=0.3, color='red', label='80% 置信区间')

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='预测起点')
    ax.set_xlabel('时间步', fontsize=12, fontweight='bold')
    ax.set_ylabel('值', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图表已保存: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='使用微调模型快速预测')
    parser.add_argument('--column', type=str, default='OT',
                      help='要预测的列名 (默认: OT)')
    parser.add_argument('--model', type=str, default='finetuning_output/finetuned_model.pt',
                      help='模型路径')
    parser.add_argument('--data', type=str, default='east_settlement.csv',
                      help='数据文件路径')
    parser.add_argument('--output', type=str, default='predictions',
                      help='输出目录')
    parser.add_argument('--plot', action='store_true',
                      help='绘制预测图表')

    args = parser.parse_args()

    print("="*80)
    print(f"快速预测 - 列: {args.column}")
    print("="*80)

    # 路径
    script_dir = Path(__file__).parent.absolute()
    model_path = script_dir / args.model
    data_path = script_dir / args.data
    output_dir = script_dir / args.output
    output_dir.mkdir(exist_ok=True)

    # 检查文件
    if not model_path.exists():
        print(f"✗ 模型不存在: {model_path}")
        return

    if not data_path.exists():
        print(f"✗ 数据文件不存在: {data_path}")
        return

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n设备: {device}")

    # 加载模型
    print(f"加载模型: {model_path}")
    model, tfm_config = load_model(str(model_path), device=device)
    print("✓ 模型加载成功")

    # 读取数据
    print(f"\n读取数据: {data_path}")
    df = pd.read_csv(data_path)

    if args.column not in df.columns:
        print(f"✗ 列 '{args.column}' 不存在")
        print(f"可用列: {list(df.columns)}")
        return

    data = df[args.column].values
    print(f"✓ 数据长度: {len(data)}")

    # 预测
    print(f"\n执行预测...")
    point_forecast, quantile_forecasts = predict_column(model, data, device=device)

    print(f"✓ 预测完成")
    print(f"  预测长度: {len(point_forecast)}")
    print(f"  预测范围: [{point_forecast.min():.2f}, {point_forecast.max():.2f}]")

    # 保存结果
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    result = {
        'step': range(1, len(point_forecast) + 1),
        'point_forecast': point_forecast,
    }

    for i, q in enumerate(quantiles):
        result[f'q{int(q*100)}'] = quantile_forecasts[:, i]

    result_df = pd.DataFrame(result)

    output_file = output_dir / f"forecast_{args.column}.csv"
    result_df.to_csv(output_file, index=False)
    print(f"\n✓ 预测结果已保存: {output_file}")

    # 显示统计
    print("\n预测统计:")
    print(f"  均值: {point_forecast.mean():.2f}")
    print(f"  标准差: {point_forecast.std():.2f}")
    print(f"  最小值: {point_forecast.min():.2f}")
    print(f"  最大值: {point_forecast.max():.2f}")

    # 绘图
    if args.plot:
        print("\n绘制预测图表...")
        plot_path = output_dir / f"forecast_{args.column}.png"
        plot_forecast(
            data, point_forecast, quantile_forecasts,
            title=f"列 '{args.column}' 的预测结果",
            save_path=plot_path
        )

    print("\n" + "="*80)
    print("完成！")
    print("="*80)


if __name__ == "__main__":
    main()
