"""
准备 east_settlement.csv 数据用于 Times 2.0 微调
将多列时间序列转换为长格式，适配微调框架
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def prepare_east_settlement_data(
    input_csv: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
):
    """
    准备数据用于微调

    Args:
        input_csv: 输入 CSV 文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    print("="*80)
    print("PREPARING EAST SETTLEMENT DATA FOR TIMESFM 2.0 FINETUNING")
    print("="*80)

    # 读取数据
    print(f"\n1. Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"   ✓ Loaded {len(df)} rows, {len(df.columns)} columns")

    # 确保日期列是 datetime 格式
    df['date'] = pd.to_datetime(df['date'])

    # 获取所有数值列（除了日期列）
    numeric_cols = df.columns[1:].tolist()
    print(f"   ✓ Found {len(numeric_cols)} numeric columns")

    # 计算分割点
    total_rows = len(df)
    train_end = int(total_rows * train_ratio)
    val_end = int(total_rows * (train_ratio + val_ratio))

    print(f"\n2. Splitting data:")
    print(f"   - Train: 0 to {train_end} ({train_ratio*100:.1f}%)")
    print(f"   - Val:   {train_end} to {val_end} ({val_ratio*100:.1f}%)")
    print(f"   - Test:  {val_end} to {total_rows} ({test_ratio*100:.1f}%)")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 方案 1: 合并所有列为单个长时间序列（适用于单变量模型）
    print("\n3. Creating Format 1: Single concatenated series per column")

    for col_idx, col_name in enumerate(numeric_cols):
        series_data = df[['date', col_name]].copy()
        series_data.columns = ['ds', 'ts']
        series_data['unique_id'] = f'series_{col_idx}_{col_name}'

        # 分割数据
        train_data = series_data.iloc[:train_end].copy()
        val_data = series_data.iloc[train_end:val_end].copy()
        test_data = series_data.iloc[val_end:].copy()

        # 保存（追加模式）
        mode = 'w' if col_idx == 0 else 'a'
        header = col_idx == 0

        train_data.to_csv(os.path.join(output_dir, 'train_long.csv'),
                         mode=mode, header=header, index=False)
        val_data.to_csv(os.path.join(output_dir, 'val_long.csv'),
                       mode=mode, header=header, index=False)
        test_data.to_csv(os.path.join(output_dir, 'test_long.csv'),
                        mode=mode, header=header, index=False)

    print(f"   ✓ Saved {len(numeric_cols)} series to train_long.csv, val_long.csv, test_long.csv")

    # 方案 2: 每列单独保存（用于分别训练）
    print("\n4. Creating Format 2: Separate files per column")

    # 创建子目录
    train_dir = os.path.join(output_dir, 'train_separate')
    val_dir = os.path.join(output_dir, 'val_separate')
    test_dir = os.path.join(output_dir, 'test_separate')

    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)

    for col_name in numeric_cols:
        series_data = df[['date', col_name]].copy()
        series_data.columns = ['ds', 'ts']

        # 分割并保存
        series_data.iloc[:train_end].to_csv(
            os.path.join(train_dir, f'{col_name}.csv'), index=False)
        series_data.iloc[train_end:val_end].to_csv(
            os.path.join(val_dir, f'{col_name}.csv'), index=False)
        series_data.iloc[val_end:].to_csv(
            os.path.join(test_dir, f'{col_name}.csv'), index=False)

    print(f"   ✓ Saved {len(numeric_cols)} separate files in train/val/test directories")

    # 方案 3: 创建滑动窗口样本（最适合 TimesFM）
    print("\n5. Creating Format 3: Sliding window samples (recommended)")

    context_len = 512
    horizon_len = 128  # 使用 128 匹配 TimesFM 2.0-500m checkpoint

    all_train_samples = []
    all_val_samples = []
    all_test_samples = []

    for col_name in numeric_cols:
        series = df[col_name].values

        # 训练集窗口
        for i in range(0, train_end - context_len - horizon_len, horizon_len // 2):
            all_train_samples.append({
                'series_id': col_name,
                'start_idx': i,
                'context': series[i:i+context_len].tolist(),
                'target': series[i+context_len:i+context_len+horizon_len].tolist()
            })

        # 验证集窗口
        for i in range(train_end, val_end - context_len - horizon_len, horizon_len):
            all_val_samples.append({
                'series_id': col_name,
                'start_idx': i,
                'context': series[i:i+context_len].tolist(),
                'target': series[i+context_len:i+context_len+horizon_len].tolist()
            })

        # 测试集窗口
        for i in range(val_end, total_rows - context_len - horizon_len, horizon_len):
            all_test_samples.append({
                'series_id': col_name,
                'start_idx': i,
                'context': series[i:i+context_len].tolist(),
                'target': series[i+context_len:i+context_len+horizon_len].tolist()
            })

    # 保存为 numpy arrays（更高效）
    np.save(os.path.join(output_dir, 'train_samples.npy'), all_train_samples)
    np.save(os.path.join(output_dir, 'val_samples.npy'), all_val_samples)
    np.save(os.path.join(output_dir, 'test_samples.npy'), all_test_samples)

    print(f"   ✓ Train samples: {len(all_train_samples)}")
    print(f"   ✓ Val samples:   {len(all_val_samples)}")
    print(f"   ✓ Test samples:  {len(all_test_samples)}")

    # 保存数据统计信息
    stats = {
        'total_rows': total_rows,
        'num_columns': len(numeric_cols),
        'column_names': numeric_cols,
        'train_end': train_end,
        'val_end': val_end,
        'context_len': context_len,
        'horizon_len': horizon_len,
        'train_samples': len(all_train_samples),
        'val_samples': len(all_val_samples),
        'test_samples': len(all_test_samples),
    }

    import json
    with open(os.path.join(output_dir, 'data_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n6. Data statistics:")
    for key, value in stats.items():
        if key != 'column_names':
            print(f"   - {key}: {value}")

    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  1. train_long.csv, val_long.csv, test_long.csv")
    print("  2. train_separate/, val_separate/, test_separate/")
    print("  3. train_samples.npy, val_samples.npy, test_samples.npy (recommended)")
    print("  4. data_stats.json")
    print("\n")

    return stats


if __name__ == "__main__":
    # 配置路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(script_dir, "east_settlement.csv")
    output_dir = os.path.join(script_dir, "finetune_data")

    # 准备数据
    stats = prepare_east_settlement_data(
        input_csv=input_csv,
        output_dir=output_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
