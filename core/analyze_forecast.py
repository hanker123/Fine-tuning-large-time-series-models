import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
plt.style.use('seaborn-v0_8-darkgrid')

class ForecastAnalyzer:
    """时间序列预测结果分析器"""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.original_data = None
        self.point_forecast = None
        self.quantile_forecast = None
        self.column_names = None
        self.horizon = None

    def load_data(self):
        """加载原始数据和预测结果"""
        print("Loading data...")

        # 加载原始数据
        csv_path = os.path.join(self.data_dir, "east_settlement.csv")
        self.original_data = pd.read_csv(csv_path)
        print(f"Original data shape: {self.original_data.shape}")

        # 加载点预测
        point_forecast_path = os.path.join(self.data_dir, "east_settlement_point_forecast.csv")
        if os.path.exists(point_forecast_path):
            self.point_forecast = pd.read_csv(point_forecast_path)
            self.horizon = len(self.point_forecast)
            print(f"Point forecast shape: {self.point_forecast.shape}")
        else:
            print("Warning: Point forecast file not found!")

        # 加载分位数预测
        quantile_forecast_path = os.path.join(self.data_dir, "east_settlement_quantile_forecast.npy")
        if os.path.exists(quantile_forecast_path):
            self.quantile_forecast = np.load(quantile_forecast_path)
            print(f"Quantile forecast shape: {self.quantile_forecast.shape}")
        else:
            print("Warning: Quantile forecast file not found!")

        self.column_names = self.original_data.columns[1:].tolist()
        print(f"Number of columns: {len(self.column_names)}")

    def plot_single_series(self, col_idx=0, num_history=200, save_path=None):
        """绘制单个时间序列的预测结果（包含历史数据、点预测和分位数区间）

        Args:
            col_idx: 列索引
            num_history: 显示多少个历史数据点
            save_path: 保存路径
        """
        col_name = self.column_names[col_idx]
        print(f"\nPlotting forecast for column: {col_name}")

        # 获取历史数据
        history = self.original_data[col_name].values[-num_history:]

        # 获取预测数据
        forecast = self.point_forecast[col_name].values

        # 创建时间索引
        # 假设数据是每小时的
        last_date = pd.to_datetime(self.original_data['date'].iloc[-1])
        history_dates = pd.date_range(end=last_date, periods=num_history, freq='H')
        forecast_dates = pd.date_range(start=last_date + timedelta(hours=1), periods=self.horizon, freq='H')

        # 创建图表
        fig, ax = plt.subplots(figsize=(16, 8))

        # 绘制历史数据
        ax.plot(history_dates, history, 'b-', linewidth=2, label='Historical Data', alpha=0.7)

        # 绘制点预测
        ax.plot(forecast_dates, forecast, 'r-', linewidth=2, label='Point Forecast', alpha=0.8)

        # 绘制分位数区间
        if self.quantile_forecast is not None:
            quantiles = self.quantile_forecast[col_idx]  # shape: [horizon, num_quantiles]
            num_quantiles = quantiles.shape[1]

            # 假设分位数是均匀分布的，从0.1到0.9
            # 绘制不同置信区间
            colors = ['#ffcccc', '#ff9999', '#ff6666']
            alphas = [0.3, 0.2, 0.1]

            if num_quantiles >= 9:
                # 90% 置信区间 (0.05-0.95 或 0.1-0.9)
                ax.fill_between(forecast_dates,
                               quantiles[:, 0], quantiles[:, -1],
                               alpha=alphas[2], color=colors[2],
                               label='90% Confidence Interval')

                # 70% 置信区间 (0.15-0.85)
                if num_quantiles >= 7:
                    q_idx_low = min(1, num_quantiles-2)
                    q_idx_high = max(num_quantiles-2, 1)
                    ax.fill_between(forecast_dates,
                                   quantiles[:, q_idx_low], quantiles[:, q_idx_high],
                                   alpha=alphas[1], color=colors[1],
                                   label='70% Confidence Interval')

                # 50% 置信区间 (0.25-0.75)
                if num_quantiles >= 5:
                    q_idx_low = num_quantiles // 4
                    q_idx_high = 3 * num_quantiles // 4
                    ax.fill_between(forecast_dates,
                                   quantiles[:, q_idx_low], quantiles[:, q_idx_high],
                                   alpha=alphas[0], color=colors[0],
                                   label='50% Confidence Interval')

        # 在历史和预测之间添加分隔线
        ax.axvline(x=last_date, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

        # 设置标签和标题
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title(f'Time Series Forecast: {col_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # 格式化x轴日期显示
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        else:
            plt.savefig(os.path.join(self.data_dir, f'forecast_{col_name}.png'),
                       dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {os.path.join(self.data_dir, f'forecast_{col_name}.png')}")

        plt.close()

    def plot_multiple_series(self, col_indices=None, num_history=200, save_path=None):
        """绘制多个时间序列的预测结果

        Args:
            col_indices: 列索引列表，如果为None则绘制前6个
            num_history: 显示多少个历史数据点
            save_path: 保存路径
        """
        if col_indices is None:
            col_indices = range(min(6, len(self.column_names)))

        num_plots = len(col_indices)
        rows = (num_plots + 2) // 3
        cols = min(3, num_plots)

        fig, axes = plt.subplots(rows, cols, figsize=(18, 5*rows))
        if num_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if num_plots > 1 else [axes]

        last_date = pd.to_datetime(self.original_data['date'].iloc[-1])
        history_dates = pd.date_range(end=last_date, periods=num_history, freq='H')
        forecast_dates = pd.date_range(start=last_date + timedelta(hours=1), periods=self.horizon, freq='H')

        for idx, col_idx in enumerate(col_indices):
            ax = axes[idx]
            col_name = self.column_names[col_idx]

            # 历史数据
            history = self.original_data[col_name].values[-num_history:]
            forecast = self.point_forecast[col_name].values

            # 绘制
            ax.plot(history_dates, history, 'b-', linewidth=1.5, label='Historical', alpha=0.7)
            ax.plot(forecast_dates, forecast, 'r-', linewidth=1.5, label='Forecast', alpha=0.8)

            # 分位数区间
            if self.quantile_forecast is not None:
                quantiles = self.quantile_forecast[col_idx]
                if quantiles.shape[1] >= 2:
                    ax.fill_between(forecast_dates,
                                   quantiles[:, 0], quantiles[:, -1],
                                   alpha=0.2, color='red')

            ax.axvline(x=last_date, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_title(f'{col_name}', fontsize=11, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 隐藏多余的子图
        for idx in range(len(col_indices), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved multi-series plot to: {save_path}")
        else:
            plt.savefig(os.path.join(self.data_dir, 'forecast_multiple_series.png'),
                       dpi=300, bbox_inches='tight')
            print(f"Saved multi-series plot to: {os.path.join(self.data_dir, 'forecast_multiple_series.png')}")

        plt.close()

    def analyze_statistics(self, save_path=None):
        """统计分析预测结果"""
        print("\n" + "="*80)
        print("FORECAST STATISTICS ANALYSIS")
        print("="*80)

        stats_data = []

        for col_idx, col_name in enumerate(self.column_names):
            # 历史数据统计
            history = self.original_data[col_name].values
            hist_mean = np.mean(history)
            hist_std = np.std(history)
            hist_min = np.min(history)
            hist_max = np.max(history)

            # 预测数据统计
            forecast = self.point_forecast[col_name].values
            fore_mean = np.mean(forecast)
            fore_std = np.std(forecast)
            fore_min = np.min(forecast)
            fore_max = np.max(forecast)

            # 计算变化率
            mean_change = ((fore_mean - hist_mean) / hist_mean) * 100

            stats_data.append({
                'Column': col_name,
                'Hist_Mean': hist_mean,
                'Hist_Std': hist_std,
                'Hist_Min': hist_min,
                'Hist_Max': hist_max,
                'Forecast_Mean': fore_mean,
                'Forecast_Std': fore_std,
                'Forecast_Min': fore_min,
                'Forecast_Max': fore_max,
                'Mean_Change_%': mean_change
            })

        stats_df = pd.DataFrame(stats_data)

        # 打印摘要
        print(f"\nNumber of time series: {len(self.column_names)}")
        print(f"Forecast horizon: {self.horizon} steps")
        print(f"\nTop 5 columns by mean forecast value:")
        print(stats_df.nlargest(5, 'Forecast_Mean')[['Column', 'Forecast_Mean', 'Mean_Change_%']])

        print(f"\nTop 5 columns by forecast volatility (std):")
        print(stats_df.nlargest(5, 'Forecast_Std')[['Column', 'Forecast_Std', 'Mean_Change_%']])

        print(f"\nTop 5 columns by mean change %:")
        print(stats_df.nlargest(5, 'Mean_Change_%')[['Column', 'Hist_Mean', 'Forecast_Mean', 'Mean_Change_%']])

        # 保存完整统计结果
        stats_path = save_path or os.path.join(self.data_dir, 'forecast_statistics.csv')
        stats_df.to_csv(stats_path, index=False)
        print(f"\nFull statistics saved to: {stats_path}")

        return stats_df

    def plot_uncertainty_comparison(self, col_indices=None, save_path=None):
        """比较不同时间序列的预测不确定性

        Args:
            col_indices: 列索引列表
            save_path: 保存路径
        """
        if self.quantile_forecast is None:
            print("No quantile forecast data available!")
            return

        if col_indices is None:
            col_indices = range(min(10, len(self.column_names)))

        # 计算每个序列的不确定性（分位数范围）
        uncertainty_data = []

        for col_idx in col_indices:
            col_name = self.column_names[col_idx]
            quantiles = self.quantile_forecast[col_idx]

            # 计算平均不确定性（90%置信区间的宽度）
            avg_uncertainty = np.mean(quantiles[:, -1] - quantiles[:, 0])

            # 计算相对不确定性
            point_forecast_mean = self.point_forecast[col_name].mean()
            relative_uncertainty = (avg_uncertainty / point_forecast_mean) * 100 if point_forecast_mean != 0 else 0

            uncertainty_data.append({
                'Column': col_name,
                'Avg_Uncertainty': avg_uncertainty,
                'Relative_Uncertainty_%': relative_uncertainty
            })

        uncertainty_df = pd.DataFrame(uncertainty_data)
        uncertainty_df = uncertainty_df.sort_values('Relative_Uncertainty_%', ascending=False)

        # 绘制条形图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 绝对不确定性
        ax1.barh(uncertainty_df['Column'], uncertainty_df['Avg_Uncertainty'], color='steelblue')
        ax1.set_xlabel('Average Uncertainty (90% CI Width)', fontsize=12, fontweight='bold')
        ax1.set_title('Absolute Forecast Uncertainty', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # 相对不确定性
        colors = ['red' if x > 50 else 'orange' if x > 20 else 'green'
                  for x in uncertainty_df['Relative_Uncertainty_%']]
        ax2.barh(uncertainty_df['Column'], uncertainty_df['Relative_Uncertainty_%'], color=colors)
        ax2.set_xlabel('Relative Uncertainty (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Relative Forecast Uncertainty', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.data_dir, 'uncertainty_comparison.png'),
                       dpi=300, bbox_inches='tight')
        print(f"Uncertainty comparison saved")

        plt.close()

        return uncertainty_df


def main():
    """主函数"""
    # 设置数据目录
    data_dir = os.path.dirname(__file__)

    # 创建分析器
    analyzer = ForecastAnalyzer(data_dir)

    # 加载数据
    analyzer.load_data()

    # 1. 绘制单个详细预测图（第一个列）
    print("\n" + "="*80)
    print("1. Plotting detailed forecast for first column...")
    analyzer.plot_single_series(col_idx=0, num_history=200)

    # 2. 绘制多个序列
    print("\n" + "="*80)
    print("2. Plotting multiple series forecasts...")
    analyzer.plot_multiple_series(col_indices=range(6), num_history=150)

    # 3. 统计分析
    print("\n" + "="*80)
    print("3. Performing statistical analysis...")
    stats_df = analyzer.analyze_statistics()

    # 4. 不确定性比较
    print("\n" + "="*80)
    print("4. Analyzing forecast uncertainty...")
    uncertainty_df = analyzer.plot_uncertainty_comparison(col_indices=range(10))

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  - forecast_<column>.png (detailed forecast plot)")
    print(f"  - forecast_multiple_series.png (multiple series overview)")
    print(f"  - forecast_statistics.csv (statistical analysis)")
    print(f"  - uncertainty_comparison.png (uncertainty analysis)")


if __name__ == "__main__":
    main()
