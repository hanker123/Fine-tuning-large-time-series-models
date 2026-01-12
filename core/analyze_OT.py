import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')


class OTColumnAnalyzer:
    """OT列专项分析器"""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.original_data = None
        self.point_forecast = None
        self.quantile_forecast = None
        self.ot_col_idx = None
        self.horizon = None

    def load_data(self):
        """加载数据"""
        print("="*80)
        print("LOADING DATA FOR OT COLUMN ANALYSIS")
        print("="*80)

        # 加载原始数据
        csv_path = os.path.join(self.data_dir, "east_settlement.csv")
        self.original_data = pd.read_csv(csv_path)
        print(f"✓ Original data loaded: {self.original_data.shape}")

        # 找到 OT 列的索引
        columns = self.original_data.columns[1:].tolist()
        if 'OT' in columns:
            self.ot_col_idx = columns.index('OT')
            print(f"✓ OT column found at index: {self.ot_col_idx}")
        else:
            print("✗ Error: OT column not found!")
            print(f"Available columns: {columns}")
            return False

        # 加载点预测
        point_forecast_path = os.path.join(self.data_dir, "east_settlement_point_forecast.csv")
        if os.path.exists(point_forecast_path):
            self.point_forecast = pd.read_csv(point_forecast_path)
            self.horizon = len(self.point_forecast)
            print(f"✓ Point forecast loaded: {self.point_forecast.shape}")
        else:
            print("✗ Point forecast file not found!")
            return False

        # 加载分位数预测
        quantile_forecast_path = os.path.join(self.data_dir, "east_settlement_quantile_forecast.npy")
        if os.path.exists(quantile_forecast_path):
            self.quantile_forecast = np.load(quantile_forecast_path)
            print(f"✓ Quantile forecast loaded: {self.quantile_forecast.shape}")
        else:
            print("⚠ Warning: Quantile forecast file not found!")

        print(f"\nForecast horizon: {self.horizon} time steps (hours)")
        return True

    def plot_detailed_forecast(self, num_history=300):
        """绘制OT列的详细预测图"""
        print("\n" + "="*80)
        print("1. DETAILED FORECAST VISUALIZATION")
        print("="*80)

        # 获取OT列的历史数据
        ot_history = self.original_data['OT'].values[-num_history:]
        ot_forecast = self.point_forecast['OT'].values

        # 创建时间索引
        last_date = pd.to_datetime(self.original_data['date'].iloc[-1])
        history_dates = pd.date_range(end=last_date, periods=num_history, freq='H')
        forecast_dates = pd.date_range(start=last_date + timedelta(hours=1),
                                      periods=self.horizon, freq='H')

        # 创建大图
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 主图：历史 + 预测 + 分位数
        ax_main = fig.add_subplot(gs[0:2, :])

        # 绘制历史数据
        ax_main.plot(history_dates, ot_history, 'b-', linewidth=2.5,
                    label='Historical Data', alpha=0.8)

        # 绘制点预测
        ax_main.plot(forecast_dates, ot_forecast, 'r-', linewidth=2.5,
                    label='Point Forecast', alpha=0.9)

        # 绘制分位数区间
        if self.quantile_forecast is not None:
            quantiles = self.quantile_forecast[self.ot_col_idx]
            num_quantiles = quantiles.shape[1]
            print(f"Number of quantiles: {num_quantiles}")

            if num_quantiles >= 2:
                # 90% 置信区间
                ax_main.fill_between(forecast_dates,
                                    quantiles[:, 0], quantiles[:, -1],
                                    alpha=0.15, color='red',
                                    label='90% Confidence Interval')

                # 如果有足够的分位数，绘制更多区间
                if num_quantiles >= 5:
                    q_25_idx = num_quantiles // 4
                    q_75_idx = 3 * num_quantiles // 4
                    ax_main.fill_between(forecast_dates,
                                        quantiles[:, q_25_idx], quantiles[:, q_75_idx],
                                        alpha=0.25, color='red',
                                        label='50% Confidence Interval')

                if num_quantiles >= 7:
                    q_35_idx = max(1, num_quantiles // 3)
                    q_65_idx = min(num_quantiles - 2, 2 * num_quantiles // 3)
                    ax_main.fill_between(forecast_dates,
                                        quantiles[:, q_35_idx], quantiles[:, q_65_idx],
                                        alpha=0.35, color='red',
                                        label='~30% Confidence Interval')

        # 添加分隔线
        ax_main.axvline(x=last_date, color='green', linestyle='--',
                       linewidth=2, alpha=0.7, label='Forecast Start')

        # 设置标签
        ax_main.set_xlabel('Date & Time', fontsize=14, fontweight='bold')
        ax_main.set_ylabel('OT Value', fontsize=14, fontweight='bold')
        ax_main.set_title('OT Column: Historical Data and Forecast with Confidence Intervals',
                         fontsize=16, fontweight='bold')
        ax_main.legend(loc='best', fontsize=11, framealpha=0.9)
        ax_main.grid(True, alpha=0.3, linestyle='--')
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=0, ha='center')

        # 子图1：预测值分布
        ax_dist = fig.add_subplot(gs[2, 0])
        ax_dist.hist(ot_history, bins=50, alpha=0.6, color='blue',
                    label='Historical', density=True, edgecolor='black')
        ax_dist.hist(ot_forecast, bins=30, alpha=0.6, color='red',
                    label='Forecast', density=True, edgecolor='black')
        ax_dist.set_xlabel('Value', fontsize=11, fontweight='bold')
        ax_dist.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax_dist.set_title('Value Distribution Comparison', fontsize=12, fontweight='bold')
        ax_dist.legend(fontsize=10)
        ax_dist.grid(True, alpha=0.3)

        # 子图2：每小时预测值
        ax_hourly = fig.add_subplot(gs[2, 1])
        hours = np.arange(1, self.horizon + 1)
        ax_hourly.plot(hours, ot_forecast, 'ro-', linewidth=2, markersize=4)
        ax_hourly.axhline(y=np.mean(ot_history), color='blue', linestyle='--',
                         linewidth=2, alpha=0.7, label='Historical Mean')
        ax_hourly.fill_between(hours,
                              np.mean(ot_history) - np.std(ot_history),
                              np.mean(ot_history) + np.std(ot_history),
                              alpha=0.2, color='blue', label='Historical ±1 Std')
        ax_hourly.set_xlabel('Forecast Hour', fontsize=11, fontweight='bold')
        ax_hourly.set_ylabel('Predicted Value', fontsize=11, fontweight='bold')
        ax_hourly.set_title('Hourly Forecast Breakdown', fontsize=12, fontweight='bold')
        ax_hourly.legend(fontsize=9)
        ax_hourly.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.data_dir, 'OT_detailed_forecast.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved detailed forecast plot to: {save_path}")
        plt.close()

    def analyze_statistics(self):
        """统计分析OT列"""
        print("\n" + "="*80)
        print("2. STATISTICAL ANALYSIS")
        print("="*80)

        ot_history = self.original_data['OT'].values
        ot_forecast = self.point_forecast['OT'].values

        # 基础统计
        stats = {
            'Metric': [],
            'Historical': [],
            'Forecast': [],
            'Change': [],
            'Change_%': []
        }

        metrics = [
            ('Mean', np.mean),
            ('Median', np.median),
            ('Std Dev', np.std),
            ('Min', np.min),
            ('Max', np.max),
            ('25th Percentile', lambda x: np.percentile(x, 25)),
            ('75th Percentile', lambda x: np.percentile(x, 75)),
        ]

        print("\n📊 Basic Statistics:")
        print("-" * 80)
        print(f"{'Metric':<20} {'Historical':<15} {'Forecast':<15} {'Change':<15} {'Change %':<10}")
        print("-" * 80)

        for name, func in metrics:
            hist_val = func(ot_history)
            fore_val = func(ot_forecast)
            change = fore_val - hist_val
            change_pct = (change / hist_val * 100) if hist_val != 0 else 0

            stats['Metric'].append(name)
            stats['Historical'].append(hist_val)
            stats['Forecast'].append(fore_val)
            stats['Change'].append(change)
            stats['Change_%'].append(change_pct)

            print(f"{name:<20} {hist_val:<15.2f} {fore_val:<15.2f} {change:<15.2f} {change_pct:<10.2f}%")

        print("-" * 80)

        # 趋势分析
        print("\n📈 Trend Analysis:")
        print("-" * 80)

        # 历史趋势
        recent_history = ot_history[-48:]  # 最近48小时
        hist_trend = np.polyfit(range(len(recent_history)), recent_history, 1)[0]

        # 预测趋势
        fore_trend = np.polyfit(range(len(ot_forecast)), ot_forecast, 1)[0]

        print(f"Historical trend (last 48h): {hist_trend:+.4f} per hour")
        print(f"Forecast trend:                {fore_trend:+.4f} per hour")
        print(f"Trend change:                  {(fore_trend - hist_trend):+.4f} per hour")

        if abs(fore_trend) < 0.1:
            trend_desc = "stable/flat"
        elif fore_trend > 0:
            trend_desc = "increasing/upward"
        else:
            trend_desc = "decreasing/downward"

        print(f"\n→ Forecast shows {trend_desc} trend")

        # 波动性分析
        print("\n📊 Volatility Analysis:")
        print("-" * 80)

        hist_volatility = np.std(np.diff(ot_history))
        fore_volatility = np.std(np.diff(ot_forecast))
        volatility_change = ((fore_volatility - hist_volatility) / hist_volatility) * 100

        print(f"Historical volatility: {hist_volatility:.4f}")
        print(f"Forecast volatility:   {fore_volatility:.4f}")
        print(f"Change:                {volatility_change:+.2f}%")

        if volatility_change > 20:
            vol_desc = "significantly more volatile"
        elif volatility_change > 5:
            vol_desc = "moderately more volatile"
        elif volatility_change < -20:
            vol_desc = "significantly more stable"
        elif volatility_change < -5:
            vol_desc = "moderately more stable"
        else:
            vol_desc = "similar volatility"

        print(f"\n→ Forecast is {vol_desc} compared to historical data")

        # 保存统计结果
        stats_df = pd.DataFrame(stats)
        stats_path = os.path.join(self.data_dir, 'OT_statistics.csv')
        stats_df.to_csv(stats_path, index=False)
        print(f"\n✓ Statistics saved to: {stats_path}")

        return stats_df

    def analyze_quantile_uncertainty(self):
        """分析分位数和不确定性"""
        if self.quantile_forecast is None:
            print("\n⚠ Quantile forecast data not available!")
            return

        print("\n" + "="*80)
        print("3. UNCERTAINTY ANALYSIS")
        print("="*80)

        quantiles = self.quantile_forecast[self.ot_col_idx]
        ot_forecast = self.point_forecast['OT'].values
        num_quantiles = quantiles.shape[1]

        # 计算置信区间
        print("\n📊 Confidence Intervals:")
        print("-" * 80)

        intervals = [
            ('90% CI', 0, -1),
            ('80% CI', 1, -2) if num_quantiles >= 4 else None,
            ('50% CI', num_quantiles // 4, 3 * num_quantiles // 4) if num_quantiles >= 5 else None,
        ]

        ci_data = []

        for interval_info in intervals:
            if interval_info is None:
                continue

            name, low_idx, high_idx = interval_info
            ci_low = quantiles[:, low_idx]
            ci_high = quantiles[:, high_idx]
            ci_width = ci_high - ci_low

            avg_width = np.mean(ci_width)
            max_width = np.max(ci_width)
            min_width = np.min(ci_width)

            # 相对宽度
            avg_forecast = np.mean(ot_forecast)
            relative_width = (avg_width / avg_forecast * 100) if avg_forecast != 0 else 0

            print(f"\n{name}:")
            print(f"  Average width:  {avg_width:.4f} ({relative_width:.2f}% of mean forecast)")
            print(f"  Max width:      {max_width:.4f}")
            print(f"  Min width:      {min_width:.4f}")

            ci_data.append({
                'Interval': name,
                'Avg_Width': avg_width,
                'Max_Width': max_width,
                'Min_Width': min_width,
                'Relative_%': relative_width
            })

        # 时变不确定性
        print("\n📈 Time-varying Uncertainty:")
        print("-" * 80)

        # 90% CI宽度随时间的变化
        ci_width_90 = quantiles[:, -1] - quantiles[:, 0]

        early_uncertainty = np.mean(ci_width_90[:self.horizon//3])
        mid_uncertainty = np.mean(ci_width_90[self.horizon//3:2*self.horizon//3])
        late_uncertainty = np.mean(ci_width_90[2*self.horizon//3:])

        print(f"Early forecast (1-{self.horizon//3}h):    {early_uncertainty:.4f}")
        print(f"Mid forecast ({self.horizon//3+1}-{2*self.horizon//3}h):      {mid_uncertainty:.4f}")
        print(f"Late forecast ({2*self.horizon//3+1}-{self.horizon}h):   {late_uncertainty:.4f}")

        if late_uncertainty > early_uncertainty * 1.2:
            uncertainty_trend = "increasing (less confident for distant future)"
        elif late_uncertainty < early_uncertainty * 0.8:
            uncertainty_trend = "decreasing (more confident for distant future)"
        else:
            uncertainty_trend = "relatively stable"

        print(f"\n→ Uncertainty is {uncertainty_trend}")

        # 可视化不确定性变化
        self._plot_uncertainty_evolution(ci_width_90, ot_forecast)

        # 保存不确定性数据
        ci_df = pd.DataFrame(ci_data)
        ci_path = os.path.join(self.data_dir, 'OT_uncertainty_analysis.csv')
        ci_df.to_csv(ci_path, index=False)
        print(f"\n✓ Uncertainty analysis saved to: {ci_path}")

    def _plot_uncertainty_evolution(self, ci_width, point_forecast):
        """绘制不确定性演变图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

        hours = np.arange(1, self.horizon + 1)

        # 子图1：绝对不确定性
        ax1.plot(hours, ci_width, 'b-', linewidth=2.5, marker='o', markersize=4)
        ax1.fill_between(hours, 0, ci_width, alpha=0.3, color='blue')
        ax1.set_xlabel('Forecast Hour', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Uncertainty (90% CI Width)', fontsize=12, fontweight='bold')
        ax1.set_title('Absolute Forecast Uncertainty Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=np.mean(ci_width), color='red', linestyle='--',
                   linewidth=2, label=f'Average: {np.mean(ci_width):.2f}')
        ax1.legend(fontsize=11)

        # 子图2：相对不确定性
        relative_uncertainty = (ci_width / point_forecast * 100)
        ax2.plot(hours, relative_uncertainty, 'g-', linewidth=2.5, marker='s', markersize=4)
        ax2.fill_between(hours, 0, relative_uncertainty, alpha=0.3, color='green')
        ax2.set_xlabel('Forecast Hour', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Relative Uncertainty (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Relative Forecast Uncertainty Over Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=np.mean(relative_uncertainty), color='red', linestyle='--',
                   linewidth=2, label=f'Average: {np.mean(relative_uncertainty):.2f}%')
        ax2.legend(fontsize=11)

        plt.tight_layout()
        save_path = os.path.join(self.data_dir, 'OT_uncertainty_evolution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Uncertainty evolution plot saved to: {save_path}")
        plt.close()

    def generate_forecast_report(self):
        """生成预测报告"""
        print("\n" + "="*80)
        print("4. FORECAST REPORT SUMMARY")
        print("="*80)

        ot_history = self.original_data['OT'].values
        ot_forecast = self.point_forecast['OT'].values

        # 最后历史值
        last_historical = ot_history[-1]
        first_forecast = ot_forecast[0]
        last_forecast = ot_forecast[-1]
        mean_forecast = np.mean(ot_forecast)

        print(f"\n📌 Key Values:")
        print(f"  Last historical value:  {last_historical:.4f}")
        print(f"  First forecast value:   {first_forecast:.4f}")
        print(f"  Mean forecast value:    {mean_forecast:.4f}")
        print(f"  Last forecast value:    {last_forecast:.4f}")

        # 预测摘要
        immediate_change = first_forecast - last_historical
        overall_change = last_forecast - first_forecast
        total_change = last_forecast - last_historical

        print(f"\n📊 Forecast Changes:")
        print(f"  Immediate change (t+1):        {immediate_change:+.4f} ({immediate_change/last_historical*100:+.2f}%)")
        print(f"  Overall forecast trend:        {overall_change:+.4f} ({overall_change/first_forecast*100:+.2f}%)")
        print(f"  Total change (end vs current): {total_change:+.4f} ({total_change/last_historical*100:+.2f}%)")

        # 极值
        forecast_max = np.max(ot_forecast)
        forecast_min = np.min(ot_forecast)
        max_hour = np.argmax(ot_forecast) + 1
        min_hour = np.argmin(ot_forecast) + 1

        print(f"\n📈 Forecast Extremes:")
        print(f"  Maximum value: {forecast_max:.4f} at hour {max_hour}")
        print(f"  Minimum value: {forecast_min:.4f} at hour {min_hour}")
        print(f"  Range:         {forecast_max - forecast_min:.4f}")

        # 与历史比较
        hist_mean = np.mean(ot_history)
        hist_std = np.std(ot_history)

        print(f"\n📊 Comparison with Historical:")
        print(f"  Historical mean:     {hist_mean:.4f}")
        print(f"  Historical std dev:  {hist_std:.4f}")
        print(f"  Forecast mean:       {mean_forecast:.4f}")
        print(f"  Difference:          {mean_forecast - hist_mean:+.4f} ({(mean_forecast-hist_mean)/hist_mean*100:+.2f}%)")

        # 预测质量评估
        print(f"\n🎯 Forecast Quality Assessment:")

        # 检查预测是否在合理范围内
        within_1std = np.sum((ot_forecast >= hist_mean - hist_std) &
                            (ot_forecast <= hist_mean + hist_std)) / len(ot_forecast) * 100
        within_2std = np.sum((ot_forecast >= hist_mean - 2*hist_std) &
                            (ot_forecast <= hist_mean + 2*hist_std)) / len(ot_forecast) * 100

        print(f"  {within_1std:.1f}% of forecasts within ±1 std dev of historical mean")
        print(f"  {within_2std:.1f}% of forecasts within ±2 std dev of historical mean")

        if within_1std > 80:
            print(f"  → Forecast appears conservative and well-aligned with historical patterns")
        elif within_1std > 60:
            print(f"  → Forecast shows moderate deviation from historical patterns")
        else:
            print(f"  → Forecast shows significant departure from historical patterns")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("OT COLUMN FORECAST ANALYSIS")
    print("="*80 + "\n")

    # 设置数据目录
    data_dir = os.path.dirname(__file__)

    # 创建分析器
    analyzer = OTColumnAnalyzer(data_dir)

    # 加载数据
    if not analyzer.load_data():
        print("\n✗ Failed to load data. Exiting...")
        return

    # 1. 详细预测可视化
    analyzer.plot_detailed_forecast(num_history=300)

    # 2. 统计分析
    analyzer.analyze_statistics()

    # 3. 不确定性分析
    analyzer.analyze_quantile_uncertainty()

    # 4. 生成报告
    analyzer.generate_forecast_report()

    print("\n" + "="*80)
    print("✓ OT COLUMN ANALYSIS COMPLETE!")
    print("="*80)
    print("\n📁 Generated files:")
    print("  - OT_detailed_forecast.png")
    print("  - OT_uncertainty_evolution.png")
    print("  - OT_statistics.csv")
    print("  - OT_uncertainty_analysis.csv")
    print("\n")


if __name__ == "__main__":
    main()
