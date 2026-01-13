from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch2
from timesfm import configs
import torch
import numpy as np
import pandas as pd
import os

# Load model
print("Loading TimesFM model...")
model = TimesFM_2p5_200M_torch2.from_pretrained("fm_200_models")

# Compile model with forecast configuration
print("Compiling model...")
model.compile(configs.ForecastConfig(
    max_context=1024,
    max_horizon=256,
    normalize_inputs=True,
    use_continuous_quantile_head=True,
    force_flip_invariance=True,
    infer_is_positive=True,
    fix_quantile_crossing=True
))

# Read CSV file
csv_path = os.path.join(os.path.dirname(__file__), "east_settlement.csv")
print(f"Reading CSV file: {csv_path}")
df = pd.read_csv(csv_path)

# Remove date column to get only numeric columns
numeric_cols = df.columns[1:]  # Skip 'date' column
print(f"Found {len(numeric_cols)} columns to forecast: {', '.join(numeric_cols)}")

# Prepare inputs - convert each column to numpy array
inputs = []
for col in numeric_cols:
    # Get the column data and convert to numpy array
    col_data = df[col].values.astype(np.float32)
    inputs.append(col_data)

# Set forecast horizon (predict next 24 hours)
horizon = 96
print(f"\nForecasting {horizon} time steps for each column...")

# Perform forecast
point_forecast, quantile_forecast = model.forecast(
    horizon=horizon,
    inputs=inputs
)

print(f"Point forecast shape: {point_forecast.shape}")
print(f"Quantile forecast shape: {quantile_forecast.shape}")

# Save results
output_dir = os.path.dirname(__file__)

# Generate forecast time column
# Get the last date from historical data
last_date = pd.to_datetime(df['date'].iloc[-1])
print(f"\nLast historical date: {last_date}")

# Generate future dates (assuming hourly frequency)
forecast_dates = pd.date_range(
    start=last_date + pd.Timedelta(hours=1),
    periods=horizon,
    freq='H'
)
print(f"Forecast date range: {forecast_dates[0]} to {forecast_dates[-1]}")

# Save point forecasts with time column
point_forecast_df = pd.DataFrame(
    point_forecast.T,
    columns=numeric_cols
)
# Insert date column as the first column
point_forecast_df.insert(0, 'date', forecast_dates)

point_forecast_path = os.path.join(output_dir, "east_settlement_point_forecast.csv")
point_forecast_df.to_csv(point_forecast_path, index=False)
print(f"\nPoint forecasts saved to: {point_forecast_path}")

# Save quantile forecasts (shape: [num_series, horizon, num_quantiles])
# We'll save each quantile level as a separate file or in a structured format
quantile_forecast_path = os.path.join(output_dir, "east_settlement_quantile_forecast.npy")
np.save(quantile_forecast_path, quantile_forecast)
print(f"Quantile forecasts saved to: {quantile_forecast_path}")

print("\nForecast complete!")
print(f"Forecasted {len(numeric_cols)} time series for {horizon} steps ahead.")





