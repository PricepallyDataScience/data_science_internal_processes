import numpy as np
import pandas as pd


def rolling_mean_forecast(series: pd.Series, horizon: int, window: int = 4) -> list:
    """
    Simple heuristic forecast using rolling mean.
    Used for products with insufficient historical data for XGBoost.
    
    Args:
        series: Historical quantity series
        horizon: Number of periods to forecast
        window: Rolling window size (default: 4 weeks)
    
    Returns:
        List of forecast values
    """
    # Calculate rolling mean from the last 'window' observations
    if len(series) < window:
        # If not enough data, use overall mean
        forecast_value = series.mean()
    else:
        # Use rolling mean of last 'window' periods
        forecast_value = series.tail(window).mean()
    
    # Handle NaN or negative values
    if pd.isna(forecast_value) or forecast_value < 0:
        forecast_value = 0
    
    # Return constant forecast for all periods
    return [forecast_value] * horizon


def exponential_smoothing_forecast(series: pd.Series, horizon: int, alpha: float = 0.3) -> list:
    """
    Simple exponential smoothing heuristic.
    Alternative to rolling mean for products with trend.
    
    Args:
        series: Historical quantity series
        horizon: Number of periods to forecast
        alpha: Smoothing parameter (0-1)
    
    Returns:
        List of forecast values
    """
    if len(series) == 0:
        return [0] * horizon
    
    # Initialize with first value
    level = series.iloc[0]
    
    # Apply exponential smoothing
    for value in series.iloc[1:]:
        level = alpha * value + (1 - alpha) * level
    
    # Return constant forecast
    forecast_value = max(0, level)  # Ensure non-negative
    return [forecast_value] * horizon


def naive_forecast(series: pd.Series, horizon: int) -> list:
    """
    Naive forecast: uses last observed value.
    Simplest possible heuristic.
    
    Args:
        series: Historical quantity series
        horizon: Number of periods to forecast
    
    Returns:
        List of forecast values
    """
    if len(series) == 0:
        return [0] * horizon
    
    last_value = series.iloc[-1]
    forecast_value = max(0, last_value)  # Ensure non-negative
    
    return [forecast_value] * horizon