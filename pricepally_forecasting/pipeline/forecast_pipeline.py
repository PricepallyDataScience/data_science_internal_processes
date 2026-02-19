import numpy as np
import pandas as pd
import logging
from time import time

from app.config import FORECAST_HORIZON, MIN_XGBOOST_ROWS, INACTIVE_GAP_WEEKS
from app.features.xgboost_features import create_xgboost_features
from app.models.xgboost_model import train_xgboost_model, xgboost_recursive_forecast
from app.models.heuristics import rolling_mean_forecast, exponential_smoothing_forecast, naive_forecast
from app.utils.date_utils import week_month_from_date

# Get logger instance
logger = logging.getLogger("forecast_pipeline")


# ---------------------------------------------------
# Forecast single product + sales_type
# ---------------------------------------------------
def forecast_single_product(ts, model, encoders, product_name, product_uom, sales_type=None):
    """
    Generate forecast for a single product-UOM-sales_type combination.
    Uses XGBoost for products with sufficient history, heuristics otherwise.
    """
    product_df = ts[
        (ts["product_name"] == product_name) &
        (ts["product_uom"] == product_uom)
    ].copy()

    if sales_type:
        product_df = product_df[product_df["sales_type"] == sales_type]

    product_df = product_df.sort_values("date")

    if product_df.empty:
        logger.debug(f"Empty data for product: {product_name} ({product_uom}) - {sales_type}")
        return pd.DataFrame()

    series = product_df["qty_for_forecast"]
    today = pd.Timestamp.now().normalize()

    # ---------- INACTIVE CHECK ----------
    last_nonzero_date = product_df[series > 0]["date"].max()
    if pd.notna(last_nonzero_date):
        gap_weeks = (today - last_nonzero_date).days // 7
    else:
        gap_weeks = np.inf

    if gap_weeks >= INACTIVE_GAP_WEEKS:
        forecast_values = np.zeros(FORECAST_HORIZON)
        method = "ZERO_INACTIVE"
        logger.debug(f"{product_name} ({product_uom}): INACTIVE - {gap_weeks} weeks gap")

    # ---------- XGBOOST ----------
    elif len(product_df.dropna(subset=["lag_1", "lag_4", "lag_8"])) >= MIN_XGBOOST_ROWS:
        try:
            forecast_values = xgboost_recursive_forecast(
                model=model,
                product_df=product_df,
                horizon=FORECAST_HORIZON,
                feature_creator=create_xgboost_features,
                encoders=encoders
            )
            method = "XGBOOST_RECURSIVE"
            logger.debug(f"{product_name} ({product_uom}): Using XGBOOST")
        except Exception as e:
            logger.warning(f"XGBoost failed for {product_name} ({product_uom}): {str(e)}")
            logger.warning(f"Falling back to heuristic for this product")
            # Fall back to heuristic
            forecast_values = rolling_mean_forecast(series, FORECAST_HORIZON, window=4)
            method = "HEURISTIC_ROLLING_MEAN"

    # ---------- HEURISTIC (ADAPTIVE) ----------
    else:
        # Adaptive heuristic selection based on product characteristics
        if len(series) == 0 or series.sum() == 0:
            forecast_values = np.zeros(FORECAST_HORIZON)
            method = "HEURISTIC_ZERO"
        
        elif len(series) == 1:
            forecast_values = naive_forecast(series, FORECAST_HORIZON)
            method = "HEURISTIC_NAIVE"
        
        else:
            mean_val = series.mean()
            std_val = series.std()
            cv = std_val / mean_val if mean_val > 0 else float('inf')
            trend = series.diff().mean() if len(series) > 1 else 0
            
            if cv < 0.3 and abs(trend) < mean_val * 0.1:
                forecast_values = naive_forecast(series, FORECAST_HORIZON)
                method = "HEURISTIC_NAIVE"
            else:
                forecast_values = rolling_mean_forecast(series, FORECAST_HORIZON, window=4)
                method = "HEURISTIC_ROLLING_MEAN"
        
        logger.debug(f"{product_name} ({product_uom}): Using {method}")

    future_dates = pd.date_range(start=today, periods=FORECAST_HORIZON, freq="W-SUN")

    return pd.DataFrame({
        "date": future_dates,
        "forecast_qty": forecast_values,
        "year": future_dates.year,
        "month": future_dates.month,
        "week_month": week_month_from_date(future_dates),
        "product_name": product_name,
        "product_uom": product_uom,
        "sales_type": sales_type if sales_type else "ALL",
        "forecast_method": method
    })


# ---------------------------------------------------
# Run full pipeline
# ---------------------------------------------------
def run_pipeline(ts: pd.DataFrame, log_methods=False):
    """
    Main forecasting pipeline:
    1. Create features
    2. Train global XGBoost model
    3. Generate forecasts for each product-UOM-sales_type
    """
    start_time = time()
    
    # Track all products BEFORE forecasting
    all_products_before = ts[["product_name", "product_uom", "sales_type"]].drop_duplicates()
    total_products_before = len(all_products_before)
    
    logger.info("=" * 60)
    logger.info("FORECAST PIPELINE STARTED")
    logger.info("=" * 60)
    logger.info(f"Total unique products to forecast: {total_products_before:,}")
    
    # ---------- CREATE FEATURES ----------
    logger.info("Creating features (lags, rolling stats, time features)...")
    feature_start = time()
    ts = create_xgboost_features(ts)
    feature_time = time() - feature_start
    logger.info(f"✅ Features created in {feature_time:.2f}s")
    logger.info(f"   Features added: lag_1, lag_4, lag_8, roll_mean_4, roll_mean_8, roll_std_4, month_sin, month_cos")

    # ---------- TRAIN GLOBAL MODEL ----------
    logger.info("Training global XGBoost model...")
    model_start = time()
    try:
        model, encoders = train_xgboost_model(ts)
        model_time = time() - model_start
        logger.info(f"✅ Model trained successfully in {model_time:.2f}s")
        logger.info(f"   Model params: n_estimators=500, max_depth=5, learning_rate=0.05")
    except Exception as e:
        logger.error(f"❌ Model training failed: {str(e)}", exc_info=True)
        logger.error("   Cannot continue without trained model")
        raise

    all_forecasts = []
    failed_products = []

    # Generate forecasts per product + sales_type
    products = ts[["product_name", "product_uom", "sales_type"]].drop_duplicates()

    logger.info(f"Generating forecasts for {len(products):,} products...")
    forecast_start = time()
    
    for idx, row in products.iterrows():
        try:
            forecast_df = forecast_single_product(
                ts,
                model,
                encoders,
                row["product_name"],
                row["product_uom"],
                sales_type=row["sales_type"]
            )
            if not forecast_df.empty:
                all_forecasts.append(forecast_df)
            else:
                failed_products.append({
                    "product_name": row["product_name"],
                    "product_uom": row["product_uom"],
                    "sales_type": row["sales_type"],
                    "reason": "Empty forecast returned"
                })
                logger.debug(f"Empty forecast for {row['product_name']} ({row['product_uom']})")
        except Exception as e:
            failed_products.append({
                "product_name": row["product_name"],
                "product_uom": row["product_uom"],
                "sales_type": row["sales_type"],
                "reason": f"Error: {str(e)}"
            })
            logger.error(f"Failed to forecast {row['product_name']} ({row['product_uom']}): {str(e)}")
        
        # Progress indicator
        if (idx + 1) % 100 == 0:
            elapsed = time() - forecast_start
            rate = (idx + 1) / elapsed
            eta = (len(products) - (idx + 1)) / rate
            logger.info(f"   Progress: {idx + 1:,}/{len(products):,} products ({(idx+1)/len(products)*100:.1f}%) - ETA: {eta:.0f}s")

    forecast_time = time() - forecast_start
    logger.info(f"✅ Forecast generation completed in {forecast_time:.2f}s")
    logger.info(f"   Average: {forecast_time/len(products):.3f}s per product")

    if all_forecasts:
        final_forecast = pd.concat(all_forecasts, ignore_index=True)
        
        # Count products AFTER forecasting
        products_after = final_forecast[["product_name", "product_uom", "sales_type"]].drop_duplicates()
        total_products_after = len(products_after)
        
        # Method distribution
        method_counts = final_forecast.groupby("forecast_method").apply(
            lambda x: x[["product_name", "product_uom", "sales_type"]].drop_duplicates().shape[0]
        )
        
        # Summary
        logger.info("=" * 60)
        logger.info("FORECAST PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total products BEFORE forecast: {total_products_before:,}")
        logger.info(f"Total products AFTER forecast:  {total_products_after:,}")
        logger.info(f"Products with failed forecasts: {len(failed_products):,}")
        logger.info(f"Success rate: {(total_products_after/total_products_before)*100:.1f}%")
        
        logger.info("\nForecast Method Distribution:")
        for method, count in method_counts.sort_values(ascending=False).items():
            percentage = (count / total_products_after) * 100
            logger.info(f"  {method}: {count:,} products ({percentage:.1f}%)")
        
        logger.info(f"\nVerification: {method_counts.sum():,} (should equal {total_products_after:,})")
        
        # Log failed products if any
        if failed_products:
            logger.warning(f"⚠️  {len(failed_products):,} products failed to generate forecasts")
            failed_df = pd.DataFrame(failed_products)
            failed_df.to_csv("failed_forecasts.csv", index=False)
            logger.warning(f"   Saved failure details to 'failed_forecasts.csv'")
            logger.warning(f"   Top failure reasons:")
            reason_counts = failed_df['reason'].value_counts().head(3)
            for reason, count in reason_counts.items():
                logger.warning(f"     - {reason}: {count} products")
        
        # Additional statistics
        logger.info(f"\nForecast Statistics:")
        logger.info(f"  Total forecast rows: {len(final_forecast):,}")
        logger.info(f"  Rows per product: {len(final_forecast)/total_products_after:.1f}")
        logger.info(f"  Date range: {final_forecast['date'].min()} to {final_forecast['date'].max()}")
        logger.info(f"  Mean forecast qty: {final_forecast['forecast_qty'].mean():.2f}")
        logger.info(f"  Median forecast qty: {final_forecast['forecast_qty'].median():.2f}")
        logger.info(f"  Zero forecasts: {(final_forecast['forecast_qty'] == 0).sum():,} ({(final_forecast['forecast_qty'] == 0).sum()/len(final_forecast)*100:.1f}%)")
        
        total_time = time() - start_time
        logger.info(f"\nTotal pipeline time: {total_time:.2f}s")
        logger.info(f"  Feature creation: {feature_time:.2f}s ({feature_time/total_time*100:.1f}%)")
        logger.info(f"  Model training: {model_time:.2f}s ({model_time/total_time*100:.1f}%)")
        logger.info(f"  Forecast generation: {forecast_time:.2f}s ({forecast_time/total_time*100:.1f}%)")
        
        logger.info("=" * 60)
        
        return final_forecast

    logger.error("❌ No forecasts generated!")
    logger.error(f"   All {total_products_before:,} products failed")
    return pd.DataFrame()