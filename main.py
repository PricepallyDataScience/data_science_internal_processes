import pandas as pd
import logging
from app.data.preprocessing import build_weekly_timeseries
from app.utils.date_utils import week_month_to_date
from app.pipeline.forecast_pipeline import run_pipeline
from app.utils.logger import setup_logging, log_dataframe_info


def main():
    """
    Main script to run XGBoost-based demand forecasting pipeline.
    Production version with comprehensive logging for AWS deployment.
    
    Steps:
    1. Load raw transaction data
    2. Filter by sales type (optional)
    3. Build weekly time series
    4. Add date column for time-based features
    5. Run forecast pipeline (XGBoost + Heuristics)
    6. Save forecast output
    """
    
    # Initialize logging
    logger = setup_logging(log_level=logging.INFO, log_dir="logs")
    
    logger.info("=" * 70)
    logger.info("STARTING PRICEPALLY DEMAND FORECASTING PIPELINE")
    logger.info("=" * 70)
    
    try:
        # 1️⃣ Load raw CSV
        logger.info("Step 1: Loading raw data from CSV...")
        df = pd.read_csv("forecast_date_1.csv")
        logger.info(f"✅ Loaded {len(df):,} rows from forecast_date_1.csv")
        log_dataframe_info(logger, df, "Raw Input Data")

        # 2️⃣ Filter by sales type
        sales_type_filter = ["b2c"]
        logger.info(f"Step 2: Filtering for sales types: {sales_type_filter}")
        initial_rows = len(df)
        df = df[df["sales_type"].str.lower().isin(sales_type_filter)]
        filtered_rows = len(df)
        logger.info(f"✅ Filtered to {filtered_rows:,} rows ({filtered_rows/initial_rows*100:.1f}% retained)")
        logger.info(f"   Removed {initial_rows - filtered_rows:,} rows")

        # 3️⃣ Build weekly timeseries
        logger.info("Step 3: Building weekly time series...")
        logger.info("   Filtering attribute-only products...")
        ts = build_weekly_timeseries(df, filter_attribute_products=True)
        logger.info(f"✅ Created {len(ts):,} weekly observations")
        log_dataframe_info(logger, ts, "Weekly Timeseries")
        
        # Log unique products
        unique_products = ts[["product_name", "product_uom", "sales_type"]].drop_duplicates()
        logger.info(f"   Unique product-UOM-salestype combinations: {len(unique_products):,}")

        # 4️⃣ Add date column
        logger.info("Step 4: Adding date column...")
        ts["date"] = ts.apply(week_month_to_date, axis=1)
        ts = ts.sort_values("date")
        date_min, date_max = ts["date"].min(), ts["date"].max()
        logger.info(f"✅ Date range: {date_min} to {date_max}")
        logger.info(f"   Total weeks span: {(date_max - date_min).days // 7} weeks")

        # 5️⃣ Run forecast pipeline
        logger.info("Step 5: Running forecast pipeline...")
        logger.info("   Pipeline steps:")
        logger.info("   - Feature engineering (lags, rolling stats, time features)")
        logger.info("   - XGBoost model training")
        logger.info("   - Forecast generation per product")
        logger.info("   - Adaptive heuristics for limited data products")
        
        forecast = run_pipeline(ts, log_methods=True)
        
        if forecast.empty:
            logger.error("❌ Forecast pipeline returned empty DataFrame!")
            raise ValueError("No forecasts generated - check data quality and pipeline logs")
        
        logger.info(f"✅ Generated {len(forecast):,} forecast rows")

        # 6️⃣ Save forecast output
        logger.info("Step 6: Saving forecast output...")
        output_file = "forecast_output.csv"
        forecast.to_csv(output_file, float_format="%.1f", index=False)
        logger.info(f"✅ Saved forecast to '{output_file}'")
        
        # Log output statistics
        logger.info("\nOutput Statistics:")
        logger.info(f"   Forecast rows: {len(forecast):,}")
        unique_forecasted = forecast[['product_name', 'product_uom', 'sales_type']].drop_duplicates()
        logger.info(f"   Unique products forecasted: {len(unique_forecasted):,}")
        logger.info(f"   Forecast methods used: {forecast['forecast_method'].nunique()}")
        logger.info(f"   Forecast horizon: {len(forecast) // len(unique_forecasted)} weeks per product")
        
        # Method breakdown
        logger.info("\nForecast Method Distribution:")
        method_counts = forecast.groupby("forecast_method").apply(
            lambda x: x[["product_name", "product_uom", "sales_type"]].drop_duplicates().shape[0]
        )
        for method, count in method_counts.sort_values(ascending=False).items():
            pct = (count / len(unique_forecasted)) * 100
            logger.info(f"   {method}: {count:,} products ({pct:.1f}%)")
        
        # SUCCESS
        logger.info("=" * 70)
        logger.info("✅ FORECAST PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        
        return forecast
        
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {str(e)}", exc_info=True)
        logger.error("   Ensure 'forecast_date_1.csv' exists in current directory")
        raise
        
    except KeyError as e:
        logger.error(f"❌ Missing required column: {str(e)}", exc_info=True)
        logger.error("   Check CSV file structure and column names")
        raise
        
    except ValueError as e:
        logger.error(f"❌ Value error: {str(e)}", exc_info=True)
        raise
        
    except Exception as e:
        logger.error("=" * 70)
        logger.error("❌ FORECAST PIPELINE FAILED!")
        logger.error("=" * 70)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        raise
    
    finally:
        logger.info("Pipeline execution finished")
        logger.info(f"Check logs in 'logs/' directory for details")


if __name__ == "__main__":
    main()