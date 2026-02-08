from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import logging

# Get logger instance
logger = logging.getLogger("forecast_pipeline")

FEATURES = [
    "lag_1", "lag_4", "lag_8",  # REMOVED lag_12
    "roll_mean_4", "roll_mean_8", "roll_std_4",
    "month_sin", "month_cos",
    "product_name_encoded", "product_uom_encoded", "sales_type_encoded"
]

CAT_COLS = ["product_name", "product_uom", "sales_type"]


def encode_categorical_features(df, encoders=None):
    """
    Encode categorical features for XGBoost.
    Returns encoded dataframe and encoders dictionary.
    """
    df = df.copy()
    
    if encoders is None:
        encoders = {}
        for col in CAT_COLS:
            if col in df.columns:
                unique_vals = df[col].unique()
                encoders[col] = {val: idx for idx, val in enumerate(unique_vals)}
                logger.debug(f"Created encoder for {col}: {len(unique_vals)} unique values")
    
    for col in CAT_COLS:
        if col in df.columns:
            df[f"{col}_encoded"] = df[col].map(encoders[col]).fillna(-1).astype(int)
    
    return df, encoders


def train_xgboost_model(train_df):
    """
    Train XGBoost model on historical data.
    Uses log transformation for the target variable.
    """
    logger.debug(f"Training XGBoost - Input shape: {train_df.shape}")
    
    # Encode categorical features
    train_df, encoders = encode_categorical_features(train_df)
    logger.debug(f"Categorical encoding complete")
    
    # Apply log transformation to target
    train_df = train_df.copy()
    train_df["y"] = np.log1p(train_df["qty_for_forecast"])
    logger.debug(f"Log transformation applied to target variable")
    
    # Drop rows with missing values in features or target
    before_drop = len(train_df)
    train_df = train_df.dropna(subset=FEATURES + ["y"])
    after_drop = len(train_df)
    dropped = before_drop - after_drop
    
    if dropped > 0:
        logger.warning(f"Dropped {dropped:,} rows ({dropped/before_drop*100:.1f}%) with missing features")
        logger.warning(f"Training data reduced to {after_drop:,} rows")
    
    if after_drop == 0:
        logger.error("❌ No training data remaining after dropping NaN values!")
        logger.error(f"   Check data quality and feature generation")
        raise ValueError("Insufficient training data after NaN removal")

    try:
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=1
        )

        logger.debug(f"Fitting XGBoost model on {len(train_df):,} samples with {len(FEATURES)} features")
        
        model.fit(
            train_df[FEATURES],
            train_df["y"],
            verbose=False
        )
        
        logger.debug(f"Model training successful")
        logger.debug(f"Features used: {', '.join(FEATURES)}")
        
        return model, encoders
        
    except Exception as e:
        logger.error(f"❌ XGBoost training failed: {str(e)}", exc_info=True)
        raise


# ---------------------------------------------------
# RECURSIVE FORECAST
# ---------------------------------------------------

def xgboost_recursive_forecast(model, product_df, horizon, feature_creator, encoders):
    """
    Generate recursive forecasts using XGBoost model.
    Each prediction becomes input for the next step.
    Model predicts in log space, so we apply expm1 to get actual quantities.
    """
    logger.debug(f"Starting recursive forecast for {horizon} periods")
    
    df = product_df.copy()
    
    # Apply log transformation to existing data
    df["y"] = np.log1p(df["qty_for_forecast"])
    
    forecasts = []

    for step in range(horizon):
        try:
            # Recreate features after appending new rows
            df = feature_creator(df)
            
            # Encode categorical features
            df, _ = encode_categorical_features(df, encoders)

            last_row = df.iloc[-1:]

            # Predict in log space
            pred_log = model.predict(last_row[FEATURES])[0]
            
            # Transform back to original scale
            pred = np.expm1(pred_log)
            pred = max(pred, 0)  # Ensure non-negative

            # Build next future row
            next_row = last_row.copy()

            next_row["date"] = next_row["date"] + pd.Timedelta(weeks=1)
            next_row["qty_for_forecast"] = pred
            next_row["y"] = pred_log  # Store in log space for next iteration

            df = pd.concat([df, next_row], ignore_index=True)

            forecasts.append(pred)
            
        except Exception as e:
            logger.error(f"Recursive forecast failed at step {step+1}/{horizon}: {str(e)}")
            # Fill remaining with zeros
            forecasts.extend([0.0] * (horizon - step))
            break

    logger.debug(f"Recursive forecast complete: {len(forecasts)} values generated")
    return forecasts