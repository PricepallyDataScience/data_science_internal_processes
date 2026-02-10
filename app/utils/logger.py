"""
Production Logging Configuration for Pricepally Forecasting Pipeline
AWS Container Deployment
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(log_level=logging.INFO, log_dir="logs"):
    """
    Configure logging for production deployment on AWS.
    
    Creates both file and console handlers with proper formatting.
    Logs are saved with timestamps for tracking and debugging.
    Console logs go to stdout for CloudWatch capture.
    
    Parameters
    ----------
    log_level : int
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_dir : str
        Directory to save log files
    
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("forecast_pipeline")
    logger.setLevel(log_level)
    
    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler - detailed logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        f"{log_dir}/forecast_{timestamp}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler - for CloudWatch
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Error file handler
    error_handler = logging.FileHandler(
        f"{log_dir}/forecast_errors_{timestamp}.log",
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(error_handler)
    
    # Log initial setup
    logger.info("=" * 70)
    logger.info("PRICEPALLY FORECAST PIPELINE - LOGGING INITIALIZED")
    logger.info(f"Log Level: {logging.getLevelName(log_level)}")
    logger.info(f"Log Directory: {log_dir}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info("=" * 70)
    
    return logger


def log_dataframe_info(logger, df, name="DataFrame"):
    """Log useful DataFrame information"""
    logger.info(f"{name} Info:")
    logger.info(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    logger.info(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    missing = df.isnull().sum()
    if missing.any():
        logger.warning(f"  Missing values in {name}:")
        for col, count in missing[missing > 0].items():
            logger.warning(f"    {col}: {count:,} ({count/len(df)*100:.1f}%)")