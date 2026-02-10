import pandas as pd
import logging

# Get logger instance
logger = logging.getLogger("forecast_pipeline")
    
ATTRIBUTE_ONLY_PRODUCTS = {
    "blend",
    "blend crayfish",
    "blend egusi",
    "blend ogbono",
    "chop cabbage",
    "chop carrot",
    "chop spring onions",
    "chop waterleaf",
    "chopped",
    "clean",
    "clean beans",
    "cleaned and dressed",
    "cut",
    "cut (4 pieces)",
    "cut (6 pieces)",
    "cut (big size)",
    "cut (medium size)",
    "cut (nkwobi size)",
    "cut (small size)",
    "cut protein",
    "cut vegetable",
    "descale",
    "descale and cut",
    "deseed tatase",
    "deseeding",
    "diced",
    "grated",
    "green only",
    "grind",
    "grind crayfish",
    "juiced",
    "packaging",
    "peeled",
    "peeled & cut",
    "peeled & pressed",
    "pick",
    "pluck",
    "prep rodo",
    "prep sombo",
    "prepping",
    "processed and cut",
    "processed only",
    "red only",
    "ripe",
    "roasted & cut (big size)",
    "roasted & cut (medium size)",
    "roasted & cut (small size)",
    "scraped & cut (big size)",
    "scraped & cut (medium size)",
    "scraped & cut (small size)",
    "semi ripe",
    "unripe",
    "wash",
    "wash & blend okazi",
    "wash & chop spring onions",
    "washed & blend",
    "washed & chopped",
    "washed & squeezed",
}


def build_weekly_timeseries(df: pd.DataFrame, filter_attribute_products=True) -> pd.DataFrame:
    """
    Aggregate raw data into weekly timeseries for forecasting.
    Optionally filters out ATTRIBUTE_ONLY_PRODUCTS.
    """
    initial_products = df["product_name"].nunique()
    logger.debug(f"Building weekly timeseries from {len(df):,} rows, {initial_products:,} unique products")
    
    # Optional product filtering
    if filter_attribute_products:
        before_filter = len(df)
        df = df[~df["product_name"].str.lower().isin(ATTRIBUTE_ONLY_PRODUCTS)]
        after_filter = len(df)
        filtered_out = before_filter - after_filter
        if filtered_out > 0:
            logger.info(f"Filtered out {filtered_out:,} attribute-only product rows ({filtered_out/before_filter*100:.1f}%)")
            logger.debug(f"Attribute products filtered: {len(ATTRIBUTE_ONLY_PRODUCTS)} product types")

    # Compute quantity for forecast
    df = df.copy()
    df.loc[:, "qty_for_forecast"] = df[["total_qty_invoiced", "total_qty_delivered"]].max(axis=1)
    logger.debug(f"Computed qty_for_forecast as max(invoiced, delivered)")

    # Aggregate to weekly level
    ts = df.groupby(
        ["year", "month", "week_month", "product_name", "product_uom", "sales_type"]
    )["qty_for_forecast"].sum().reset_index()
    
    final_products = ts[["product_name", "product_uom", "sales_type"]].drop_duplicates().shape[0]
    logger.info(f"Weekly timeseries created: {len(ts):,} rows, {final_products:,} unique product-UOM-salestype combinations")

    return ts