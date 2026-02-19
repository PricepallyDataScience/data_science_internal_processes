import pandas as pd

# Mapping week number to the start day of the week
WEEK_START_DAY = {
    1: 1,
    2: 8,
    3: 15,
    4: 22
}

def week_month_to_date(row: pd.Series) -> pd.Timestamp:
    """
    Convert (year, month, week_month) into a representative date
    based on Pricepally 4-week month business calendar.
    """
    try:
        week = int(row["week_month"])
        if week not in WEEK_START_DAY:
            raise ValueError(f"Invalid week number {week}, must be 1-4")
        day = WEEK_START_DAY[week]
        return pd.to_datetime(f"{int(row['year'])}-{int(row['month'])}-{day}")
    except Exception:
        return pd.NaT


def week_month_from_date(dates):
    """
    Convert a date or series of dates into the Pricepally week number (1-4)
    """
    if isinstance(dates, pd.DatetimeIndex):
        dates = pd.Series(dates)
    elif isinstance(dates, pd.Timestamp):
        dates = pd.Series([dates])
    
    return ((dates.dt.day - 1) // 7 + 1).clip(upper=4)
