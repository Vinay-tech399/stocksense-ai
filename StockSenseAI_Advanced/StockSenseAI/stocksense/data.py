"""Data loading, validation and cleaning."""
from __future__ import annotations

import pandas as pd

REQUIRED = [
    "date", "product_id", "product_name", "category", "units_sold",
    "stock_on_hand", "unit_cost", "lead_time_days", "expiry_date",
]

# Common alternative column names -> canonical names
ALIASES = {
    "sale_date": "date", "sales_date": "date", "day": "date",
    "sku": "product_id", "item_id": "product_id", "product": "product_name",
    "item_name": "product_name", "qty_sold": "units_sold", "quantity_sold": "units_sold",
    "sales_qty": "units_sold", "stock": "stock_on_hand", "on_hand": "stock_on_hand",
    "cost": "unit_cost", "lead_time": "lead_time_days", "expiry": "expiry_date",
}


def normalize_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean an uploaded dataframe. Returns (clean_df, data_quality_report)."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={k: v for k, v in ALIASES.items() if k in df.columns and v not in df.columns})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    raw_rows = len(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")
    for c in ["units_sold", "stock_on_hand", "unit_cost", "lead_time_days"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    bad_dates = int(df["date"].isna().sum())
    df = df.dropna(subset=["date", "product_id", "product_name", "units_sold", "stock_on_hand"])
    df["product_id"] = df["product_id"].astype(str)
    df["product_name"] = df["product_name"].astype(str)
    df["category"] = df["category"].fillna("Other").astype(str)
    negatives = int((df["units_sold"] < 0).sum())
    df["units_sold"] = df["units_sold"].clip(lower=0)
    df["stock_on_hand"] = df["stock_on_hand"].clip(lower=0)
    df["unit_cost"] = df["unit_cost"].fillna(0).clip(lower=0)
    df["lead_time_days"] = df["lead_time_days"].fillna(3).clip(lower=1)
    dupes = int(df.duplicated(subset=["date", "product_id"]).sum())

    report = {
        "raw_rows": raw_rows,
        "clean_rows": len(df),
        "dropped_rows": raw_rows - len(df),
        "invalid_dates": bad_dates,
        "negative_sales_clipped": negatives,
        "duplicate_product_days": dupes,
        "products": df["product_id"].nunique(),
        "categories": df["category"].nunique(),
        "days": df["date"].nunique(),
    }
    return df.sort_values("date").reset_index(drop=True), report


def daily_series(hist: pd.DataFrame) -> pd.DataFrame:
    """Daily total units for one product, with missing dates filled as zero sales."""
    daily = hist.groupby("date", as_index=False)["units_sold"].sum()
    idx = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    return (daily.set_index("date").reindex(idx, fill_value=0)
            .rename_axis("date").reset_index())
