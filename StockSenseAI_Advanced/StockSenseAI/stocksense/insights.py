"""Rule-based insight engine: turns the numbers into plain-language actions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _inr(x: float) -> str:
    return f"₹{x:,.0f}"


def generate_insights(df: pd.DataFrame, inv: pd.DataFrame, horizon: int) -> list[dict]:
    out: list[dict] = []

    # 1. Imminent stockouts (sorted by probability)
    hi = inv[inv.stockout_risk == "High"].sort_values("stockout_probability", ascending=False)
    for _, r in hi.head(3).iterrows():
        day = r.projected_stockout_day
        when = f"in about {day} day{'s' if day != 1 else ''}" if day else "within the lead time"
        out.append({
            "level": "critical", "icon": "🔴",
            "title": f"{r.product_name} will likely run out {when}",
            "text": (f"{r.stockout_probability:.0%} chance of a stockout before a new delivery could arrive "
                     f"(supplier lead time {r.lead_time_days:g} d). Order {r.recommended_reorder_qty:,} units "
                     f"({_inr(r.estimated_reorder_cost)}) today."),
        })

    # 2. Expiry waste
    w = inv[inv.expected_waste_value > 0].sort_values("expected_waste_value", ascending=False)
    if len(w):
        top = w.iloc[0]
        out.append({
            "level": "serious", "icon": "🟠",
            "title": f"{_inr(w.expected_waste_value.sum())} of stock is forecast to expire unsold",
            "text": (f"Biggest exposure: {top.product_name} — {top.expected_waste_units:,.0f} units "
                     f"won't sell in the {max(top.days_to_expiry, 0):.0f} days before expiry. Suggested action: "
                     f"{top.expiry_action.lower()}."),
        })

    # 3. Category momentum: last 7 days vs previous 7 days
    last = df.date.max()
    cur = df[df.date > last - pd.Timedelta(days=7)].groupby("category").units_sold.sum()
    prev = df[(df.date <= last - pd.Timedelta(days=7)) & (df.date > last - pd.Timedelta(days=14))] \
        .groupby("category").units_sold.sum()
    chg = ((cur - prev) / prev.replace(0, np.nan) * 100).dropna()
    if len(chg):
        up, down = chg.idxmax(), chg.idxmin()
        if chg[up] > 3:
            out.append({"level": "info", "icon": "📈", "title": f"{up} demand is up {chg[up]:.0f}% week-on-week",
                        "text": "Check that safety stock for this category keeps up with the new run-rate."})
        if chg[down] < -3:
            out.append({"level": "info", "icon": "📉", "title": f"{down} demand is down {abs(chg[down]):.0f}% week-on-week",
                        "text": "Consider smaller, more frequent orders until demand recovers."})

    # 4. Weekend effect
    daily = df.groupby("date").units_sold.sum()
    wk = daily[daily.index.dayofweek >= 5].mean()
    wd = daily[daily.index.dayofweek < 5].mean()
    if wd > 0 and abs(wk / wd - 1) > 0.05:
        diff = (wk / wd - 1) * 100
        out.append({"level": "info", "icon": "🗓️",
                    "title": f"Weekends sell {abs(diff):.0f}% {'more' if diff > 0 else 'less'} than weekdays",
                    "text": "Schedule deliveries for Thursday/Friday so shelves are full before the weekend."
                    if diff > 0 else "Weekday deliveries are sufficient; avoid weekend restocking costs."})

    # 5. Overstock / tied-up capital
    o = inv[inv.overstock]
    if len(o):
        out.append({"level": "warning", "icon": "🟡",
                    "title": f"{len(o)} product{'s' if len(o) > 1 else ''} overstocked — {_inr(o.stock_value.sum())} tied up",
                    "text": "Pause reorders for: " + ", ".join(o.sort_values('stock_value', ascending=False).product_name.head(4)) + "."})

    # 6. Pareto
    a = inv[inv.abc == "A"]
    if len(a) and inv.annual_value.sum() > 0:
        share = a.annual_value.sum() / inv.annual_value.sum() * 100
        out.append({"level": "info", "icon": "🎯",
                    "title": f"{len(a)} of {len(inv)} products drive {share:.0f}% of inventory spend",
                    "text": "These A-class items deserve the tightest monitoring: " + ", ".join(a.product_name.head(5)) + "."})

    # 7. Forecast quality
    wape = inv.backtest_wape.dropna()
    if len(wape):
        out.append({"level": "good", "icon": "✅",
                    "title": f"Forecasts are within {wape.median():.0f}% of actual sales (median backtest error)",
                    "text": f"Each product's model was selected by testing on its last 14 days of real sales; "
                            f"next {horizon}-day demand totals {inv.avg_daily_forecast.sum() * horizon:,.0f} units."})
    if not len(hi):
        out.insert(0, {"level": "good", "icon": "🟢", "title": "No product is at high stockout risk",
                       "text": "Current stock covers every supplier lead time at the chosen service level."})
    return out
