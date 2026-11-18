"""Inventory policy engine.

For every product this computes, from its demand forecast:
  * service-level safety stock     SS  = z · σ · √LT
  * reorder point                  s   = μ · LT + SS
  * economic order quantity        EOQ = √(2·D·K / (h·c))
  * (s, S) order-up-to level       S   = s + EOQ, clipped to [review period, max cover] of demand
  * shelf-life cap                 never order more than can be sold before it expires
  * Monte Carlo stockout risk      P(demand during lead time > stock on hand)
  * expiry waste forecast          stock that will not sell before its expiry date
  * ABC-XYZ class                  value contribution × demand variability
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

ABC_POLICY = {
    "AX": "Automate: tight safety stock, frequent small orders",
    "AY": "Review weekly; buffer for swings in demand",
    "AZ": "High value + erratic: monitor daily, order to forecast",
    "BX": "Standard reorder-point policy",
    "BY": "Standard policy with moderate buffer",
    "BZ": "Order on demand signal; avoid large lots",
    "CX": "Bulk order infrequently to save ordering cost",
    "CY": "Periodic bulk review",
    "CZ": "Candidate for delisting or make-to-order",
}


@dataclass
class Policy:
    service_level: float = 0.95
    review_days: int = 7
    order_cost: float = 150.0          # ₹ per purchase order
    max_cover_days: int = 14           # never order more than this many days of demand at once
    holding_rate: float = 0.20         # annual holding cost as a share of unit cost
    demand_mult: float = 1.0           # what-if: demand shock
    lead_time_add: float = 0.0         # what-if: supplier delay (days)
    elasticity: float = 1.8            # % sales uplift per 1% markdown (assumption)
    n_sims: int = 5000
    seed: int = 7


def shelf_life_days(df: pd.DataFrame) -> pd.Series:
    sl = (df["expiry_date"] - df["date"]).dt.days
    return sl[sl > 0].groupby(df["product_id"]).median()


def abc_xyz(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["product_id", "date"])["units_sold"].sum().reset_index()
    stats = g.groupby("product_id")["units_sold"].agg(["mean", "std"])
    stats["cv"] = (stats["std"] / stats["mean"].replace(0, np.nan)).fillna(9)
    cost = df.sort_values("date").groupby("product_id")["unit_cost"].last()
    stats["annual_value"] = stats["mean"] * 365 * cost
    stats = stats.sort_values("annual_value", ascending=False)
    total = stats["annual_value"].sum() or 1
    stats["cum_share"] = stats["annual_value"].cumsum() / total
    prev = stats["cum_share"].shift(fill_value=0)
    stats["abc"] = np.where(prev < 0.80, "A", np.where(prev < 0.95, "B", "C"))
    stats["xyz"] = np.where(stats["cv"] < 0.25, "X", np.where(stats["cv"] < 0.50, "Y", "Z"))
    stats["abc_xyz"] = stats["abc"] + stats["xyz"]
    return stats.reset_index()[["product_id", "annual_value", "cum_share", "cv", "abc", "xyz", "abc_xyz"]]


def _simulate(paths_mean: np.ndarray, sigma: float, stock: float, days: int, rng, n: int):
    """Simulate `n` demand paths over `days`; return P(stockout) and median stockout day."""
    days = max(1, int(np.ceil(days)))
    mu = np.resize(paths_mean, days) if len(paths_mean) else np.zeros(days)
    sims = np.maximum(0, rng.normal(mu, max(sigma, 1e-6), size=(n, days)))
    cum = sims.cumsum(axis=1)
    out = cum > stock
    p = float(out[:, -1].mean())
    first = np.where(out.any(axis=1), out.argmax(axis=1) + 1, np.nan)
    med = float(np.nanmedian(first)) if np.isfinite(first).any() else np.nan
    return p, med, cum[:, -1]


def compute_inventory(df: pd.DataFrame, forecasts: dict, policy: Policy) -> pd.DataFrame:
    as_of = df["date"].max()
    latest = df.sort_values("date").groupby("product_id").tail(1).set_index("product_id")
    shelf = shelf_life_days(df)
    classes = abc_xyz(df).set_index("product_id")
    z = float(norm.ppf(policy.service_level))
    rng = np.random.default_rng(policy.seed)
    target_risk = 1 - policy.service_level

    rows = []
    for pid, row in latest.iterrows():
        res = forecasts[pid]
        path = res.forecast["predicted_demand"].to_numpy(float) * policy.demand_mult
        mu = float(path.mean()) if len(path) else 0.0
        sigma = res.residual_sigma * policy.demand_mult
        lt = float(row.lead_time_days) + policy.lead_time_add
        stock = float(row.stock_on_hand)
        cost = float(row.unit_cost)

        ss = z * sigma * np.sqrt(lt)
        rop = mu * lt + ss
        annual_d = mu * 365
        h = policy.holding_rate * cost
        eoq = np.sqrt(2 * annual_d * policy.order_cost / h) if h > 0 and annual_d > 0 else mu * policy.review_days
        lot = float(np.clip(eoq, mu * policy.review_days, mu * max(policy.max_cover_days, policy.review_days)))
        order_up_to = rop + lot

        life = float(shelf.get(pid, np.nan))
        shelf_cap = mu * life * 0.9 if np.isfinite(life) else np.inf  # sell 90% of a batch's life
        raw_qty = max(0.0, order_up_to - stock) if stock <= rop else 0.0
        qty = int(np.ceil(min(raw_qty, max(0.0, shelf_cap - stock)))) if raw_qty > 0 else 0
        if raw_qty > 0 and qty == 0:
            qty = int(np.ceil(max(0.0, rop - stock)))  # at least cover the reorder gap
        capped = bool(raw_qty > 0 and np.isfinite(shelf_cap) and raw_qty > shelf_cap - stock)

        p_out, med_day, _ = _simulate(path, sigma, stock, lt, rng, policy.n_sims)
        if stock <= 0:
            p_out, med_day = 1.0, 0.0
        cover = stock / mu if mu > 0 else np.inf
        cum = np.cumsum(np.resize(path, 120)) if mu > 0 else np.zeros(120)
        stockout_day = int(np.argmax(cum > stock) + 1) if mu > 0 and (cum > stock).any() else None

        risk = "High" if p_out >= 0.5 else ("Medium" if p_out > target_risk else "Low")

        exp_date = row.expiry_date
        dte = (exp_date - as_of).days if pd.notna(exp_date) else np.nan
        if np.isnan(dte):
            sell_before, waste = np.nan, 0.0
        elif dte < 0:
            sell_before, waste = 0.0, stock
        else:
            sell_before = float(np.resize(path, max(int(dte), 1))[: int(dte)].sum()) if dte > 0 else 0.0
            waste = max(0.0, stock - sell_before)
        if waste > 0 and sell_before and sell_before > 0:
            uplift = waste / sell_before
            disc = uplift / policy.elasticity
            action = f"Mark down ~{min(50, max(5, round(disc * 100 / 5) * 5)):.0f}%" if disc <= 0.5 else "Liquidate / donate"
        elif waste > 0:
            action = "Remove from shelf" if dte < 0 else "Liquidate / donate"
        else:
            action = "—"

        cls = classes.loc[pid] if pid in classes.index else None
        overstock = bool(cover > max(45, 4 * (lt + policy.review_days)) or
                         (np.isfinite(dte) and dte >= 0 and cover > dte * 1.2 and waste > 0))
        weight = {"A": 3, "B": 2, "C": 1}.get(cls["abc"] if cls is not None else "C", 1)
        priority = p_out * mu * max(cost, 1) * weight

        rows.append({
            "product_id": pid, "product_name": row.product_name, "category": row.category,
            "stock_on_hand": stock, "unit_cost": cost, "stock_value": stock * cost,
            "avg_daily_forecast": round(mu, 1), "demand_sigma": round(sigma, 2),
            "lead_time_days": lt, "days_of_cover": round(cover, 1) if np.isfinite(cover) else np.inf,
            "projected_stockout_day": stockout_day,
            "safety_stock": int(np.ceil(ss)), "reorder_point": int(np.ceil(rop)),
            "eoq": int(np.ceil(eoq)), "order_up_to": int(np.ceil(order_up_to)),
            "recommended_reorder_qty": qty, "shelf_life_capped": capped,
            "estimated_reorder_cost": round(qty * cost, 2),
            "stockout_probability": round(p_out, 4), "median_stockout_day": med_day,
            "stockout_risk": risk, "expiry_date": exp_date, "days_to_expiry": dte,
            "shelf_life_days": life, "sellable_before_expiry": sell_before,
            "expected_waste_units": round(waste, 1), "expected_waste_value": round(waste * cost, 2),
            "expiry_action": action, "overstock": overstock,
            "abc": cls["abc"] if cls is not None else "C", "xyz": cls["xyz"] if cls is not None else "Z",
            "abc_xyz": cls["abc_xyz"] if cls is not None else "CZ",
            "cv": cls["cv"] if cls is not None else np.nan,
            "annual_value": cls["annual_value"] if cls is not None else 0.0,
            "policy": ABC_POLICY.get(cls["abc_xyz"] if cls is not None else "CZ", ""),
            "priority_score": priority, "forecast_method": res.method,
            "backtest_wape": (float(res.leaderboard.loc[res.leaderboard.Model == res.method, "WAPE"].iloc[0])
                              if not res.leaderboard.empty and (res.leaderboard.Model == res.method).any() else np.nan),
        })
    return pd.DataFrame(rows)


def health_score(inv: pd.DataFrame) -> tuple[float, dict]:
    """0–100 composite score: service (50%), waste avoidance (30%), capital efficiency (20%)."""
    n = max(len(inv), 1)
    service = (inv["stockout_risk"] == "Low").sum() / n
    value = inv["stock_value"].sum() or 1
    waste = 1 - min(1.0, inv["expected_waste_value"].sum() / value)
    capital = 1 - inv["overstock"].sum() / n
    parts = {"Service level": service * 100, "Waste avoidance": waste * 100, "Capital efficiency": capital * 100}
    return 0.5 * parts["Service level"] + 0.3 * parts["Waste avoidance"] + 0.2 * parts["Capital efficiency"], parts


def allocate_budget(inv: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Greedy budget allocation: highest-risk, highest-value items are funded first."""
    need = inv[inv["recommended_reorder_qty"] > 0].copy()
    rank = {"High": 0, "Medium": 1, "Low": 2}
    need["_r"] = need["stockout_risk"].map(rank)
    need = need.sort_values(["_r", "priority_score"], ascending=[True, False])
    remaining = budget
    funded = []
    for _, r in need.iterrows():
        if r.unit_cost <= 0:
            q = r.recommended_reorder_qty
        else:
            q = int(min(r.recommended_reorder_qty, remaining // r.unit_cost))
        funded.append(q)
        remaining -= q * r.unit_cost
    need["funded_qty"] = funded
    need["funded_cost"] = need["funded_qty"] * need["unit_cost"]
    need["status"] = np.where(need["funded_qty"] >= need["recommended_reorder_qty"], "Fully funded",
                              np.where(need["funded_qty"] > 0, "Partially funded", "Deferred"))
    return need.drop(columns="_r")


def depletion_paths(res, stock: float, lead_time: float, policy: Policy, days: int = 30, n: int = 400):
    """Simulated stock-on-hand trajectories (no replenishment) for fan-chart display."""
    rng = np.random.default_rng(policy.seed)
    path = res.forecast["predicted_demand"].to_numpy(float) * policy.demand_mult
    mu = np.resize(path, days)
    sims = np.maximum(0, rng.normal(mu, max(res.residual_sigma * policy.demand_mult, 1e-6), size=(n, days)))
    level = stock - sims.cumsum(axis=1)
    pct = np.percentile(level, [5, 25, 50, 75, 95], axis=0)
    pct = np.hstack([np.full((5, 1), stock), pct])  # start from today's stock
    dates = pd.date_range(res.history["date"].max(), periods=days + 1)
    return pd.DataFrame({"date": dates, "p5": pct[0], "p25": pct[1], "p50": pct[2], "p75": pct[3], "p95": pct[4]})
