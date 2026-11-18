"""Demand forecasting: feature engineering, model zoo, walk-forward backtest, intervals.

Every product gets four candidate models. Each is backtested on the most recent
`test_days` of history (trained only on the data before it) and the model with the
lowest WAPE wins ("Auto" mode). Prediction intervals come from the bias and
spread of that model's backtest errors, so they reflect how wrong the model
actually was on this product, not a textbook assumption.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from .data import daily_series

MODELS = ["Random Forest", "Gradient Boosting", "Weekday average", "Seasonal naive"]
ML_MODELS = {"Random Forest", "Gradient Boosting"}
LAGS = [1, 7, 14]
FEATURES = ["day_of_week", "weekend", "day_of_month", "month", "trend",
            "lag_1", "lag_7", "lag_14", "roll_mean_7", "roll_mean_28", "roll_std_7"]
FEATURE_LABELS = {
    "day_of_week": "Day of week", "weekend": "Weekend flag", "day_of_month": "Day of month",
    "month": "Month", "trend": "Long-run trend", "lag_1": "Yesterday's sales",
    "lag_7": "Sales 7 days ago", "lag_14": "Sales 14 days ago",
    "roll_mean_7": "7-day average", "roll_mean_28": "28-day average",
    "roll_std_7": "7-day volatility",
}
MIN_ML_DAYS = 56


@dataclass
class ForecastResult:
    forecast: pd.DataFrame            # date, predicted_demand, lo80, hi80, lo95, hi95
    method: str
    history: pd.DataFrame             # date, units_sold
    backtest: pd.DataFrame            # date, actual, predicted  (for the chosen model)
    leaderboard: pd.DataFrame         # model, MAE, WAPE, Bias, skill vs naive
    residual_sigma: float
    importances: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------- features
def _row_features(values: list[float], d: pd.Timestamp, t: int) -> dict:
    """Features for the day at position len(values) given past `values`."""
    arr = np.asarray(values, dtype=float)
    def lag(k):
        return arr[-k] if len(arr) >= k else (arr.mean() if len(arr) else 0.0)
    last7 = arr[-7:] if len(arr) else np.array([0.0])
    last28 = arr[-28:] if len(arr) else np.array([0.0])
    return {
        "day_of_week": d.dayofweek, "weekend": int(d.dayofweek >= 5),
        "day_of_month": d.day, "month": d.month, "trend": t,
        "lag_1": lag(1), "lag_7": lag(7), "lag_14": lag(14),
        "roll_mean_7": last7.mean(), "roll_mean_28": last28.mean(),
        "roll_std_7": last7.std(ddof=1) if len(last7) > 1 else 0.0,
    }


def _training_frame(daily: pd.DataFrame) -> pd.DataFrame:
    y = daily["units_sold"].astype(float)
    x = pd.DataFrame({"date": daily["date"], "y": y})
    x["day_of_week"] = x["date"].dt.dayofweek
    x["weekend"] = (x["day_of_week"] >= 5).astype(int)
    x["day_of_month"] = x["date"].dt.day
    x["month"] = x["date"].dt.month
    x["trend"] = np.arange(len(x))
    for k in LAGS:
        x[f"lag_{k}"] = y.shift(k)
    x["roll_mean_7"] = y.shift(1).rolling(7).mean()
    x["roll_mean_28"] = y.shift(1).rolling(28, min_periods=7).mean()
    x["roll_std_7"] = y.shift(1).rolling(7).std()
    return x.dropna().reset_index(drop=True)


def _make_model(name: str):
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=120, max_depth=10, min_samples_leaf=2,
                                     random_state=42, n_jobs=-1)
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_depth=4,
                                         min_samples_leaf=8, l2_regularization=0.5,
                                         random_state=42)


# ---------------------------------------------------------------- predictors
def _predict(name: str, daily: pd.DataFrame, horizon: int):
    """Fit `name` on `daily` and forecast `horizon` days ahead. Returns (preds, model)."""
    values = daily["units_sold"].astype(float).tolist()
    last_date = daily["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon)

    if name == "Seasonal naive":
        hist = values[:]
        out = []
        for _ in range(horizon):
            v = hist[-7] if len(hist) >= 7 else np.mean(hist)
            out.append(v)
            hist.append(v)
        return np.array(out), None

    if name == "Weekday average":
        recent = daily.tail(28)
        dow = recent.groupby(recent["date"].dt.dayofweek)["units_sold"].mean()
        overall = recent["units_sold"].mean()
        return np.array([dow.get(d.dayofweek, overall) for d in future_dates]), None

    frame = _training_frame(daily)
    model = _make_model(name)
    model.fit(frame[FEATURES], frame["y"])
    if hasattr(model, "n_jobs"):
        model.set_params(n_jobs=1)  # single-row recursive predictions are faster without threads
    hist = values[:]
    t0 = len(values)
    out = []
    for i, d in enumerate(future_dates):
        feats = pd.DataFrame([_row_features(hist, d, t0 + i)])[FEATURES]
        p = max(0.0, float(model.predict(feats)[0]))
        out.append(p)
        hist.append(p)  # recursive multi-step forecast
    return np.array(out), model


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - actual
    denom = max(np.abs(actual).sum(), 1e-9)
    return {"MAE": float(np.abs(err).mean()),
            "WAPE": float(np.abs(err).sum() / denom * 100),
            "Bias %": float(err.sum() / denom * 100)}


# ---------------------------------------------------------------- public API
def forecast_product(hist: pd.DataFrame, horizon: int = 7, model: str = "Auto",
                     test_days: int = 14) -> ForecastResult:
    daily = daily_series(hist)
    n = len(daily)
    candidates = MODELS if n >= MIN_ML_DAYS else ["Weekday average", "Seasonal naive"]
    if model != "Auto" and model in candidates:
        candidates_to_score = list(dict.fromkeys([model, "Seasonal naive"]))
    else:
        candidates_to_score = candidates

    test_days = min(test_days, max(7, n // 4))
    train, test = daily.iloc[:-test_days], daily.iloc[-test_days:]
    actual = test["units_sold"].to_numpy(float)

    rows, preds_by_model = [], {}
    can_backtest = len(train) >= 14
    for name in candidates_to_score:
        if not can_backtest:
            break
        if name in ML_MODELS and len(train) < MIN_ML_DAYS - test_days:
            continue
        p, _ = _predict(name, train, test_days)
        preds_by_model[name] = p
        rows.append({"Model": name, **_metrics(actual, p)})

    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty:
        naive = leaderboard.loc[leaderboard.Model == "Seasonal naive", "WAPE"]
        naive_wape = float(naive.iloc[0]) if len(naive) else np.nan
        leaderboard["Skill vs naive %"] = (1 - leaderboard["WAPE"] / naive_wape) * 100 if naive_wape else np.nan
        leaderboard = leaderboard.sort_values("WAPE").reset_index(drop=True)

    if model == "Auto" or model not in candidates:
        chosen = leaderboard.iloc[0]["Model"] if not leaderboard.empty else "Weekday average"
    else:
        chosen = model

    # Interval from backtest residuals of the chosen model (split-conformal style)
    if chosen in preds_by_model:
        resid = actual - preds_by_model[chosen]
        backtest = pd.DataFrame({"date": test["date"].values, "actual": actual,
                                 "predicted": preds_by_model[chosen]})
    else:
        resid = daily["units_sold"].diff(7).dropna().to_numpy(float)
        backtest = pd.DataFrame(columns=["date", "actual", "predicted"])
    sigma = float(np.std(resid)) if len(resid) > 1 else float(daily["units_sold"].std() or 1.0)
    bias = float(np.mean(resid)) if len(resid) else 0.0

    preds, fitted = _predict(chosen, daily, horizon)
    future_dates = pd.date_range(daily["date"].max() + pd.Timedelta(days=1), periods=horizon)
    # interval = forecast + mean backtest error ± z · σ(backtest errors)
    lo80, hi80 = bias - 1.2816 * sigma, bias + 1.2816 * sigma
    lo95, hi95 = bias - 1.96 * sigma, bias + 1.96 * sigma
    # interval width grows gently with horizon (errors compound in recursive forecasts)
    grow = np.sqrt(1 + np.arange(horizon) / max(horizon, 1))
    band = lambda off: preds + bias + (off - bias) * grow  # noqa: E731 — widens around the centre
    fc = pd.DataFrame({
        "date": future_dates,
        "predicted_demand": np.round(np.maximum(0, preds), 1),
        "lo80": np.round(np.maximum(0, band(lo80)), 1),
        "hi80": np.round(np.maximum(0, band(hi80)), 1),
        "lo95": np.round(np.maximum(0, band(lo95)), 1),
        "hi95": np.round(np.maximum(0, band(hi95)), 1),
    })

    importances = pd.DataFrame()
    if fitted is not None and hasattr(fitted, "feature_importances_"):
        importances = (pd.DataFrame({"feature": [FEATURE_LABELS[f] for f in FEATURES],
                                     "importance": fitted.feature_importances_})
                       .sort_values("importance", ascending=False))

    return ForecastResult(fc, chosen, daily, backtest, leaderboard, sigma, importances)


def explain_importance(hist: pd.DataFrame) -> pd.DataFrame:
    """Random-forest feature importances for one product (used for explainability)."""
    daily = daily_series(hist)
    if len(daily) < MIN_ML_DAYS:
        return pd.DataFrame()
    frame = _training_frame(daily)
    m = _make_model("Random Forest").fit(frame[FEATURES], frame["y"])
    return (pd.DataFrame({"feature": [FEATURE_LABELS[f] for f in FEATURES],
                          "importance": m.feature_importances_})
            .sort_values("importance", ascending=False))
