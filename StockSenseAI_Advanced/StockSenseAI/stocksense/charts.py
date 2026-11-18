"""Plotly chart builders. Every chart: one y-axis, thin marks, hover on by default."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import (AXIS, BLUE, CRITICAL, GOOD, GRID, INK, INK_2, MUTED, ORANGE, ORDINAL_ABC,
                    RISK_COLORS, SEQ_BLUE, SERIOUS, SURFACE, WARNING, category_colors, rgba)


def _h(fig, height):
    fig.update_layout(height=height)
    return fig


# ---------------------------------------------------------------- command centre
def sales_trend(df: pd.DataFrame, height=380):
    d = df.groupby("date", as_index=False).agg(units=("units_sold", "sum"))
    d["value"] = df.assign(v=df.units_sold * df.unit_cost).groupby("date").v.sum().values
    d["ma7"] = d.units.rolling(7, min_periods=1).mean()
    roll_mu = d.units.rolling(14, min_periods=7).mean().shift(1)
    roll_sd = d.units.rolling(14, min_periods=7).std().shift(1)
    d["z"] = (d.units - roll_mu) / roll_sd
    anom = d[d.z.abs() > 2.2]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.date, y=d.units, name="Daily units", mode="lines",
                             line=dict(color=rgba(BLUE, 0.45), width=1.5),
                             fill="tozeroy", fillcolor=rgba(BLUE, 0.08),
                             customdata=d[["value"]],
                             hovertemplate="%{x|%a %d %b}<br><b>%{y:,} units</b><br>₹%{customdata[0]:,.0f} at cost<extra></extra>"))
    fig.add_trace(go.Scatter(x=d.date, y=d.ma7, name="7-day average", mode="lines",
                             line=dict(color=BLUE, width=2.5),
                             hovertemplate="7-day avg %{y:,.0f}<extra></extra>"))
    if len(anom):
        fig.add_trace(go.Scatter(x=anom.date, y=anom.units, name="Anomaly (|z| > 2.2)", mode="markers",
                                 marker=dict(color=ORANGE, size=10, line=dict(color=SURFACE, width=2)),
                                 customdata=anom[["z"]],
                                 hovertemplate="%{x|%d %b}: %{y:,} units<br>z-score %{customdata[0]:.1f}<extra>Anomaly</extra>"))
    fig.update_layout(title="Daily units sold · 7-day trend · anomaly detection", hovermode="x unified",
                      yaxis_title="Units")
    fig.update_xaxes(rangeselector=dict(
        buttons=[dict(count=14, label="2W", step="day", stepmode="backward"),
                 dict(count=1, label="1M", step="month", stepmode="backward"),
                 dict(count=3, label="3M", step="month", stepmode="backward"),
                 dict(step="all", label="All")],
        bgcolor="#222220", activecolor="#383835", font=dict(color=INK_2), x=0, y=1.0, yanchor="bottom"))
    fig.update_layout(legend=dict(orientation="h", y=-0.12, yanchor="top", x=0, xanchor="left"),
                      margin=dict(b=40))
    return _h(fig, height)


def value_treemap(inv: pd.DataFrame, height=380):
    cmap = category_colors(inv.category)
    t = inv.assign(risk_label=inv.stockout_risk + " risk")
    fig = px.treemap(t, path=[px.Constant("All inventory"), "category", "product_name"], values="stock_value",
                     color="category", color_discrete_map={**cmap, "(?)": "#222220"},
                     custom_data=["stockout_risk", "days_of_cover", "stock_on_hand"])
    fig.update_traces(
        root_color="#222220", marker=dict(line=dict(color=SURFACE, width=2)),
        texttemplate="<b>%{label}</b><br>₹%{value:,.0f}", textfont=dict(color="#ffffff"),
        hovertemplate="<b>%{label}</b><br>Stock value ₹%{value:,.0f}<br>Share of parent %{percentParent:.0%}<extra></extra>",
        tiling=dict(pad=2))
    fig.update_layout(title="Where the money sits · inventory value by category → product",
                      margin=dict(l=8, r=8, t=56, b=8))
    return _h(fig, height)


def health_gauge(score: float, parts: dict, height=300):
    color = GOOD if score >= 75 else (WARNING if score >= 55 else CRITICAL)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(score), number=dict(suffix="/100", valueformat=".0f", font=dict(size=44, color=INK)),
        gauge=dict(axis=dict(range=[0, 100], tickcolor=AXIS, tickfont=dict(color=MUTED)),
                   bar=dict(color=color, thickness=0.28), bgcolor="#222220", borderwidth=0,
                   steps=[dict(range=[0, 55], color=rgba(CRITICAL, 0.14)),
                          dict(range=[55, 75], color=rgba(WARNING, 0.12)),
                          dict(range=[75, 100], color=rgba(GOOD, 0.12))]),
        title=dict(text="Inventory health score", font=dict(color=INK, size=15))))
    sub = " · ".join(f"{k} {v:.0f}" for k, v in parts.items())
    fig.add_annotation(text=sub, x=0.5, y=-0.08, xref="paper", yref="paper", showarrow=False,
                       font=dict(color=MUTED, size=11))
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=34))
    return _h(fig, height)


def risk_segment_bar(inv: pd.DataFrame, height=150):
    counts = inv.stockout_risk.value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
    fig = go.Figure()
    for k, v in counts.items():
        fig.add_trace(go.Bar(y=["Products"], x=[v], orientation="h", name=f"{k} risk",
                             marker=dict(color=RISK_COLORS[k], line=dict(color=SURFACE, width=2)),
                             text=(f"{k} · {v}" if v / max(counts.sum(), 1) >= 0.2 else str(v)) if v else "",
                             textposition="inside", insidetextanchor="middle", textangle=0,
                             textfont=dict(color="#0b0b0b" if k == "Medium" else "#ffffff"),
                             hovertemplate=f"{k} risk: %{{x}} products<extra></extra>"))
    fig.update_layout(barmode="stack", title="Stockout risk mix", showlegend=True,
                      legend=dict(y=1.02, x=1, xanchor="right", traceorder="normal"),
                      xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=8, r=8, t=44, b=8))
    return _h(fig, height)


def weekday_heatmap(df: pd.DataFrame, height=300):
    d = df.groupby("date").units_sold.sum().reset_index()
    d["week"] = d.date - pd.to_timedelta(d.date.dt.dayofweek, unit="D")
    d["dow"] = d.date.dt.dayofweek
    piv = d.pivot_table(index="dow", columns="week", values="units_sold", aggfunc="sum")
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=piv.columns, y=[labels[i] for i in piv.index],
        colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)],
        xgap=2, ygap=2, colorbar=dict(title=dict(text="Units", font=dict(color=MUTED)), tickfont=dict(color=MUTED),
                                      thickness=10, outlinewidth=0),
        hovertemplate="Week of %{x|%d %b} · %{y}<br><b>%{z:,} units</b><extra></extra>"))
    fig.update_layout(title="Demand calendar · weekday × week", yaxis=dict(autorange="reversed"))
    return _h(fig, height)


def category_momentum(df: pd.DataFrame, height=300):
    last = df.date.max()
    cur = df[df.date > last - pd.Timedelta(days=7)].groupby("category").units_sold.sum()
    prev = df[(df.date <= last - pd.Timedelta(days=7)) & (df.date > last - pd.Timedelta(days=14))].groupby("category").units_sold.sum()
    chg = ((cur - prev) / prev.replace(0, np.nan) * 100).dropna().sort_values()
    colors = [BLUE if v >= 0 else "#e66767" for v in chg.values]
    fig = go.Figure(go.Bar(x=chg.values, y=chg.index, orientation="h",
                           marker=dict(color=colors, cornerradius=4),
                           text=[f"{v:+.0f}%" for v in chg.values], textposition="outside",
                           textfont=dict(color=INK_2), cliponaxis=False,
                           customdata=np.c_[cur.reindex(chg.index).values, prev.reindex(chg.index).values],
                           hovertemplate="<b>%{y}</b><br>%{x:+.1f}% week-on-week<br>Last 7 d: %{customdata[0]:,} · Prior 7 d: %{customdata[1]:,}<extra></extra>"))
    span = max(abs(chg.min()), abs(chg.max()), 5) * 1.45
    fig.update_layout(title="Category momentum · last 7 days vs prior 7", xaxis=dict(ticksuffix="%", showgrid=True,
                      range=[-span if chg.min() < 0 else -span * 0.15, span if chg.max() > 0 else span * 0.15],
                      zeroline=True, zerolinecolor=MUTED, zerolinewidth=1))
    return _h(fig, height)


# ---------------------------------------------------------------- forecasting
def forecast_fan(res, product: str, show_days=60, height=440):
    h = res.history.tail(show_days)
    fc = res.forecast
    fig = go.Figure()
    # 95% and 80% bands
    for lo, hi, a, name in [("lo95", "hi95", 0.14, "95% interval"), ("lo80", "hi80", 0.26, "80% interval")]:
        fig.add_trace(go.Scatter(x=fc.date, y=fc[hi], mode="lines", line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fc.date, y=fc[lo], mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor=rgba(ORANGE, a), name=name, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=h.date, y=h.units_sold, name="Actual sales", mode="lines",
                             line=dict(color=BLUE, width=2),
                             hovertemplate="%{x|%a %d %b}: <b>%{y:,}</b> actual<extra></extra>"))
    if len(res.backtest):
        fig.add_trace(go.Scatter(x=res.backtest.date, y=res.backtest.predicted, name="Backtest prediction",
                                 mode="lines", line=dict(color=MUTED, width=1.5, dash="dot"),
                                 hovertemplate="%{x|%d %b}: %{y:,.1f} predicted (backtest)<extra></extra>"))
    bridge = pd.concat([h.tail(1).rename(columns={"units_sold": "predicted_demand"})[["date", "predicted_demand"]],
                        fc[["date", "predicted_demand"]]])
    fig.add_trace(go.Scatter(x=bridge.date, y=bridge.predicted_demand, name="Forecast", mode="lines+markers",
                             line=dict(color=ORANGE, width=2.5, dash="dash"), marker=dict(size=8, line=dict(color=SURFACE, width=2)),
                             customdata=np.r_[[[np.nan] * 4], fc[["lo80", "hi80", "lo95", "hi95"]].values],
                             hovertemplate="%{x|%a %d %b}: <b>%{y:,.1f}</b> forecast<br>80%: %{customdata[0]:.0f}–%{customdata[1]:.0f} · 95%: %{customdata[2]:.0f}–%{customdata[3]:.0f}<extra></extra>"))
    fig.add_vline(x=h.date.max(), line=dict(color=AXIS, width=1))
    fig.add_annotation(x=h.date.max(), y=1, yref="paper", text="today", showarrow=False, xanchor="left",
                       font=dict(color=MUTED, size=11), xshift=4)
    if len(res.backtest):
        fig.add_vrect(x0=res.backtest.date.min(), x1=res.backtest.date.max(), fillcolor="rgba(255,255,255,0.03)",
                      line_width=0, annotation_text="backtest window", annotation_position="top left",
                      annotation_font=dict(color=MUTED, size=11))
    fig.update_layout(title=f"{product} · actual vs forecast with prediction intervals", yaxis_title="Units",
                      hovermode="x unified")
    return _h(fig, height)


def leaderboard_chart(lb: pd.DataFrame, chosen: str, height=280):
    lb = lb.sort_values("WAPE", ascending=False)
    colors = [BLUE if m == chosen else "#383835" for m in lb.Model]
    fig = go.Figure(go.Bar(x=lb.WAPE, y=lb.Model, orientation="h", marker=dict(color=colors, cornerradius=4),
                           text=[f"{v:.1f}%" for v in lb.WAPE], textposition="outside", cliponaxis=False,
                           textfont=dict(color=INK_2),
                           customdata=lb[["MAE", "Bias %"]].values,
                           hovertemplate="<b>%{y}</b><br>WAPE %{x:.1f}%<br>MAE %{customdata[0]:.2f} units/day<br>Bias %{customdata[1]:+.1f}%<extra></extra>"))
    fig.update_layout(title="Model tournament · backtest error (lower is better)",
                      xaxis=dict(ticksuffix="%", showgrid=True, range=[0, lb.WAPE.max() * 1.22]))
    return _h(fig, height)


def importance_chart(imp: pd.DataFrame, height=320):
    imp = imp.head(8).sort_values("importance")
    fig = go.Figure(go.Bar(x=imp.importance, y=imp.feature, orientation="h", marker=dict(color=BLUE, cornerradius=4),
                           hovertemplate="<b>%{y}</b><br>%{x:.1%} of model decisions<extra></extra>"))
    fig.update_layout(title="What drives demand · Random Forest importance", xaxis=dict(tickformat=".0%", showgrid=True))
    return _h(fig, height)


def weekday_profile(res, height=280):
    d = res.history.copy()
    d["dow"] = d.date.dt.dayofweek
    prof = d.groupby("dow").units_sold.mean().reindex(range(7))
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    avg = prof.mean()
    colors = [BLUE if v >= avg else "#2a5c99" for v in prof.values]
    fig = go.Figure(go.Bar(x=labels, y=prof.values, marker=dict(color=colors, cornerradius=4),
                           hovertemplate="%{x}: <b>%{y:.1f}</b> units on average<extra></extra>"))
    fig.add_hline(y=avg, line=dict(color=MUTED, width=1), annotation_text=f"avg {avg:.1f}",
                  annotation_font=dict(color=MUTED, size=11), annotation_position="top left")
    fig.update_layout(title="Weekly seasonality profile", yaxis=dict(showgrid=True, gridcolor=GRID))
    return _h(fig, height)


# ---------------------------------------------------------------- risk
def risk_matrix(inv: pd.DataFrame, height=460):
    d = inv.copy()
    d["cover_plot"] = d.days_of_cover.replace(np.inf, 99).clip(upper=99)
    d["daily_value"] = d.avg_daily_forecast * d.unit_cost
    d["size"] = 12 + 34 * np.sqrt(d.daily_value / max(d.daily_value.max(), 1))
    fig = go.Figure()
    xmin, xmax = max(0.5, d.cover_plot.min() * 0.7), d.cover_plot.max() * 1.4
    fig.add_hrect(y0=0.5, y1=1.1, fillcolor=rgba(CRITICAL, 0.06), line_width=0,
                  annotation_text="act now · ≥ 50% chance of stockout", annotation_position="top right",
                  annotation_font=dict(color=MUTED, size=11))
    for k in ["High", "Medium", "Low"]:
        s = d[d.stockout_risk == k]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s.cover_plot, y=s.stockout_probability, mode="markers", name=f"{k} risk",
            marker=dict(size=s["size"], color=rgba(RISK_COLORS[k], 0.85), line=dict(color=SURFACE, width=2)),
            customdata=np.c_[s.product_name, s.lead_time_days, s.stock_on_hand, s.avg_daily_forecast, s.daily_value],
            hovertemplate="<b>%{customdata[0]}</b><br>Days of cover %{x:.1f} · lead time %{customdata[1]:g} d"
                          "<br>Stockout probability <b>%{y:.0%}</b><br>Stock %{customdata[2]:,.0f} · demand %{customdata[3]:.1f}/day"
                          "<br>₹%{customdata[4]:,.0f} of sales/day at stake<extra></extra>"))
    fig.update_layout(title="Risk radar · stock cover vs stockout probability (bubble size = ₹ demand/day)",
                      xaxis=dict(title="Days of stock cover (log scale)", type="log", range=[np.log10(xmin), np.log10(xmax)],
                                 tickvals=[1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90],
                                 showgrid=True, gridcolor=GRID),
                      yaxis=dict(title="P(stockout before delivery)", tickformat=".0%", range=[-0.06, 1.1], showgrid=True))
    return _h(fig, height)


def stockout_prob_bars(inv: pd.DataFrame, height=460):
    d = inv.sort_values("stockout_probability")
    fig = go.Figure(go.Bar(
        x=d.stockout_probability, y=d.product_name, orientation="h",
        marker=dict(color=[RISK_COLORS[r] for r in d.stockout_risk], cornerradius=4),
        text=[f"{p:.0%}" for p in d.stockout_probability], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK_2),
        customdata=np.c_[d.stockout_risk, d.lead_time_days],
        hovertemplate="<b>%{y}</b><br>%{x:.1%} chance of stockout within %{customdata[1]:g} d<br>%{customdata[0]} risk<extra></extra>"))
    fig.update_layout(title="Stockout probability · 5,000 simulations each",
                      xaxis=dict(tickformat=".0%", range=[0, 1.12], showgrid=True))
    return _h(fig, height)


def depletion_chart(paths: pd.DataFrame, row, product: str, height=420):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=paths.date, y=paths.p95, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=paths.date, y=paths.p5, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=rgba(BLUE, 0.14), name="90% of simulations", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=paths.date, y=paths.p75, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=paths.date, y=paths.p25, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=rgba(BLUE, 0.28), name="50% of simulations", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=paths.date, y=paths.p50, mode="lines", line=dict(color=BLUE, width=2.5),
                             name="Median stock level",
                             customdata=paths[["p5", "p95"]].values,
                             hovertemplate="%{x|%a %d %b}: median <b>%{y:,.0f}</b> units<br>range %{customdata[0]:,.0f} to %{customdata[1]:,.0f}<extra></extra>"))
    fig.add_hline(y=row.reorder_point, line=dict(color=WARNING, width=1.5),
                  annotation_text=f"Reorder point {row.reorder_point}", annotation_position="top right",
                  annotation_font=dict(color=WARNING, size=11))
    fig.add_hline(y=row.safety_stock, line=dict(color=SERIOUS, width=1, dash="dot"),
                  annotation_text=f"Safety stock {row.safety_stock}", annotation_position="bottom right",
                  annotation_font=dict(color=SERIOUS, size=11))
    lo = float(min(paths.p5.min(), 0))
    if lo < 0:
        fig.add_hrect(y0=lo * 1.05, y1=0, fillcolor=rgba(CRITICAL, 0.08), line_width=0,
                      annotation_text="stockout zone · unmet demand", annotation_position="bottom left",
                      annotation_font=dict(color=CRITICAL, size=11))
    fig.add_hline(y=0, line=dict(color=CRITICAL, width=1.5))
    arrive = paths.date.min() + pd.Timedelta(days=float(row.lead_time_days))
    fig.add_vline(x=arrive, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_annotation(x=arrive, y=1, yref="paper", text="order today → arrives", showarrow=False, xanchor="left",
                       xshift=4, font=dict(color=MUTED, size=11))
    top = max(row.stock_on_hand, row.reorder_point) * 1.15
    fig.update_yaxes(range=[max(lo * 1.05, -0.6 * top), top])
    fig.update_layout(title=f"{product} · simulated stock depletion if nothing is reordered",
                      yaxis_title="Units on hand", hovermode="x unified")
    return _h(fig, height)


# ---------------------------------------------------------------- ABC-XYZ
def pareto_chart(inv: pd.DataFrame, height=460):
    d = inv.sort_values("annual_value", ascending=False).reset_index(drop=True)
    total = d.annual_value.sum() or 1
    d["cum"] = d.annual_value.cumsum() / total
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.06)
    for cls in ["A", "B", "C"]:
        s = d[d.abc == cls]
        fig.add_trace(go.Bar(x=s.product_name, y=s.annual_value, name=f"Class {cls}",
                             marker=dict(color=ORDINAL_ABC[cls], cornerradius=4),
                             hovertemplate="<b>%{x}</b><br>₹%{y:,.0f} annual consumption value<extra>Class " + cls + "</extra>"),
                      row=1, col=1)
    fig.add_trace(go.Scatter(x=d.product_name, y=d.cum, mode="lines+markers", name="Cumulative share",
                             line=dict(color=INK_2, width=2), marker=dict(size=7, color=INK_2, line=dict(color=SURFACE, width=2)),
                             hovertemplate="<b>%{x}</b><br>Top items cover %{y:.0%} of value<extra></extra>"), row=2, col=1)
    for y, lbl in [(0.8, "80% · A/B cut"), (0.95, "95% · B/C cut")]:
        fig.add_hline(y=y, line=dict(color=MUTED, width=1, dash="dot"), row=2, col=1,
                      annotation_text=lbl, annotation_font=dict(color=MUTED, size=10), annotation_position="bottom right")
    fig.update_yaxes(title_text="₹ / year", row=1, col=1, showgrid=True)
    fig.update_yaxes(title_text="Cumulative", tickformat=".0%", range=[0, 1.05], row=2, col=1, showgrid=True)
    fig.update_xaxes(tickangle=-35, row=2, col=1)
    fig.update_layout(title="Pareto analysis · which products carry the value", barmode="stack",
                      legend=dict(traceorder="normal"))
    return _h(fig, height)


def abc_xyz_matrix(inv: pd.DataFrame, height=460):
    abc, xyz = ["A", "B", "C"], ["X", "Y", "Z"]
    z = np.zeros((3, 3))
    names = [[""] * 3 for _ in range(3)]
    for i, a in enumerate(abc):
        for j, x in enumerate(xyz):
            s = inv[(inv.abc == a) & (inv.xyz == x)]
            z[i, j] = len(s)
            names[i][j] = "<br>".join(s.product_name.head(6)) or "—"
    text = [[f"<b>{a}{x}</b> · {int(z[i, j])}" for j, x in enumerate(xyz)] for i, a in enumerate(abc)]
    fig = go.Figure(go.Heatmap(
        z=z, x=["X · steady", "Y · variable", "Z · erratic"], y=["A · high value", "B · medium", "C · low value"],
        text=text, texttemplate="%{text}", textfont=dict(color=INK, size=14), customdata=names,
        colorscale=[[0, "#1f2329"], [0.0001, SEQ_BLUE[0]], [1, SEQ_BLUE[3]]], showscale=False, xgap=3, ygap=3,
        hovertemplate="<b>%{y} × %{x}</b><br>%{customdata}<extra></extra>"))
    fig.update_layout(title="ABC × XYZ segmentation · value vs demand variability",
                      yaxis=dict(autorange="reversed"), xaxis=dict(side="bottom"))
    return _h(fig, height)


# ---------------------------------------------------------------- expiry
def expiry_timeline(inv: pd.DataFrame, as_of: pd.Timestamp, window=60, height=420):
    d = inv.dropna(subset=["days_to_expiry"]).copy()
    d = d[d.days_to_expiry <= window].sort_values("days_to_expiry", ascending=False)
    if d.empty:
        return None
    def status(x):
        return "Expired" if x < 0 else ("≤ 7 days" if x <= 7 else ("8–30 days" if x <= 30 else "31–60 days"))
    d["status"] = d.days_to_expiry.apply(status)
    cmap = {"Expired": CRITICAL, "≤ 7 days": SERIOUS, "8–30 days": WARNING, "31–60 days": GOOD}
    fig = go.Figure()
    for k, c in cmap.items():
        s = d[d.status == k]
        if s.empty:
            continue
        fig.add_trace(go.Bar(
            x=s.days_to_expiry.clip(lower=0.4), y=s.product_name, orientation="h", name=k,
            marker=dict(color=c, cornerradius=4),
            customdata=np.c_[s.expiry_date.dt.strftime("%d %b %Y"), s.stock_on_hand, s.expected_waste_units, s.expiry_action],
            hovertemplate="<b>%{y}</b><br>Expires %{customdata[0]} (in %{x:.0f} d)<br>On hand %{customdata[1]:,.0f} · "
                          "forecast unsold %{customdata[2]:,.0f}<br>Action: %{customdata[3]}<extra></extra>"))
    fig.update_layout(title=f"Expiry runway · days left from {as_of:%d %b}", barmode="overlay",
                      xaxis=dict(title="Days until expiry", showgrid=True))
    return _h(fig, height)


def waste_chart(inv: pd.DataFrame, height=420):
    d = inv[(inv.days_to_expiry.notna()) & (inv.days_to_expiry <= 60)].copy()
    if d.empty:
        return None
    d["sellable"] = np.minimum(d.stock_on_hand, d.sellable_before_expiry.fillna(0))
    d = d.sort_values("expected_waste_units")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=d.product_name, x=d.sellable, orientation="h", name="Sells before expiry",
                         marker=dict(color=BLUE, line=dict(color=SURFACE, width=2)),
                         hovertemplate="<b>%{y}</b><br>%{x:,.0f} units sell in time<extra></extra>"))
    fig.add_trace(go.Bar(y=d.product_name, x=d.expected_waste_units, orientation="h", name="Forecast to expire unsold",
                         marker=dict(color=CRITICAL, line=dict(color=SURFACE, width=2)),
                         customdata=np.c_[d.expected_waste_value, d.expiry_action],
                         hovertemplate="<b>%{y}</b><br>%{x:,.0f} units unsold · ₹%{customdata[0]:,.0f}<br>%{customdata[1]}<extra></extra>"))
    fig.update_layout(barmode="stack", title="Waste forecast · current stock vs demand before expiry",
                      xaxis=dict(title="Units on hand", showgrid=True))
    return _h(fig, height)


# ---------------------------------------------------------------- reorder / what-if
def budget_chart(alloc: pd.DataFrame, height=420):
    d = alloc.copy()
    d["deferred_cost"] = (d.recommended_reorder_qty - d.funded_qty) * d.unit_cost
    d = d.sort_values("estimated_reorder_cost")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=d.product_name, x=d.funded_cost, orientation="h", name="Funded",
                         marker=dict(color=BLUE, line=dict(color=SURFACE, width=2)),
                         customdata=np.c_[d.funded_qty, d.stockout_risk],
                         hovertemplate="<b>%{y}</b><br>Funded ₹%{x:,.0f} (%{customdata[0]:,} units)<br>%{customdata[1]} risk<extra></extra>"))
    fig.add_trace(go.Bar(y=d.product_name, x=d.deferred_cost, orientation="h", name="Deferred (over budget)",
                         marker=dict(color="#4a4a46", line=dict(color=SURFACE, width=2)),
                         hovertemplate="<b>%{y}</b><br>Deferred ₹%{x:,.0f}<extra></extra>"))
    fig.update_layout(barmode="stack", title="Budget allocation · risk-weighted priority",
                      xaxis=dict(title="₹", showgrid=True))
    return _h(fig, height)


def spend_by_category(alloc: pd.DataFrame, height=300):
    cmap = category_colors(alloc.category)
    d = alloc.groupby("category", as_index=False).funded_cost.sum().sort_values("funded_cost")
    fig = go.Figure(go.Bar(x=d.funded_cost, y=d.category, orientation="h",
                           marker=dict(color=[cmap[c] for c in d.category], cornerradius=4),
                           hovertemplate="<b>%{y}</b><br>₹%{x:,.0f}<extra></extra>"))
    fig.update_layout(title="Funded spend by category", xaxis=dict(showgrid=True))
    return _h(fig, height)


def dumbbell(base: pd.DataFrame, scen: pd.DataFrame, height=480):
    d = base[["product_id", "product_name", "stockout_probability"]].merge(
        scen[["product_id", "stockout_probability"]], on="product_id", suffixes=("_base", "_scen"))
    d = d.sort_values("stockout_probability_scen")
    fig = go.Figure()
    for _, r in d.iterrows():
        fig.add_trace(go.Scatter(x=[r.stockout_probability_base, r.stockout_probability_scen], y=[r.product_name] * 2,
                                 mode="lines", line=dict(color="#4a4a46", width=3), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d.stockout_probability_base, y=d.product_name, mode="markers", name="Today",
                             marker=dict(color=MUTED, size=11, line=dict(color=SURFACE, width=2)),
                             hovertemplate="<b>%{y}</b><br>Today %{x:.0%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=d.stockout_probability_scen, y=d.product_name, mode="markers", name="Scenario",
                             marker=dict(color=ORANGE, size=12, line=dict(color=SURFACE, width=2)),
                             hovertemplate="<b>%{y}</b><br>Scenario %{x:.0%}<extra></extra>"))
    fig.update_layout(title="Stockout probability · today vs scenario",
                      xaxis=dict(tickformat=".0%", range=[-0.03, 1.05], showgrid=True))
    return _h(fig, height)
