"""StockSense AI — intelligent inventory command centre (Streamlit)."""
from __future__ import annotations

import io
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from stocksense import charts
from stocksense.data import REQUIRED, normalize_data
from stocksense.forecasting import MODELS, explain_importance, forecast_product
from stocksense.insights import generate_insights
from stocksense.inventory import Policy, allocate_budget, compute_inventory, depletion_paths, health_score
from stocksense.theme import CRITICAL, CSS, GOOD, PLOTLY_CONFIG, RISK_ICON, SERIOUS, WARNING, register_template

st.set_page_config(page_title="StockSense AI", page_icon="📦", layout="wide", initial_sidebar_state="expanded")
register_template()
st.markdown(CSS, unsafe_allow_html=True)


def show(fig, key=None):
    if fig is None:
        st.info("Nothing to show for the current selection.")
        return
    st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG, key=key)


def signed_inr(x: float) -> str:
    return ("+" if x > 0 else "-" if x < 0 else "") + inr(abs(x))


def inr(x: float) -> str:
    if abs(x) >= 1e7:
        return f"₹{x / 1e7:.2f} Cr"
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:.2f} L"
    return f"₹{x:,.0f}"


# ---------------------------------------------------------------- cached compute
@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    return pd.read_csv("data/sample_supermarket_sales.csv")


@st.cache_data(show_spinner=False)
def run_forecasts(df: pd.DataFrame, horizon: int, model: str) -> dict:
    return {pid: forecast_product(g, horizon, model) for pid, g in df.groupby("product_id")}


@st.cache_data(show_spinner=False)
def run_inventory(df: pd.DataFrame, _forecasts: dict, fkey: tuple, policy: Policy) -> pd.DataFrame:
    return compute_inventory(df, _forecasts, policy)


@st.cache_data(show_spinner=False)
def cached_importance(hist: pd.DataFrame) -> pd.DataFrame:
    return explain_importance(hist)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ⚙️ Control centre")
    uploaded = st.file_uploader("Upload sales & stock CSV", type=["csv"],
                                help="One row per product per day. Leave empty to use the demo dataset.")
    st.markdown("**Forecasting**")
    horizon = st.slider("Forecast horizon (days)", 3, 28, 14)
    model_choice = st.selectbox("Model", ["Auto"] + MODELS, index=0,
                                help="Auto backtests every model on each product's last 14 days and keeps the most accurate one.")
    st.markdown("**Inventory policy**")
    service_level = st.select_slider("Target service level", options=[0.80, 0.85, 0.90, 0.95, 0.975, 0.99],
                                     value=0.95, format_func=lambda v: f"{v:.1%}".replace(".0%", "%"),
                                     help="Probability of NOT running out during the supplier lead time.")
    review_days = st.slider("Review period (days)", 1, 14, 7)
    max_cover = st.slider("Max order cover (days of demand)", 7, 45, 14)
    with st.expander("Cost assumptions"):
        order_cost = st.number_input("Ordering cost per PO (₹)", 0, 5000, 150, 10)
        holding = st.slider("Annual holding cost (% of unit cost)", 5, 50, 20) / 100
        elasticity = st.slider("Markdown elasticity", 0.5, 4.0, 1.8, 0.1,
                               help="% extra sales for each 1% discount. Used to size expiry markdowns.")
    st.divider()
    st.caption("Required columns")
    st.code(", ".join(REQUIRED), language="text")

policy = Policy(service_level=service_level, review_days=review_days, order_cost=float(order_cost),
                holding_rate=holding, elasticity=elasticity, max_cover_days=max_cover)

# ---------------------------------------------------------------- data
try:
    raw = pd.read_csv(uploaded) if uploaded is not None else load_sample()
    data, dq = normalize_data(raw)
    source = uploaded.name if uploaded is not None else "Built-in demo dataset"
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load the dataset: {e}")
    st.stop()
if data.empty:
    st.warning("No valid rows found in the dataset.")
    st.stop()

with st.spinner(f"Training and backtesting forecast models for {data.product_id.nunique()} products…"):
    forecasts = run_forecasts(data, horizon, model_choice)
inv = run_inventory(data, forecasts, (horizon, model_choice, len(data)), policy)
as_of = data.date.max()
score, score_parts = health_score(inv)
insights = generate_insights(data, inv, horizon)
name_to_id = dict(zip(inv.product_name, inv.product_id))

# ---------------------------------------------------------------- header
st.markdown(f"""
<div class="hero">
  <h1>📦 StockSense AI</h1>
  <p>Intelligent inventory command centre — ML demand forecasting, Monte Carlo stockout risk,
     expiry-waste prevention and budget-aware replenishment.</p>
  <div class="chips">
    <span class="chip">📁 {source}</span>
    <span class="chip">📅 As of {as_of:%d %b %Y}</span>
    <span class="chip">🧾 {len(data):,} records · {dq['days']} days</span>
    <span class="chip">🎯 Service target {service_level:.1%}</span>
    <span class="chip">🔮 {horizon}-day horizon · {model_choice}</span>
  </div>
</div>""", unsafe_allow_html=True)

daily_units = data.groupby("date").units_sold.sum()
last7 = daily_units.tail(7).sum()
prev7 = daily_units.iloc[-14:-7].sum() if len(daily_units) >= 14 else np.nan
reorder_spend = inv.estimated_reorder_cost.sum()
waste_val = inv.expected_waste_value.sum()
high = int((inv.stockout_risk == "High").sum())

k = st.columns(6)
k[0].metric("Inventory value", inr(inv.stock_value.sum()), f"{len(inv)} products", delta_color="off", border=True)
k[1].metric("Units sold · 7 days", f"{last7:,.0f}",
            f"{(last7 / prev7 - 1) * 100:+.1f}% vs prior week" if prev7 else None, border=True)
k[2].metric("Health score", f"{score:.0f}/100", "Good" if score >= 75 else ("Fair" if score >= 55 else "Poor"),
            delta_color="normal" if score >= 75 else "inverse", border=True)
k[3].metric("High stockout risk", f"{high}", "Act today" if high else "On track",
            delta_color="inverse" if high else "normal", border=True)
k[4].metric("Waste value at risk", inr(waste_val), "Markdown needed" if waste_val else "No waste forecast",
            delta_color="inverse" if waste_val else "normal", border=True)
k[5].metric("Replenishment needed", inr(reorder_spend),
            f"{int((inv.recommended_reorder_qty > 0).sum())} purchase lines", delta_color="off", border=True)

tabs = st.tabs(["🧭 Command centre", "🔮 Demand forecast", "🚨 Stockout risk", "📐 ABC-XYZ",
                "⏳ Expiry & waste", "🛒 Reorder optimizer", "🧪 What-if simulator", "🗂️ Data"])

# ================================================================ 1. command centre
with tabs[0]:
    left, right = st.columns([2.1, 1])
    with left:
        show(charts.sales_trend(data), "trend")
    with right:
        st.markdown("##### 🧠 Smart insights")
        accent = {"critical": CRITICAL, "serious": SERIOUS, "warning": WARNING, "good": GOOD, "info": "#3987e5"}
        box = st.container(height=372, border=False)
        for i in insights:
            box.markdown(f"<div class='insight' style='--accent:{accent[i['level']]}'>"
                         f"<div class='t'>{i['icon']} {i['title']}</div><div class='d'>{i['text']}</div></div>",
                         unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        show(charts.health_gauge(score, score_parts), "gauge")
        show(charts.risk_segment_bar(inv), "riskmix")
    with c2:
        show(charts.value_treemap(inv, height=460), "treemap")
    c3, c4 = st.columns([1.6, 1])
    with c3:
        show(charts.weekday_heatmap(data), "heat")
    with c4:
        show(charts.category_momentum(data), "momentum")

# ================================================================ 2. forecast
with tabs[1]:
    a, b = st.columns([2, 1])
    selected = a.selectbox("Product", sorted(inv.product_name), key="fc_product",
                           index=sorted(inv.product_name).index(inv.sort_values("annual_value", ascending=False).product_name.iloc[0]))
    pid = name_to_id[selected]
    res = forecasts[pid]
    lb = res.leaderboard
    chosen_row = lb[lb.Model == res.method]
    m = st.columns(4)
    m[0].metric("Selected model", res.method, border=True)
    if len(chosen_row):
        r = chosen_row.iloc[0]
        m[1].metric("Backtest WAPE", f"{r.WAPE:.1f}%", help="Weighted absolute % error on the last 14 days, unseen during training.", border=True)
        m[2].metric("Beats seasonal-naive by", f"{r['Skill vs naive %']:.0f}%" if pd.notna(r['Skill vs naive %']) else "—", border=True)
    tot = res.forecast.predicted_demand.sum()
    half = 1.2816 * res.residual_sigma * np.sqrt(horizon)
    m[3].metric(f"Demand · next {horizon} days", f"{tot:,.0f} units",
                f"80% range {max(0, tot - half):,.0f}–{tot + half:,.0f}", delta_color="off", border=True)
    show(charts.forecast_fan(res, selected), "fan")
    c1, c2, c3 = st.columns(3)
    with c1:
        if not lb.empty:
            show(charts.leaderboard_chart(lb, res.method), "lb")
    with c2:
        imp = res.importances if not res.importances.empty else cached_importance(data[data.product_id == pid])
        if not imp.empty:
            show(charts.importance_chart(imp), "imp")
        else:
            st.info("Not enough history for an ML model; using a statistical baseline.")
    with c3:
        show(charts.weekday_profile(res), "dow")
    with st.expander("📋 Forecast table & model leaderboard"):
        t1, t2 = st.columns([1.3, 1])
        t1.dataframe(res.forecast.rename(columns={"predicted_demand": "forecast", "lo80": "80% low", "hi80": "80% high",
                                                   "lo95": "95% low", "hi95": "95% high"}),
                     hide_index=True, column_config={"date": st.column_config.DateColumn("Date", format="ddd DD MMM")})
        t2.dataframe(lb.round(2), hide_index=True)
    st.caption("Models: Random Forest and Gradient Boosting use calendar features plus lagged and rolling sales, "
               "forecasting recursively day by day. Intervals are built from each model's real backtest errors.")

# ================================================================ 3. stockout risk
with tabs[2]:
    c1, c2 = st.columns([1.5, 1])
    with c1:
        show(charts.risk_matrix(inv), "radar")
    with c2:
        show(charts.stockout_prob_bars(inv), "probs")
    st.markdown("##### 📉 Stock depletion simulator")
    risky_first = inv.sort_values("stockout_probability", ascending=False).product_name.tolist()
    sel = st.selectbox("Product", risky_first, key="dep_product")
    row = inv.set_index("product_name").loc[sel]
    d1, d2 = st.columns([2.2, 1])
    with d1:
        win = int(np.clip(np.ceil(row.days_of_cover * 1.5 + row.lead_time_days + 2), 7, 30)) if np.isfinite(row.days_of_cover) else 30
        paths = depletion_paths(forecasts[row.product_id], row.stock_on_hand, row.lead_time_days, policy, days=win)
        show(charts.depletion_chart(paths, row, sel), "deplete")
    with d2:
        st.metric("Stockout probability", f"{row.stockout_probability:.0%}",
                  f"{RISK_ICON[row.stockout_risk]} {row.stockout_risk} risk", delta_color="off", border=True)
        st.metric("Days of cover", "∞" if not np.isfinite(row.days_of_cover) else f"{row.days_of_cover:.1f}",
                  f"Lead time {row.lead_time_days:g} days", delta_color="off", border=True)
        st.markdown(f"""<div class="formula">
        <b>Safety stock</b> = z({service_level:.1%}) × σ × √LT = <b>{row.safety_stock}</b><br>
        <b>Reorder point</b> = μ × LT + SS = {row.avg_daily_forecast:.1f} × {row.lead_time_days:g} + {row.safety_stock} = <b>{row.reorder_point}</b><br>
        <b>Stock on hand</b> = {row.stock_on_hand:,.0f} → {"<b style='color:#d03b3b'>below</b>" if row.stock_on_hand <= row.reorder_point else "above"} reorder point<br>
        <b>Recommended order</b> = <b>{row.recommended_reorder_qty:,}</b> units</div>""", unsafe_allow_html=True)
    st.markdown("##### 🚦 Early-warning list")
    flt = st.segmented_control("Risk level", ["High", "Medium", "Low"], selection_mode="multi", default=["High", "Medium"])
    view = inv[inv.stockout_risk.isin(flt or [])].sort_values("stockout_probability", ascending=False)
    st.dataframe(
        view[["product_name", "category", "stock_on_hand", "avg_daily_forecast", "days_of_cover", "lead_time_days",
              "reorder_point", "stockout_probability", "stockout_risk", "recommended_reorder_qty"]],
        hide_index=True,
        column_config={
            "product_name": "Product", "category": "Category", "stock_on_hand": "On hand",
            "avg_daily_forecast": st.column_config.NumberColumn("Demand/day", format="%.1f"),
            "days_of_cover": st.column_config.NumberColumn("Cover (d)", format="%.1f"),
            "lead_time_days": st.column_config.NumberColumn("Lead time", format="%g d"),
            "reorder_point": "Reorder pt",
            "stockout_probability": st.column_config.ProgressColumn("P(stockout)", format="percent", min_value=0, max_value=1),
            "stockout_risk": "Risk", "recommended_reorder_qty": "Order qty"})

# ================================================================ 4. ABC-XYZ
with tabs[3]:
    st.markdown("<div class='section-note'>ABC ranks products by annual consumption value (A = top 80%). "
                "XYZ ranks demand stability by coefficient of variation (X &lt; 0.25 ≤ Y &lt; 0.5 ≤ Z). "
                "Together they decide how tightly each item should be managed.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.4, 1])
    with c1:
        show(charts.pareto_chart(inv), "pareto")
    with c2:
        show(charts.abc_xyz_matrix(inv), "matrix")
    st.dataframe(inv.sort_values("annual_value", ascending=False)[
        ["product_name", "category", "annual_value", "cv", "abc_xyz", "policy"]],
        hide_index=True,
        column_config={"product_name": "Product", "category": "Category",
                       "annual_value": st.column_config.NumberColumn("Annual value", format="₹%,.0f"),
                       "cv": st.column_config.NumberColumn("Demand CV", format="%.2f"),
                       "abc_xyz": "Class", "policy": "Recommended policy"})

# ================================================================ 5. expiry
with tabs[4]:
    e = st.columns(4)
    exp = inv.dropna(subset=["days_to_expiry"])
    e[0].metric("Expiring ≤ 7 days", int(((exp.days_to_expiry >= 0) & (exp.days_to_expiry <= 7)).sum()), border=True)
    e[1].metric("Already expired", int((exp.days_to_expiry < 0).sum()), border=True)
    e[2].metric("Units forecast to waste", f"{inv.expected_waste_units.sum():,.0f}", border=True)
    e[3].metric("Waste value at risk", inr(waste_val), border=True)
    c1, c2 = st.columns(2)
    with c1:
        show(charts.expiry_timeline(inv, as_of), "timeline")
    with c2:
        show(charts.waste_chart(inv), "waste")
    w = inv[inv.expected_waste_units > 0].sort_values("expected_waste_value", ascending=False)
    if len(w):
        st.markdown("##### 🏷️ Markdown recommendations")
        st.dataframe(w[["product_name", "stock_on_hand", "days_to_expiry", "sellable_before_expiry",
                        "expected_waste_units", "expected_waste_value", "expiry_action"]], hide_index=True,
                     column_config={"product_name": "Product", "stock_on_hand": "On hand",
                                    "days_to_expiry": st.column_config.NumberColumn("Days left", format="%d"),
                                    "sellable_before_expiry": st.column_config.NumberColumn("Will sell", format="%.0f"),
                                    "expected_waste_units": st.column_config.NumberColumn("Unsold", format="%.0f"),
                                    "expected_waste_value": st.column_config.NumberColumn("Value at risk", format="₹%,.0f"),
                                    "expiry_action": "Suggested action"})
        st.caption(f"Markdown size assumes {elasticity:.1f}% extra sales per 1% discount (set in Cost assumptions). "
                   "Expiry is tracked at product level; batch-level data would make this exact.")
    else:
        st.success("No stock is forecast to expire unsold.")

# ================================================================ 6. reorder
with tabs[5]:
    need = inv[inv.recommended_reorder_qty > 0]
    full = float(need.estimated_reorder_cost.sum())
    st.markdown(f"<div class='section-note'>(s, S) policy: when stock falls to the reorder point <i>s</i>, order up to "
                f"<i>S</i> = s + economic lot (EOQ, kept between {review_days} and {max_cover} days of demand). "
                "Perishable orders are capped so they can be sold within ~90% of shelf life.</div>", unsafe_allow_html=True)
    if full <= 0:
        st.success("No product needs reordering right now.")
    else:
        budget = st.slider("Purchase budget (₹)", 0, int(np.ceil(full / 1000) * 1000), int(np.ceil(full * 0.6 / 1000) * 1000),
                           step=1000, help="Orders are funded in priority order: risk level, then probability × ₹ demand × ABC weight.")
        alloc = allocate_budget(inv, budget)
        r = st.columns(4)
        r[0].metric("Full requirement", inr(full), border=True)
        r[1].metric("Funded", inr(alloc.funded_cost.sum()), f"{alloc.funded_cost.sum() / full:.0%} of need", delta_color="off", border=True)
        hi_unfunded = int(((alloc.stockout_risk == "High") & (alloc.status != "Fully funded")).sum())
        r[2].metric("High-risk lines not fully funded", hi_unfunded, "Increase budget" if hi_unfunded else "All covered",
                    delta_color="inverse" if hi_unfunded else "normal", border=True)
        r[3].metric("Lines fully funded", f"{int((alloc.status == 'Fully funded').sum())} / {len(alloc)}", border=True)
        c1, c2 = st.columns([1.6, 1])
        with c1:
            show(charts.budget_chart(alloc), "budget")
        with c2:
            show(charts.spend_by_category(alloc, height=420), "spendcat")
        plan = alloc[["product_name", "category", "stockout_risk", "stock_on_hand", "reorder_point", "safety_stock", "eoq",
                      "recommended_reorder_qty", "funded_qty", "unit_cost", "funded_cost", "shelf_life_capped", "status"]]
        st.dataframe(plan, hide_index=True, column_config={
            "product_name": "Product", "category": "Category", "stockout_risk": "Risk", "stock_on_hand": "On hand",
            "reorder_point": "ROP", "safety_stock": "Safety", "eoq": "EOQ", "recommended_reorder_qty": "Needed",
            "funded_qty": "Order now", "unit_cost": st.column_config.NumberColumn("Unit ₹", format="₹%.0f"),
            "funded_cost": st.column_config.NumberColumn("Cost", format="₹%,.0f"),
            "shelf_life_capped": st.column_config.CheckboxColumn("Shelf-life cap"), "status": "Status"})

        all_fc = pd.concat([forecasts[p].forecast.assign(product_name=n) for n, p in name_to_id.items()])
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            plan.to_excel(xw, sheet_name="Purchase plan", index=False)
            inv.drop(columns=["priority_score"]).to_excel(xw, sheet_name="Inventory snapshot", index=False)
            all_fc.to_excel(xw, sheet_name="Forecasts", index=False)
            pd.DataFrame(insights)[["title", "text"]].to_excel(xw, sheet_name="Insights", index=False)
        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button("⬇️ Purchase plan (CSV)", plan.to_csv(index=False).encode(), "stocksense_purchase_plan.csv",
                           "text/csv", width="stretch")
        d2.download_button("📊 Full report (Excel)", buf.getvalue(), "stocksense_report.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")

# ================================================================ 7. what-if
with tabs[6]:
    st.markdown("<div class='section-note'>Stress-test the store: what happens if demand spikes (festival, heatwave) "
                "or a supplier is late? Everything is re-simulated instantly.</div>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    shock = p1.slider("Demand change", -50, 100, 30, 5, format="%d%%")
    delay = p2.slider("Supplier delay (extra days)", 0, 7, 1)
    sl2 = p3.select_slider("Service level in scenario", options=[0.80, 0.85, 0.90, 0.95, 0.975, 0.99], value=service_level,
                           format_func=lambda v: f"{v:.1%}".replace(".0%", "%"))
    scen_policy = replace(policy, demand_mult=1 + shock / 100, lead_time_add=float(delay), service_level=sl2)
    scen = compute_inventory(data, forecasts, scen_policy)
    s_score, _ = health_score(scen)
    w = st.columns(4)
    w[0].metric("High-risk products", int((scen.stockout_risk == "High").sum()),
                f"{int((scen.stockout_risk == 'High').sum()) - high:+d} vs today", delta_color="inverse", border=True)
    w[1].metric("Replenishment spend", inr(scen.estimated_reorder_cost.sum()),
                f"{signed_inr(scen.estimated_reorder_cost.sum() - reorder_spend)} vs today", delta_color="off", border=True)
    w[2].metric("Waste value at risk", inr(scen.expected_waste_value.sum()),
                f"{signed_inr(scen.expected_waste_value.sum() - waste_val)} vs today", delta_color="inverse", border=True)
    w[3].metric("Health score", f"{s_score:.0f}", f"{s_score - score:+.0f} vs today", border=True)
    c1, c2 = st.columns([1.4, 1])
    with c1:
        show(charts.dumbbell(inv, scen), "dumbbell")
    with c2:
        st.markdown("##### Products that flip to high risk")
        flips = scen.merge(inv[["product_id", "stockout_risk"]], on="product_id", suffixes=("", "_today"))
        flips = flips[(flips.stockout_risk == "High") & (flips.stockout_risk_today != "High")]
        if flips.empty:
            st.success("No additional products become high-risk in this scenario.")
        else:
            st.dataframe(flips[["product_name", "stockout_risk_today", "stockout_probability", "recommended_reorder_qty"]],
                         hide_index=True, column_config={
                             "product_name": "Product", "stockout_risk_today": "Today",
                             "stockout_probability": st.column_config.ProgressColumn("Scenario P(stockout)", format="percent", min_value=0, max_value=1),
                             "recommended_reorder_qty": "Order qty"})

# ================================================================ 8. data
with tabs[7]:
    q = st.columns(5)
    q[0].metric("Rows loaded", f"{dq['clean_rows']:,}", f"{dq['dropped_rows']} dropped", delta_color="off", border=True)
    q[1].metric("Products", dq["products"], border=True)
    q[2].metric("Categories", dq["categories"], border=True)
    q[3].metric("Days of history", dq["days"], border=True)
    q[4].metric("Duplicate product-days", dq["duplicate_product_days"], border=True)
    st.dataframe(data.sort_values(["date", "product_id"], ascending=[False, True]), hide_index=True, height=420)

st.divider()
st.caption("**StockSense AI** · Hackathon prototype · Decision-support estimates only — validate forecasts and "
           "supplier lead times before placing purchase orders.")
