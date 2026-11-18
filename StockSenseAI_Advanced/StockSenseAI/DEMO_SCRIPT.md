# 🎤 3-minute demo script

**0:00–0:20 — Problem**
Supermarkets lose money in two opposite ways: empty shelves (lost sales) and expired stock (waste). Most stores reorder by gut feel. StockSense AI turns daily sales records into data-backed decisions.

**0:20–0:50 — Command centre** *(tab 1)*
- The six KPI tiles summarise the store: inventory value, sales momentum, health score, high-risk items, waste at risk and the budget needed.
- **Smart insights** (right) explain the situation in plain language. Read the top red card aloud.
- The trend chart flags **anomalies** automatically (orange dots). The treemap shows where the money sits; the calendar heatmap shows weekends selling more.

**0:50–1:25 — Forecasting** *(tab 2)*
- Choose a product. For every product, **four models compete**, each tested on 14 days it never saw during training. The most accurate one wins (model tournament chart).
- Point at the shaded **80% / 95% prediction intervals**: we forecast a range, not just a single number.
- Feature importance shows *why* — e.g. day of week and recent sales.

**1:25–1:55 — Stockout risk** *(tab 3)*
- Risk radar: top-left bubbles = little stock cover and a high chance of running out.
- "We simulate **5,000 possible futures** per product." Open the depletion simulator for Orange Juice: stock crosses zero before a new order could arrive.
- The formula card shows safety stock, reorder point and order quantity — transparent, not a black box.

**1:55–2:20 — Expiry & ABC-XYZ** *(tabs 5 and 4)*
- Waste forecast: red = stock that will expire unsold. The app suggests a markdown size, e.g. "Yogurt — mark down ~15%".
- ABC-XYZ: 9 products drive most of the spend, so they get the tightest control.

**2:20–2:45 — Reorder optimizer + what-if** *(tabs 6 and 7)*
- Move the **budget slider**: high-risk, high-value items are funded first. Download the Excel report.
- What-if: set demand +30% (festival) and supplier delay +1 day. Show how many products flip to high risk.

**2:45–3:00 — Close**
"StockSense AI combines ML forecasting, Monte Carlo risk and inventory science in one working tool that runs on any store's CSV. Next steps: batch-level expiry, festival and weather signals, and live POS integration."

---
### Likely judge questions
- **How do you know the forecast is good?** Every product is backtested on its last 14 days; we show WAPE and how much each model beats the seasonal-naive baseline.
- **Why Monte Carlo?** Average demand hides risk. Simulation gives a real *probability* of stockout, tied to the store's chosen service level.
- **What if a store's data is small?** Under 56 days, the app switches automatically to statistical baselines.
- **Is it a black box?** No. Formulas, feature importance and backtest errors are all shown in the app.
