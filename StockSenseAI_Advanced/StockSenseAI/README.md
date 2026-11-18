# 📦 StockSense AI — Intelligent Inventory Command Centre

StockSense AI turns a supermarket's daily sales and stock records into decisions: **what will sell, what will run out, what will expire, and exactly what to order within budget.**

It combines machine-learning demand forecasting, Monte Carlo risk simulation and classical inventory science (safety stock, reorder point, EOQ, ABC-XYZ) in one interactive dark-mode dashboard.

---

## ✨ Features

| Module | What it does | Technique |
|---|---|---|
| 🧭 **Command centre** | KPIs, smart insights, sales trend with anomaly detection, health score gauge, value treemap, demand calendar heatmap, category momentum | Rolling z-score anomalies, composite health index |
| 🔮 **Demand forecast** | Per-product forecast with 80%/95% prediction intervals, model tournament, feature importance, weekly seasonality | Random Forest, Gradient Boosting, weekday average, seasonal-naive — **auto-selected by walk-forward backtest** |
| 🚨 **Stockout risk** | Risk radar, stockout probability per product, stock-depletion fan chart, early-warning list | **Monte Carlo simulation** (5,000 demand paths per product) |
| 📐 **ABC-XYZ** | Pareto curve + 3×3 segmentation matrix with a recommended policy per segment | ABC by annual consumption value, XYZ by coefficient of variation |
| ⏳ **Expiry & waste** | Expiry runway, forecast of stock that will expire unsold, markdown recommendations | Forecast demand until expiry vs stock on hand, price-elasticity markdown sizing |
| 🛒 **Reorder optimizer** | Budget slider → risk-weighted allocation of the purchase budget, CSV + multi-sheet Excel export | (s, S) policy, service-level safety stock, EOQ, shelf-life cap, greedy budget allocation |
| 🧪 **What-if simulator** | Demand shock, supplier delay and service-level stress test with today-vs-scenario comparison | Full re-simulation of the inventory policy |
| 🗂️ **Data** | Data-quality report + raw data browser | Validation, column aliases, duplicate detection |

### The maths in one place
- **Safety stock** `SS = z(service level) × σ × √LT` — σ is the forecast model's own backtest error
- **Reorder point** `ROP = μ × LT + SS`
- **EOQ** `√(2 · annual demand · ordering cost / (holding rate · unit cost))`, kept between the review period and a max-cover limit
- **Order quantity** — when stock ≤ ROP, order up to `ROP + lot`, capped so perishables can be sold within ~90% of shelf life
- **Stockout probability** — P(simulated demand during lead time > stock on hand)
- **Risk level** — High ≥ 50%; Medium when above the service-level target; otherwise Low
- **Health score** — 50% service level + 30% waste avoidance + 20% capital efficiency

---

## 🚀 Quick start

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. run
streamlit run app.py
```
The app opens at http://localhost:8501 with the built-in demo dataset. The first load trains and backtests ~60 models (a few seconds); after that, results are cached.

---

## 📁 Project structure

```
StockSenseAI/
├── app.py                     # Streamlit UI — sidebar, KPIs, 8 tabs
├── stocksense/
│   ├── data.py                # loading, validation, column aliases
│   ├── forecasting.py         # features, model zoo, backtest, intervals
│   ├── inventory.py           # SS/ROP/EOQ, Monte Carlo, expiry, ABC-XYZ, budget
│   ├── insights.py            # rule-based plain-language insight engine
│   ├── charts.py              # all Plotly charts
│   └── theme.py               # colours, Plotly template, CSS
├── .streamlit/config.toml     # dark theme
├── data/sample_supermarket_sales.csv
├── requirements.txt
├── README.md
└── DEMO_SCRIPT.md
```

---

## 🧾 CSV format

One row per product per day.

| Column | Meaning |
|---|---|
| `date` | Sales date (`YYYY-MM-DD`) |
| `product_id` | Unique product identifier |
| `product_name` | Product name |
| `category` | Product category |
| `units_sold` | Units sold that day |
| `stock_on_hand` | Stock on hand (the latest row per product is used as current stock) |
| `unit_cost` | Cost per unit (₹) |
| `lead_time_days` | Supplier lead time in days |
| `expiry_date` | Expiry date of the current stock (`YYYY-MM-DD`) |

Common alternative names (`sku`, `qty_sold`, `stock`, `cost`, `lead_time`, `expiry`, …) are mapped automatically.
With fewer than 56 days of history per product, the app falls back to statistical baselines.

The demo dataset has 120 days × 15 products. Its latest day includes a realistic scenario — a few over-ordered perishables and one overstocked item — so the expiry and overstock features have something to show.

---

## ⚠️ Assumptions & limitations
- Decision support, not autonomous purchasing — verify supplier lead times and quantities before ordering.
- "Today" is the latest date in the data, so historical files work correctly.
- Expiry is tracked at product level; batch-level data (batch ID + quantity + expiry) would make waste forecasts exact.
- Markdown sizes assume a configurable price elasticity (default 1.8).
- Monte Carlo assumes normally distributed daily forecast errors with the spread measured in backtesting.
- Weather, festivals and promotions are not yet model features — easy to add as extra columns.

## 🛠️ Tech stack
Python · Streamlit · Pandas · NumPy · scikit-learn · SciPy · Plotly · OpenPyXL

## 🗺️ Roadmap
Batch-level expiry · festival & weather features · supplier reliability scoring · multi-store transfers · live POS integration · LLM chat assistant over the data.
