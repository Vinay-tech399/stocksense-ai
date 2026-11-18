<div align="center">

# 📦 StockSense AI

### AI-powered inventory management and demand forecasting for supermarkets

**Know what will sell, what will run out, what will expire and exactly what to reorder, from a single CSV.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50%2B-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-interactive-3F4F75?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-199E70)

**Team InsightX** · Built for **FirstCommit '26**

![StockSense AI dashboard](docs/screenshots/dashboard.png)

</div>

---

## 📌 Table of contents
- [The problem](#-the-problem)
- [Our solution](#-our-solution)
- [Features](#-features)
- [Screenshots](#-screenshots)
- [How it works](#️-how-it-works)
- [Quick start](#-quick-start)
- [Dataset format](#-dataset-format)
- [Project structure](#-project-structure)
- [Tech stack](#️-tech-stack)
- [Results on a test store](#-results-on-a-test-store)
- [Roadmap](#️-roadmap)
- [Team](#-team)

---

## ❗ The problem
Supermarkets still decide what to reorder from past sales, spreadsheets and manual shelf checks. Demand swings with weekends and festivals, so:

- 🛒 **Best-sellers run out** before the next delivery, which means lost sales.
- 📦 **Slow movers pile up**, tying up cash and shelf space.
- ⏳ **Perishables expire unsold**, because nobody sees the risk until a manual check.

**Who faces it:** store managers, inventory managers and warehouse operators who make these decisions every day.

## 💡 Our solution
**StockSense AI** turns a store's daily sales and stock data into clear, explained decisions. It combines **machine-learning demand forecasting**, **Monte Carlo risk simulation** and classical **inventory science** (safety stock, reorder point, EOQ, ABC-XYZ) in one interactive dashboard.

| Question | How StockSense AI answers it |
|---|---|
| **What will sell?** | ML demand forecast for the next 3 to 28 days, with 80% and 95% prediction ranges |
| **What will run out?** | % chance of a stockout before the next delivery, from 5,000 simulated futures |
| **What will expire?** | ₹ value of stock that won't sell before expiry, plus a suggested markdown |
| **What should we reorder?** | Exact order quantities, prioritised by risk within the purchase budget |

---

## ✨ Features

| Module | What it does | Technique |
|---|---|---|
| 🧭 **Command centre** | KPI cards, plain-English smart insights, sales trend with anomaly detection, health score, value treemap, demand heatmap | Rolling z-score anomalies, composite health index |
| 🔮 **Demand forecast** | Per-product forecast with prediction ranges, model leaderboard, feature importance, weekly seasonality | Random Forest, Gradient Boosting, weekday average, seasonal-naive — **auto-selected by backtest** |
| 🚨 **Stockout risk** | Risk radar, stockout probability per product, stock-depletion simulator, early-warning list | **Monte Carlo simulation** (5,000 demand paths per product) |
| 📐 **ABC-XYZ** | Pareto curve and a 3×3 segmentation matrix, with a recommended policy for each segment | ABC by annual value, XYZ by coefficient of variation |
| ⏳ **Expiry & waste** | Expiry runway, forecast of stock that will expire unsold, markdown suggestions | Demand-until-expiry vs stock, price-elasticity markdown sizing |
| 🛒 **Reorder optimizer** | Budget slider with risk-weighted allocation, CSV and multi-sheet Excel export | (s, S) policy, service-level safety stock, EOQ, shelf-life cap |
| 🧪 **What-if simulator** | Stress test with demand spikes, supplier delays and service levels | Full re-simulation of the inventory policy |
| 🗂️ **Data** | Data-quality report and raw data browser | Validation, column aliases, duplicate detection |

---

## 📸 Screenshots

| Demand forecast with prediction ranges | Model tournament |
|---|---|
| ![Forecast](docs/screenshots/forecast.png) | ![Model tournament](docs/screenshots/model-tournament.png) |

| Risk radar (Monte Carlo) | Stock depletion simulator |
|---|---|
| ![Risk radar](docs/screenshots/risk-radar.png) | ![Stock depletion](docs/screenshots/stock-depletion.png) |

| Expiry & waste forecast | ABC-XYZ segmentation |
|---|---|
| ![Expiry and waste](docs/screenshots/expiry-waste.png) | ![ABC-XYZ](docs/screenshots/abc-xyz.png) |

| Budget-aware reorder plan | What-if simulator |
|---|---|
| ![Budget allocation](docs/screenshots/budget-allocation.png) | ![What-if](docs/screenshots/what-if.png) |

📑 **Presentation:** [docs/StockSense_AI_Presentation.pdf](docs/StockSense_AI_Presentation.pdf)

---

## ⚙️ How it works

```
 ┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │ 1 Upload │ → │ 2 Clean  │ → │ 3 Forecast   │ → │ 4 Simulate   │ → │ 5 Recommend  │
 │   CSV    │   │ validate │   │ 4 models     │   │ 5,000 demand │   │ reorder qty, │
 │          │   │ & fix    │   │ compete, best│   │ scenarios →  │   │ markdowns,   │
 │          │   │          │   │ one wins     │   │ P(stockout)  │   │ budget, Excel│
 └──────────┘   └──────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

1. **Forecasting.** For every product, four models are trained and tested on its **last 14 days of unseen sales**. The model with the lowest error (WAPE) makes the forecast. ML models use 11 features: weekday, weekend flag, lags (1, 7 and 14 days), rolling averages and volatility.
2. **Prediction ranges.** The 80% and 95% bands come from the chosen model's real backtest errors.
3. **Risk.** 5,000 simulated demand paths over the supplier lead time give the probability of running out.
4. **Policy.** Order quantities follow the formulas below.

| Quantity | Formula |
|---|---|
| Safety stock | `SS = z(service level) × σ × √LT` |
| Reorder point | `ROP = μ × LT + SS` |
| Economic order quantity | `EOQ = √(2 · annual demand · ordering cost / (holding rate · unit cost))` |
| Order quantity | When stock ≤ ROP, order up to `ROP + lot` (EOQ, capped by max cover and shelf life) |
| Risk level | **High** ≥ 50% stockout chance · **Medium** above the service-level target · **Low** otherwise |
| Health score | 50% service level + 30% waste avoidance + 20% capital efficiency |

---

## 🚀 Quick start

**Requirements:** Python 3.10 or newer.

```bash
# 1. Clone the repository
git clone https://github.com/Vinay-tech399/stocksense-ai.git
cd stocksense-ai

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows (PowerShell)
source .venv/bin/activate         # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

The app opens at **http://localhost:8501** with the built-in demo dataset. The first load trains and backtests the models, which takes a few seconds; after that, results are cached.

> **Windows tip:** if activation fails with *"running scripts is disabled"*, run
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same terminal, then activate again.

**Try your own data:** use **Upload sales & stock CSV** in the sidebar. A second ready-made dataset is in `data/freshmart_hyderabad_sales.csv`.

---

## 🧾 Dataset format
One row per product per day:

| Column | Description |
|---|---|
| `date` | Sales date (`YYYY-MM-DD`) |
| `product_id` | Unique product ID |
| `product_name` | Product name |
| `category` | Product category |
| `units_sold` | Units sold that day |
| `stock_on_hand` | Stock on hand; the latest row is used as current stock |
| `unit_cost` | Cost per unit (₹) |
| `lead_time_days` | Supplier lead time in days |
| `expiry_date` | Expiry date of the current stock (`YYYY-MM-DD`) |

Common alternative names (`sku`, `qty_sold`, `stock`, `cost`, `lead_time`, `expiry`, …) are mapped automatically. Products with fewer than 56 days of history use statistical baselines instead of ML models.

**Included datasets**

| File | Description |
|---|---|
| `data/sample_supermarket_sales.csv` | 15 products × 120 days (built-in demo) |
| `data/freshmart_hyderabad_sales.csv` | 20 products × 150 days, with weekend and festival effects (Ganesh Chaturthi, Dussehra, Diwali) |

Both datasets are synthetic and were created for testing.

---

## 📁 Project structure

```
stocksense-ai/
├── app.py                         # Streamlit UI: sidebar, KPIs and 8 tabs
├── stocksense/
│   ├── data.py                    # Loading, validation, column aliases
│   ├── forecasting.py             # Features, model zoo, backtesting, prediction ranges
│   ├── inventory.py               # SS / ROP / EOQ, Monte Carlo, expiry, ABC-XYZ, budget
│   ├── insights.py                # Rule-based plain-English insight engine
│   ├── charts.py                  # All Plotly charts
│   └── theme.py                   # Colours, Plotly template, CSS
├── data/
│   ├── sample_supermarket_sales.csv
│   └── freshmart_hyderabad_sales.csv
├── docs/
│   ├── screenshots/               # README images
│   └── StockSense_AI_Presentation.pdf
├── .streamlit/config.toml         # Dark theme
├── DEMO_SCRIPT.md                 # 3-minute demo walkthrough
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🛠️ Tech stack

| Layer | Tools |
|---|---|
| Language | Python |
| Dashboard | Streamlit |
| Charts | Plotly |
| Data | Pandas, NumPy |
| Machine learning | Scikit-learn (Random Forest, Histogram Gradient Boosting) |
| Simulation & statistics | NumPy, SciPy |
| Export | OpenPyXL (Excel reports) |

---

## 📊 Results on a test store
Run on `freshmart_hyderabad_sales.csv` (20 products, 150 days), StockSense AI:

- flagged **6 products** at high risk of running out, each with an order quantity;
- found **₹12,675** of stock likely to expire unsold, and suggested a ~40% markdown on post-Diwali sweets;
- spotted **₹24,700** tied up in overstock (post-festival agarbatti);
- backtested **80 models** across all 20 products in about **10 seconds**.

---

## ⚠️ Assumptions & limitations
- This is a decision-support tool: check supplier lead times and quantities before placing orders.
- "Today" is the latest date in the data, so historical files work correctly.
- Expiry is tracked at product level; batch-level data would make waste forecasts exact.
- Markdown sizes assume a configurable price elasticity (default 1.8).
- Monte Carlo assumes normally distributed daily forecast errors, with the spread measured in backtesting.

## 🗺️ Roadmap
- [ ] Festival, weather and promotion signals as model features
- [ ] Batch-level expiry tracking
- [ ] Multi-store view with stock transfers between branches
- [ ] Live POS / ERP integration instead of CSV uploads
- [ ] Production app with a React frontend, FastAPI backend and PostgreSQL

---

## 👥 Team
**Team InsightX** — FirstCommit '26

## 📄 License
Released under the [MIT License](LICENSE).
