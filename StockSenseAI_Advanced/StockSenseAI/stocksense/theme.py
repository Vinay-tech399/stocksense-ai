"""Design tokens + Plotly template + page CSS (dark command-centre look)."""
import plotly.graph_objects as go
import plotly.io as pio

PAGE = "#111110"
SURFACE = "#1a1a19"
SURFACE_2 = "#222220"
INK = "#f0efec"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"
BORDER = "rgba(255,255,255,0.10)"

# Categorical slots (validated for CVD + contrast on the dark surface). Fixed order, never cycled.
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
BLUE = SERIES[0]
ORANGE = SERIES[1]

# Status colours are reserved for meaning (good / warning / serious / critical)
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
RISK_COLORS = {"High": CRITICAL, "Medium": WARNING, "Low": GOOD}
RISK_ICON = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

# Sequential (one hue) – dark end recedes into the dark surface
SEQ_BLUE = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
ORDINAL_ABC = {"A": "#6da7ec", "B": "#2a78d6", "C": "#184f95"}


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def category_colors(categories) -> dict:
    """Colour follows the entity: alphabetical, so filters never repaint survivors."""
    cats = sorted(set(categories))
    return {c: SERIES[i % len(SERIES)] if i < len(SERIES) else MUTED for i, c in enumerate(cats)}


def register_template():
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family='system-ui, -apple-system, "Segoe UI", Roboto, sans-serif', color=INK_2, size=13),
        title=dict(font=dict(color=INK, size=15), x=0, xanchor="left", y=0.97),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        colorway=SERIES,
        margin=dict(l=12, r=16, t=56, b=12),
        xaxis=dict(automargin=True, gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickcolor=AXIS,
                   tickfont=dict(color=MUTED), title=dict(font=dict(color=MUTED)), showgrid=False),
        yaxis=dict(automargin=True, gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickcolor=AXIS,
                   tickfont=dict(color=MUTED), title=dict(font=dict(color=MUTED)), showline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=AXIS, font=dict(color=INK, size=12)),
        hovermode="closest",
        bargap=0.25,
    )
    pio.templates["stocksense"] = t
    pio.templates.default = "stocksense"


PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}

CSS = f"""
<style>
:root {{ --surface:{SURFACE}; --ink:{INK}; --ink2:{INK_2}; --muted:{MUTED}; --border:{BORDER}; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.hero {{
  position: relative; overflow: hidden;
  padding: 26px 30px; border-radius: 18px; margin-bottom: 18px;
  background: radial-gradient(1200px 300px at 0% 0%, rgba(57,135,229,.35), transparent 60%),
              radial-gradient(800px 260px at 100% 100%, rgba(25,158,112,.25), transparent 60%),
              linear-gradient(120deg, #13202f, #141a17);
  border: 1px solid var(--border);
}}
.hero h1 {{ color: var(--ink); margin: 0; font-size: 2.0rem; letter-spacing: -0.02em; }}
.hero p {{ color: var(--ink2); margin: 6px 0 0 0; font-size: 0.98rem; }}
.hero .chips {{ margin-top: 14px; display:flex; gap:8px; flex-wrap:wrap; }}
.hero .chip {{ font-size: .78rem; color: var(--ink2); padding: 4px 10px; border-radius: 999px;
              border: 1px solid var(--border); background: rgba(255,255,255,.04); }}
div[data-testid="stMetric"] {{ background: var(--surface); border-radius: 14px; }}
div[data-testid="stMetricLabel"] p {{ color: var(--muted); font-size: .82rem; text-transform: uppercase; letter-spacing: .04em; }}
div[data-testid="stMetricValue"] {{ color: var(--ink); font-size: 1.85rem; }}
.insight {{ background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--accent);
           border-radius: 12px; padding: 12px 14px; margin-bottom: 10px; }}
.insight .t {{ color: var(--ink); font-weight: 600; font-size: .95rem; }}
.insight .d {{ color: var(--ink2); font-size: .86rem; margin-top: 4px; line-height: 1.4; }}
.section-note {{ color: var(--muted); font-size: .85rem; margin-top: -6px; margin-bottom: 8px; }}
.formula {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 12px 16px;
           color: var(--ink2); font-size: .86rem; line-height: 1.7; }}
.formula b {{ color: var(--ink); }}
div[data-testid="stTabs"] button p {{ font-size: .95rem; }}
.stPlotlyChart {{ border-radius: 14px; overflow: hidden; border: 1px solid var(--border); }}
</style>
"""
