"""
ChurnMaster Pro — services/chart_service.py
============================================
Matplotlib chart generation, extracted from customer_churn's app.py and
upgraded with the dark colour palette used across the entire UI.

All charts are returned as base64-encoded PNG strings ready for embedding
in HTML templates as:  <img src="data:image/png;base64,{{ chart }}">

Why a separate service?
  Matplotlib's Agg backend must be set before pyplot is imported, and chart
  code is substantial enough to pollute route handlers. Keeping it here
  makes it independently testable and easy to swap for a different library.
"""

import io
import base64
import logging

import matplotlib
matplotlib.use("Agg")           # must come BEFORE plt import
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# ── Dark theme palette (matches static/css/style.css) ─────────────────────
_BG     = "#0c1117"
_CARD   = "#1a2235"
_TEXT   = "#d1d5db"
_CYAN   = "#00d4ff"
_GREEN  = "#10b981"
_RED    = "#ef4444"
_AMBER  = "#f59e0b"
_PURPLE = "#8b5cf6"


def generate_all_charts(input_dict: dict, probability, prediction: int) -> dict:
    """
    Generate 3 dark-themed charts for a prediction result page.

    Args:
        input_dict:  the raw feature dict passed by the route
        probability: the predict_proba() output array [no_churn, churn]
        prediction:  0 or 1

    Returns:
        {"pie_chart": "...", "bar_chart": "...", "feature_chart": "..."}
    """
    charts = {}
    try:
        charts["pie_chart"]     = _pie_chart(probability, prediction)
        charts["bar_chart"]     = _bar_chart(probability)
        charts["feature_chart"] = _feature_chart(input_dict)
    except Exception as exc:
        logger.error("Chart generation failed: %s", exc)
        # Return empty strings rather than crashing the result page
        charts.setdefault("pie_chart",     "")
        charts.setdefault("bar_chart",     "")
        charts.setdefault("feature_chart", "")
    return charts


# ══════════════════════════════════════════════════════════════════════════
#  INDIVIDUAL CHART GENERATORS
# ══════════════════════════════════════════════════════════════════════════

def _pie_chart(probability, prediction: int) -> str:
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=_BG)
    ax.set_facecolor(_BG)

    sizes   = [probability[1] * 100, probability[0] * 100]
    colors  = [_RED, _GREEN]
    explode = (0.08, 0) if prediction == 1 else (0, 0.08)

    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode,
        labels=["Churn Risk", "No Churn"],
        colors=colors, autopct="%1.1f%%",
        shadow=True, startangle=90,
        textprops={"fontsize": 11, "fontweight": "bold", "color": _TEXT},
    )
    for at in autotexts:
        at.set_color("white")

    ax.set_title("Churn Probability Distribution", fontsize=14,
                 fontweight="bold", color="white", pad=16)
    return _save()


def _bar_chart(probability) -> str:
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=_BG)
    ax.set_facecolor(_CARD)

    values = [probability[0] * 100, probability[1] * 100]
    bars = ax.bar(
        ["No Churn", "Churn Risk"], values,
        color=[_GREEN, _RED], alpha=0.9,
        edgecolor="#374151", linewidth=1.5, width=0.45,
    )
    ax.set_ylabel("Probability (%)", fontsize=11, fontweight="bold", color=_TEXT)
    ax.set_title("Probability Breakdown", fontsize=14, fontweight="bold", color="white")
    ax.set_ylim(0, 115)
    ax.tick_params(colors=_TEXT, labelsize=11)
    ax.yaxis.grid(True, color="#374151", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#374151")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + 1.5,
            f"{val:.1f}%", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="white",
        )
    return _save()


def _feature_chart(input_dict: dict) -> str:
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    internet_map = {"No": 0, "DSL": 1, "Fiber optic": 2}

    labels = ["Tenure (mo)", "Monthly ($)", "Total ($) /100", "Contract", "Internet"]
    values = [
        float(input_dict.get("tenure", 0)),
        float(input_dict.get("MonthlyCharges", 0)),
        min(float(input_dict.get("TotalCharges", 0)) / 100, 100),
        float(contract_map.get(input_dict.get("Contract", "Month-to-month"), 0)),
        float(internet_map.get(input_dict.get("InternetService", "No"), 0)),
    ]
    bar_colors = [_CYAN, _AMBER, _AMBER, _PURPLE, _GREEN]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=_BG)
    ax.set_facecolor(_CARD)

    bars = ax.barh(labels, values, color=bar_colors, alpha=0.85,
                   edgecolor="#374151", linewidth=1.2, height=0.5)
    ax.set_xlabel("Value", fontsize=11, fontweight="bold", color=_TEXT)
    ax.set_title("Key Customer Profile Metrics", fontsize=14,
                 fontweight="bold", color="white")
    ax.tick_params(colors=_TEXT, labelsize=10)
    ax.xaxis.grid(True, color="#374151", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#374151")

    for bar, val in zip(bars, values):
        ax.text(
            val + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", ha="left", va="center",
            fontsize=10, fontweight="bold", color="white",
        )
    return _save()


# ══════════════════════════════════════════════════════════════════════════
#  SHARED SAVE HELPER
# ══════════════════════════════════════════════════════════════════════════

def _save() -> str:
    """Save current plt figure to base64 PNG string and close the figure."""
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=plt.gcf().get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    return encoded
