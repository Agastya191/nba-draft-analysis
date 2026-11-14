"""
visualize.py
------------
All figures for the NBA draft value analysis.
Saves to outputs/ directory.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
PALETTE    = ["#1d428a", "#c8102e", "#ffc72c", "#007a33", "#552583"]
NBA_BLUE   = "#1d428a"
NBA_RED    = "#c8102e"


def _save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: outputs/{name}.png")
    return path


def set_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "#f8f8f8",
        "axes.edgecolor":   "#CCCCCC",
    })


def plot_pick_value_curve(curve_df: pd.DataFrame,
                          baseline: dict) -> str:
    """
    Scatter of actual median WS per pick + model predicted curve.
    Shows the classic drop-off and where late picks overperform.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.scatter(curve_df["pick"], curve_df["median_ws"],
               color=NBA_BLUE, alpha=0.6, s=40, label="Actual median WS")
    ax.plot(baseline["pick_range"], baseline["expected_ws"],
            color=NBA_RED, linewidth=2.5, label="Model prediction (average WS)")

    ax.axvline(30, color="gray", linestyle="--", alpha=0.5, label="Round 1 / Round 2")
    ax.set_xlabel("Draft Pick Number", fontsize=12)
    ax.set_ylabel("Career Win Shares", fontsize=12)
    ax.set_title("NBA Draft Pick Value Curve (2000–2020)\nMedian Career Win Shares by Pick Number",
                 fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "01_pick_value_curve")


def plot_top_steals(steals_df: pd.DataFrame) -> str:
    """Horizontal bar chart of top 15 draft steals."""
    top15 = steals_df.head(15).copy()
    top15["label"] = top15["player"] + f"\n(Pick " + top15["pick"].astype(str) + ", " + top15["draft_year"].astype(str) + ")"

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top15["label"][::-1], top15["value_over_expected"][::-1],
                   color=NBA_BLUE, edgecolor="white")

    for bar, val in zip(bars, top15["value_over_expected"][::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"+{val:.1f}", va="center", fontsize=9)

    ax.set_xlabel("Value Over Expected (composite score)", fontsize=12)
    ax.set_title("Greatest NBA Draft Steals (2000–2020)\nPlayers Who Most Outperformed Their Draft Slot",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "02_top_steals")


def plot_top_busts(busts_df: pd.DataFrame) -> str:
    """Horizontal bar chart of top 15 draft busts (picks 1-20 only)."""
    top15 = busts_df.head(15).copy()
    top15["label"] = top15["player"] + f"\n(Pick " + top15["pick"].astype(str) + ", " + top15["draft_year"].astype(str) + ")"

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top15["label"], top15["value_over_expected"],
                   color=NBA_RED, edgecolor="white")

    for bar, val in zip(bars, top15["value_over_expected"]):
        ax.text(bar.get_width() - 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", ha="right", fontsize=9, color="white")

    ax.set_xlabel("Value Over Expected (composite score)", fontsize=12)
    ax.set_title("Biggest NBA Draft Busts (2000–2020, Picks 1–20)\nHigh Picks Who Most Underperformed Expectations",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "03_top_busts")


def plot_value_by_tier(tier_df: pd.DataFrame) -> str:
    """Grouped bar chart: median WS and bust/steal % by pick tier."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Median win shares
    sns.barplot(data=tier_df, x="pick_tier", y="median_ws",
                palette="Blues_d", ax=axes[0])
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Median Career Win Shares")
    axes[0].set_title("Median Win Shares by Pick Tier", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=25)

    # Bust vs steal rates
    x = range(len(tier_df))
    w = 0.35
    axes[1].bar([i - w/2 for i in x], tier_df["pct_steals"], w,
                label="% Steals", color=NBA_BLUE)
    axes[1].bar([i + w/2 for i in x], tier_df["pct_busts"], w,
                label="% Busts",  color=NBA_RED)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(tier_df["pick_tier"], rotation=25, ha="right")
    axes[1].set_ylabel("% of picks")
    axes[1].set_title("Bust vs Steal Rate by Pick Tier", fontweight="bold")
    axes[1].legend()

    fig.suptitle("NBA Draft Value by Pick Tier (2000–2020)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "04_value_by_tier")


def plot_team_efficiency(teams_df: pd.DataFrame) -> str:
    """Bar chart: top 15 and bottom 5 teams by average value over expected."""
    top    = teams_df.head(12)
    bottom = teams_df.tail(5)
    plot_df = pd.concat([top, bottom])

    colors = [NBA_BLUE if v >= 0 else NBA_RED
              for v in plot_df["avg_value_over_exp"]]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(plot_df["team"], plot_df["avg_value_over_exp"],
                  color=colors, edgecolor="white")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Team", fontsize=12)
    ax.set_ylabel("Avg Value Over Expected per Pick", fontsize=12)
    ax.set_title("NBA Team Drafting Efficiency (2000–2020)\nAverage Value Over Expected Per Pick",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=40, ha="right")

    blue_patch = mpatches.Patch(color=NBA_BLUE, label="Above average")
    red_patch  = mpatches.Patch(color=NBA_RED,  label="Below average")
    ax.legend(handles=[blue_patch, red_patch])

    fig.tight_layout()
    return _save(fig, "05_team_efficiency")


def plot_second_round_steals(sr_df: pd.DataFrame) -> str:
    """Bar chart of best second-round picks by career win shares."""
    sr_df = sr_df.head(12).copy()
    sr_df["label"] = sr_df["player"] + "\n(#" + sr_df["pick"].astype(str) + ", " + sr_df["draft_year"].astype(str) + ")"

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(sr_df["label"], sr_df["win_shares"],
                  color=NBA_BLUE, edgecolor="white")

    for bar, val in zip(bars, sr_df["win_shares"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.0f}", ha="center", fontsize=9)

    ax.set_ylabel("Career Win Shares", fontsize=12)
    ax.set_title("Best Second-Round Picks (2000–2020)\nThe Ultimate Draft Steals",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    return _save(fig, "06_second_round_steals")


def plot_era_comparison(era_df: pd.DataFrame) -> str:
    """Compare median win shares and bust rate across draft eras."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.barplot(data=era_df, x="era", y="median_ws",
                palette="Blues_d", ax=axes[0])
    axes[0].set_title("Median Win Shares by Era", fontweight="bold")
    axes[0].set_xlabel("")

    sns.barplot(data=era_df, x="era", y="pct_zero_ws",
                palette="Reds_d", ax=axes[1])
    axes[1].set_title("% Picks With Zero Career WS by Era", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("% of picks")

    fig.suptitle("NBA Draft Quality by Era (2000–2020)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "07_era_comparison")


def generate_all_figures(results: dict, model_results: dict) -> list:
    set_style()
    print("Generating figures...")
    paths = [
        plot_pick_value_curve(results["pick_value_curve"], model_results["baseline"]),
        plot_top_steals(results["top_steals"]),
        plot_top_busts(results["top_busts"]),
        plot_value_by_tier(results["value_by_tier"]),
        plot_second_round_steals(results["second_round_steals"]),
        plot_era_comparison(results["value_by_era"]),
    ]
    if not model_results["teams"].empty:
        paths.append(plot_team_efficiency(model_results["teams"]))
    print(f"  {len(paths)} figures saved.")
    return paths
