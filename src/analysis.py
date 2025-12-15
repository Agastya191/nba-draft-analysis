"""
analysis.py
-----------
Identifies the biggest draft steals and busts in NBA history (2000-2020).
Computes pick value curves, era comparisons, and positional analysis.
"""

import pandas as pd
import numpy as np
from preprocess import TIER_ORDER


def top_steals(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    Top N draft steals — players who massively outperformed
    their draft position's expected value.
    """
    cols = ["player", "draft_year", "pick", "pick_tier", "team",
            "win_shares", "VORP", "value_score", "expected_value",
            "value_over_expected"]
    available = [c for c in cols if c in df.columns]
    return (
        df[available]
        .sort_values("value_over_expected", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def top_busts(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    Top N draft busts — high picks who severely underperformed expectations.
    Focus on picks 1-20 (busts only matter when high capital was spent).
    """
    cols = ["player", "draft_year", "pick", "pick_tier", "team",
            "win_shares", "VORP", "value_score", "expected_value",
            "value_over_expected"]
    available = [c for c in cols if c in df.columns]
    bust_df = df[df["pick"] <= 20]
    return (
        bust_df[available]
        .sort_values("value_over_expected", ascending=True)
        .head(n)
        .reset_index(drop=True)
    )


def value_by_pick_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Average value metrics grouped by pick tier."""
    tiers = [t for t in TIER_ORDER if t in df["pick_tier"].values]
    result = (
        df.groupby("pick_tier")
        .agg(
            n_picks          = ("player", "count"),
            median_ws        = ("win_shares", "median"),
            median_vorp      = ("VORP", "median"),
            median_value     = ("value_score", "median"),
            pct_busts        = ("is_bust", "mean"),
            pct_steals       = ("is_steal", "mean"),
        )
        .round(3)
        .reindex(tiers)
        .reset_index()
    )
    result["pct_busts"]  = (result["pct_busts"]  * 100).round(1)
    result["pct_steals"] = (result["pct_steals"] * 100).round(1)
    return result


def value_by_era(df: pd.DataFrame) -> pd.DataFrame:
    """Compare draft value across eras."""
    return (
        df.groupby("era", observed=True)
        .agg(
            n_picks        = ("player", "count"),
            median_ws      = ("win_shares", "median"),
            median_vorp    = ("VORP", "median"),
            pct_zero_ws    = ("win_shares", lambda x: (x == 0).mean() * 100),
        )
        .round(2)
        .reset_index()
    )


def pick_value_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Median win shares and value score by pick number (1-60).
    Shows the draft value drop-off curve.
    """
    return (
        df.groupby("pick")
        .agg(
            median_ws    = ("win_shares", "median"),
            median_value = ("value_score", "median"),
            n            = ("player", "count"),
        )
        .round(3)
        .reset_index()
    )


def second_round_steals(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Best second-round picks — the ultimate draft steals."""
    cols = ["player", "draft_year", "pick", "team",
            "win_shares", "VORP", "value_score", "value_over_expected"]
    available = [c for c in cols if c in df.columns]
    return (
        df[df["round"] == 2][available]
        .sort_values("win_shares", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def run_all_analyses(df: pd.DataFrame) -> dict:
    """Run all analyses and return as a dict."""
    print("Running analyses...")
    results = {
        "top_steals":         top_steals(df, 20),
        "top_busts":          top_busts(df, 20),
        "value_by_tier":      value_by_pick_tier(df),
        "value_by_era":       value_by_era(df),
        "pick_value_curve":   pick_value_curve(df),
        "second_round_steals": second_round_steals(df, 15),
    }
    print("  All analyses complete.")
    return results
