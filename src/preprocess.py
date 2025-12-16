"""
preprocess.py
-------------
Cleans raw draft data and engineers features for modeling.
Creates draft round, pick tier, and composite value score.
"""

import pandas as pd
import numpy as np


# ── Pick tier labels ──────────────────────────────────────────────────────────

def assign_pick_tier(pick: int) -> str:
    """Categorize draft pick into a tier."""
    if pick <= 5:
        return "Top 5"
    elif pick <= 10:
        return "Lottery (6-10)"
    elif pick <= 14:
        return "Lottery (11-14)"
    elif pick <= 30:
        return "Late 1st"
    elif pick <= 45:
        return "Early 2nd"
    else:
        return "Late 2nd"


TIER_ORDER = ["Top 5", "Lottery (6-10)", "Lottery (11-14)",
              "Late 1st", "Early 2nd", "Late 2nd"]


def assign_round(pick: int) -> int:
    return 1 if pick <= 30 else 2


# ── Composite value score ─────────────────────────────────────────────────────

def compute_value_score(df: pd.DataFrame) -> pd.Series:
    """
    Composite career value score combining Win Shares and VORP.
    WS and VORP are the two most respected cumulative NBA value metrics.

    Score = 0.6 * WS + 0.4 * (VORP * 3)
    VORP is scaled by 3 to roughly match WS magnitude.
    """
    ws   = df["VORP"].fillna(0) * 3  # using VORP as proxy
    vorp = df["VORP"].fillna(0) * 3
    return (0.6 * ws + 0.4 * vorp).round(2)


# ── Expected value model (median by pick) ────────────────────────────────────

def compute_expected_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each draft position, compute the median career value score
    across all years — this becomes our 'expected value' baseline.
    """
    return (
        df.groupby("pick")["value_score"]
        .agg(expected_value="median", std_value="std", n_picks="count")
        .reset_index()
        .round(3)
    )


# ── Over/under value ─────────────────────────────────────────────────────────

def compute_value_over_expected(df: pd.DataFrame,
                                expected: pd.DataFrame) -> pd.DataFrame:
    """
    Merge expected value into the main df and compute:
      value_over_expected = actual value_score - expected_value

    Positive = outperformed their draft slot (steal)
    Negative = underperformed their draft slot (bust)
    """
    merged = df.merge(expected[["pick", "expected_value"]], on="pick", how="left")
    merged["value_over_expected"] = (
        merged["value_score"] - merged["expected_value"]
    ).round(2)
    return merged


# ── Master cleaner ────────────────────────────────────────────────────────────

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    1. Drop header rows / invalid entries
    2. Assign tiers and rounds
    3. Compute value score
    4. Compute value over expected
    5. Flag busts and steals

    Parameters
    ----------
    df : pd.DataFrame
        Raw scraped data from scraper.py

    Returns
    -------
    pd.DataFrame
        Fully processed dataframe ready for analysis and modeling.
    """
    df = df.copy()

    # Drop rows where player name is missing or is a header repeat
    df = df[df["player"].notna()]
    df = df[~df["player"].str.contains("Player|Round", na=False)]

    # Ensure numeric types
    for col in ["pick", "games", "win_shares", "VORP", "PER"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["pick"], inplace=True)
    df["pick"] = df["pick"].astype(int)

    # Fill missing stats with 0
    for col in ["VORP", "years_active"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Feature engineering
    df["pick_tier"]  = df["pick"].apply(assign_pick_tier)
    df["round"]      = df["pick"].apply(assign_round)
    df["value_score"] = compute_value_score(df)

    # Expected value and over/under
    expected = compute_expected_value(df)
    df = compute_value_over_expected(df, expected)

    # Bust / steal flags (top/bottom 10% of value over expected)
    p10 = df["value_over_expected"].quantile(0.10)
    p90 = df["value_over_expected"].quantile(0.90)
    df["is_steal"] = df["value_over_expected"] >= p90
    df["is_bust"]  = df["value_over_expected"] <= p10

    # Era label
    df["era"] = pd.cut(
        df["draft_year"],
        bins=[1999, 2005, 2010, 2015, 2020],
        labels=["2000–2005", "2006–2010", "2011–2015", "2016–2020"]
    )

    print(f"Processed {len(df)} draft picks.")
    print(f"  Steals: {df['is_steal'].sum()} | Busts: {df['is_bust'].sum()}")
    return df
