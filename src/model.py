"""
model.py
--------
Predicts career Win Shares from draft position alone (baseline),
then from draft position + additional features (improved model).

Also computes draft efficiency scores per team and era.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# ── Baseline model: pick position only ───────────────────────────────────────

def train_baseline_model(df: pd.DataFrame) -> dict:
    """
    Baseline: predict career Win Shares from draft pick number alone.
    Uses polynomial regression (degree 2) to capture the non-linear
    drop-off in value after the top picks.

    Returns dict with model, predictions, and evaluation metrics.
    """
    sub = df[["pick", "VORP"]].dropna()
    X = sub[["pick"]].values
    y = sub["VORP"].values

    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("ridge", Ridge(alpha=10.0)),
    ])

    # Cross-validated metrics
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae = -cross_val_score(pipe, X, y, cv=kf,
                               scoring="neg_mean_absolute_error").mean()
    cv_r2  =  cross_val_score(pipe, X, y, cv=kf, scoring="r2").mean()

    pipe.fit(X, y)

    # Predict expected WS for each pick slot 1–60
    pick_range = np.arange(1, 61).reshape(-1, 1)
    expected_ws = pipe.predict(pick_range)
    expected_ws = np.clip(expected_ws, 0, None)   # no negative WS predictions

    return {
        "model":       pipe,
        "cv_mae":      round(cv_mae, 2),
        "cv_r2":       round(cv_r2, 3),
        "pick_range":  pick_range.flatten(),
        "expected_ws": expected_ws,
    }


# ── Improved model: pick + era features ──────────────────────────────────────

def train_improved_model(df: pd.DataFrame) -> dict:
    """
    Improved model: predict career value_score from:
    - draft pick number
    - draft round (1 or 2)
    - draft year (era context)

    Uses Gradient Boosting for non-linear interactions.
    """
    feature_cols = ["pick", "round", "draft_year"]
    sub = df[feature_cols + ["value_score"]].dropna()

    X = sub[feature_cols].values
    y = sub["value_score"].values

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae = -cross_val_score(model, X, y, cv=kf,
                               scoring="neg_mean_absolute_error").mean()
    cv_r2  =  cross_val_score(model, X, y, cv=kf, scoring="r2").mean()

    model.fit(X, y)

    feature_importance = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    return {
        "model":              model,
        "cv_mae":             round(cv_mae, 2),
        "cv_r2":              round(cv_r2, 3),
        "feature_importance": feature_importance,
        "feature_cols":       feature_cols,
    }


# ── Team draft efficiency ─────────────────────────────────────────────────────

def team_draft_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team, compute:
    - Total picks made
    - Average value over expected per pick
    - Total value over expected (drafting alpha)
    - Best pick (highest value over expected)

    Teams with consistently positive value_over_expected are
    the best drafting organizations.
    """
    if "team" not in df.columns:
        return pd.DataFrame()

    team_df = (
        df.groupby("team")
        .agg(
            total_picks          = ("player", "count"),
            avg_value_over_exp   = ("value_over_expected", "mean"),
            total_value_over_exp = ("value_over_expected", "sum"),
        )
        .round(2)
        .reset_index()
    )

    # Best steal per team
    best = (
        df.loc[df.groupby("team")["value_over_expected"].idxmax(),
               ["team", "player", "value_over_expected"]]
        .rename(columns={"player": "best_pick",
                          "value_over_expected": "best_pick_voe"})
    )

    team_df = team_df.merge(best, on="team", how="left")
    team_df.sort_values("avg_value_over_exp", ascending=False, inplace=True)
    team_df = team_df[team_df["total_picks"] >= 10]   # min 10 picks for fairness
    return team_df.reset_index(drop=True)


# ── Master runner ─────────────────────────────────────────────────────────────

def run_models(df: pd.DataFrame) -> dict:
    """Run all models and return results."""
    print("Training models...")

    baseline = train_baseline_model(df)
    improved = train_improved_model(df)
    teams    = team_draft_efficiency(df)

    print(f"  Baseline model  — CV MAE: {baseline['cv_mae']} WS | R²: {baseline['cv_r2']}")
    print(f"  Improved model  — CV MAE: {improved['cv_mae']} VS | R²: {improved['cv_r2']}")
    print(f"  Team efficiency — {len(teams)} teams ranked")

    return {
        "baseline": baseline,
        "improved": improved,
        "teams":    teams,
    }
