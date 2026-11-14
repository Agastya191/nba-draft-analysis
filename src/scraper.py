"""
scraper.py
----------
Scrapes NBA draft data and career statistics from Basketball Reference.
Collects draft picks (2000-2020) and their career Win Shares, VORP, BPM, and games played.

Rate limiting is built in to respect Basketball Reference's servers.
"""

import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

BASE_URL = "https://www.basketball-reference.com"
DRAFT_URL = BASE_URL + "/draft/NBA_{year}.html"

# Seconds to wait between requests — be respectful
RATE_LIMIT = 4.0


def _get_html(url: str) -> str:
    """Fetch a URL and return its HTML, decoded as UTF-8 so names like Jokić survive."""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    time.sleep(RATE_LIMIT)
    return resp.content.decode("utf-8", errors="replace")


# Basketball Reference tags every table cell with a stable `data-stat` name,
# so we read cells by that name instead of guessing from the visible headers.
STAT_MAP = {
    "pick_overall": "pick",
    "team_id":      "team",
    "player":       "player",
    "college_name": "college",
    "seasons":      "years_active",
    "g":            "games",
    "mp":           "minutes",
    "ws":           "win_shares",
    "ws_per_48":    "ws_per_48",
    "bpm":          "BPM",
    "vorp":         "VORP",
}
NUMERIC_COLS = ["years_active", "games", "minutes", "win_shares", "ws_per_48", "BPM", "VORP"]


def parse_draft_html(html: str, year: int) -> pd.DataFrame:
    """Parse the draft table (id="stats") from a Basketball Reference draft page."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": "stats"})
    if table is None or table.tbody is None:
        return pd.DataFrame()

    rows = []
    for tr in table.tbody.find_all("tr"):
        # Skip the repeated header rows between rounds
        if "thead" in (tr.get("class") or []):
            continue
        row = {}
        for cell in tr.find_all(["th", "td"]):
            key = STAT_MAP.get(cell.get("data-stat"))
            if key:
                row[key] = cell.get_text(strip=True)
        if row.get("pick", "").isdigit():
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["pick"] = df["pick"].astype(int)
    df["draft_year"] = year
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace("", None), errors="coerce")

    # Players who never appeared in an NBA game have blank stats: count them as 0
    for col in ["years_active", "games", "minutes", "win_shares", "VORP"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    keep = [c for c in ["pick", "player", "team", "college", "draft_year",
                        "years_active", "games", "minutes", "win_shares",
                        "ws_per_48", "BPM", "VORP"] if c in df.columns]
    return df[keep]


def scrape_draft_year(year: int) -> pd.DataFrame:
    """
    Scrape one NBA draft year from Basketball Reference.

    Returns one row per pick with draft position, player, team, and career stats.
    """
    url = DRAFT_URL.format(year=year)
    print(f"  Scraping {year} draft... ({url})")
    df = parse_draft_html(_get_html(url), year)
    if df.empty:
        print(f"    No draft table found for {year}, skipping.")
    return df


def scrape_all_drafts(start: int = 2000, end: int = 2020,
                      save_path: str = "data/draft_data_raw.csv") -> pd.DataFrame:
    """
    Scrape all NBA drafts from start to end year inclusive.
    Saves raw CSV to save_path and returns combined DataFrame.

    Parameters
    ----------
    start : int
        First draft year to scrape.
    end : int
        Last draft year to scrape (inclusive).
    save_path : str
        Path to save the raw CSV.

    Returns
    -------
    pd.DataFrame
        Combined data for all years.
    """
    print(f"Scraping NBA drafts {start}–{end}...")
    frames = []
    for year in range(start, end + 1):
        try:
            df = scrape_draft_year(year)
            if not df.empty:
                frames.append(df)
                print(f"    {year}: {len(df)} picks")
        except Exception as e:
            print(f"    {year}: ERROR — {e}")

    if not frames:
        raise RuntimeError(
            "No draft pages could be downloaded. Basketball Reference often blocks "
            "cloud servers like Google Colab (403 Forbidden); run the scraper on your "
            "own computer, or use the data/draft_data_raw.csv snapshot in the repo."
        )
    combined = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    combined.to_csv(save_path, index=False, encoding="utf-8")
    print(f"\nSaved {len(combined)} total picks to {save_path}")
    return combined
