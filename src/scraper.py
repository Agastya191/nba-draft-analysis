"""
scraper.py
----------
Scrapes NBA draft data and career statistics from Basketball Reference.
Collects draft picks (2000-2020) and their career Win Shares, PER, VORP.

Rate limiting is built in to respect Basketball Reference's servers.
"""

import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO

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


def _get_soup(url: str) -> BeautifulSoup:
    """Fetch a URL and return parsed BeautifulSoup."""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    time.sleep(RATE_LIMIT)
    return BeautifulSoup(resp.text, "html.parser")


def scrape_draft_year(year: int) -> pd.DataFrame:
    """
    Scrape one NBA draft year from Basketball Reference.

    Parameters
    ----------
    year : int
        Draft year (e.g. 2003)

    Returns
    -------
    pd.DataFrame
        One row per pick with draft position, player, team, and career stats.
    """
    url = DRAFT_URL.format(year=year)
    print(f"  Scraping {year} draft... ({url})")

    soup = _get_soup(url)
    table = soup.find("table", {"id": "stats"})
    if table is None:
        print(f"    No table found for {year}, skipping.")
        return pd.DataFrame()

    # Read HTML table
    df = pd.read_html(StringIO(str(table)))[0]

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(c).strip() for c in df.columns]

    # Rename columns to standard names
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if "pk" in cl or "pick" in cl:
            col_map[col] = "pick"
        elif "player" in cl:
            col_map[col] = "player"
        elif "tm" in cl or "team" in cl:
            col_map[col] = "team"
        elif "college" in cl or "from" in cl:
            col_map[col] = "college"
        elif "yrs" in cl:
            col_map[col] = "years_active"
        elif col.strip().upper() == "G" or cl == "g":
            col_map[col] = "games"
        elif "mp" in cl and "per" not in cl:
            col_map[col] = "minutes"
        elif "per" == cl.strip():
            col_map[col] = "PER"
        elif "ws" == cl.strip() or col.strip() == "WS":
            col_map[col] = "win_shares"
        elif "ws/48" in cl or "ws /48" in cl:
            col_map[col] = "ws_per_48"
        elif "bpm" in cl:
            col_map[col] = "BPM"
        elif "vorp" in cl:
            col_map[col] = "VORP"

    df.rename(columns=col_map, inplace=True)

    # Keep only rows with valid pick numbers
    if "pick" not in df.columns:
        return pd.DataFrame()

    df = df[df["pick"].apply(lambda x: str(x).isdigit())].copy()
    df["pick"] = df["pick"].astype(int)
    df["draft_year"] = year

    # Convert stat columns to numeric
    for col in ["years_active", "games", "minutes", "PER", "win_shares",
                "ws_per_48", "BPM", "VORP"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing career stats with 0 (undrafted/never played)
    for col in ["win_shares", "VORP", "PER", "games"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    keep = [c for c in ["pick", "player", "team", "college", "draft_year",
                         "years_active", "games", "PER", "win_shares",
                         "ws_per_48", "BPM", "VORP"] if c in df.columns]
    return df[keep]


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

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(save_path, index=False)
    print(f"\nSaved {len(combined)} total picks to {save_path}")
    return combined
