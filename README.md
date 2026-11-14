# NBA Draft Value Analysis (2000–2020)
### Who Were the Biggest Steals and Busts?

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Agastya191/nba-draft-analysis/blob/main/notebooks/analysis.ipynb)

A data science project analyzing 21 years of NBA draft data to quantify which players over and under performed their draft position, which teams draft best, and whether career value can be predicted from draft slot alone.

---

## Research Questions

1. What career value does each draft position typically produce?
2. Which players massively outperformed their draft slot
3. Which high picks underperformed expectations
4. Which teams are the most and least efficient drafters?
5. How well can draft position predict career value?

---

## Dataset

**Source:** Basketball Reference (scraped via `requests` + `BeautifulSoup`). A snapshot is saved in `data/draft_data_raw.csv` (scraped November 2026), so career totals reflect stats through that date.  
**Coverage:** NBA Drafts 2000–2020 (1,249 picks)  
**Key stats:** Win Shares (WS), VORP, Box Plus/Minus (BPM), Games Played

**Value Metric:** Composite score = `0.6 × WS + 0.4 × (VORP × 3)`  
Combines the two most respected cumulative NBA value metrics.

---

## Methods

| Method | Purpose |
|---|---|
| Web scraping (BeautifulSoup) | Collect draft + career data from Basketball Reference |
| Expected value baseline | Median value score per pick position across all years |
| Value Over Expected (VOE) | Actual − expected → identifies steals and busts |
| Polynomial regression | Predict WS from pick number (baseline model) |
| Gradient Boosting | Predict value from pick + era + round (improved model) |
| Team efficiency ranking | Avg VOE per pick per franchise (relocated/renamed teams merged, e.g. SEA → OKC, NJN → BRK) |

---

## Setup & Usage

**In the browser (no setup):** click the Colab badge above, then `Runtime → Run all`. The notebook downloads the code and scrapes the data itself (about 2 minutes).

**Locally:**
```bash
git clone https://github.com/Agastya191/nba-draft-analysis.git
cd nba-draft-analysis
pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb
```

> **Note:** The first run scrapes 21 pages from Basketball Reference with a 4-second delay between requests. Later runs load the cached CSV in `data/`.

---

## Key Findings

#### Draft Value Curve
- Value drops off steeply with pick number. The median Top 5 pick produced 35.4 career Win Shares, compared with 23.0 for picks 6–10, 10.3 for picks 15–30, and 1.1 for picks 31–45.
- 45% of second-round picks finished with zero or negative career Win Shares, compared with 9% of first-round picks.

#### Greatest Steals
- Measured against what their pick slot usually produces, the biggest outperformers were LeBron James (#1, 2003), Chris Paul (#4, 2005), Kevin Durant (#2, 2007), and Nikola Jokić (#41, 2014).
- Jokić is the standout late pick: 144.5 career Win Shares, the most of any second-rounder and 6th-most of anyone drafted 2000–2020. Other late-pick steals include Rudy Gobert (#27), Tony Parker (#28), Jimmy Butler (#30), and Marc Gasol (#48).

#### Biggest Busts
- Among picks 1–20, the furthest below expectation were Adam Morrison (#3, 2006), Jahlil Okafor (#3, 2015), and Anthony Bennett (#1, 2013).
- Top 5 picks were flagged as busts most often (35%), partly because high expectations leave the most room to fall short.

#### Best Drafting Teams
- Oklahoma City (including its Seattle years) had the highest average value over expected per pick (+14.6), driven by Kevin Durant, Russell Westbrook, and James Harden. New Orleans (+14.4, Chris Paul) and Denver (+13.9, Nikola Jokić) were next.

#### Predicting Career Value
- Draft position alone explains only about 20% of the variation in career Win Shares (cross-validated R² = 0.20, mean error 15.5 WS). Adding draft year and round did not improve predictions (R² = 0.12), so most of what makes a career is not captured by where a player was picked.

---

## Key Figures

| # | Figure |
|---|---|
| 1 | Draft pick value curve — actual vs model predicted |
| 2 | Top 15 greatest draft steals |
| 3 | Top 15 biggest draft busts (picks 1–20) |
| 4 | Win shares and bust/steal rate by pick tier |
| 5 | Best second-round picks of the era |
| 6 | Draft quality comparison by era |
| 7 | Team drafting efficiency ranking |

---

## Limitations

- Win Shares and VORP are career totals, so players drafted recently (especially 2016–2020) have had fewer seasons to accumulate value. Their picks can look like busts partly because their careers aren't finished.
- Players who never appeared in an NBA game are counted as 0 Win Shares and 0 VORP.

---

## License & Data

Statistics sourced from [Basketball Reference](https://www.basketball-reference.com) for educational/research use.
