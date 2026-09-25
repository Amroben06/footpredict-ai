import os
import sys
import io
import logging
import requests
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

URL = "https://en.wikipedia.org/wiki/2023%E2%80%9324_Premier_League"


def scrape_premier_league_table():
    logging.info("Starting web scrape from Wikipedia...")
    try:
        response = requests.get(URL, headers=HEADERS, timeout=15)
        response.raise_for_status()

        # Step 1: Use BeautifulSoup to locate the exact HTML table
        soup = BeautifulSoup(response.text, "html.parser")
        tables = soup.find_all("table", {"class": "wikitable"})

        target_table = None
        for table in tables:
            header_text = table.text
            if "Pts" in header_text and "W" in header_text and "D" in header_text:
                target_table = str(table)
                break

        if not target_table:
            logging.error("Could not find the standings table on the page.")
            return

        # Step 2: Parse table HTML into a clean DataFrame
        dfs = pd.read_html(io.StringIO(target_table))
        df = dfs[0]

        # Standardize column names (lower-case and strip whitespace)
        df.columns = [str(col).strip().lower() for col in df.columns]

        # Locate team column dynamically (usually 'team' or 'team (v · t · e)')
        team_col = next((col for col in df.columns if "team" in col), None)
        if not team_col:
            logging.error(f"Could not identify team column among: {df.columns.tolist()}")
            return

        # Keep relevant stats: team, played, won, drawn, lost, gf, ga, gd, pts
        rename_map = {
            team_col: "team_name",
            "pld": "played",
            "w": "won",
            "d": "drawn",
            "l": "lost",
            "gf": "goals_for",
            "ga": "goals_against",
            "gd": "goal_difference",
            "pts": "points"
        }
        df = df.rename(columns=rename_map)

        # Retain only required columns that exist
        keep_cols = [c for c in rename_map.values() if c in df.columns]
        df = df[keep_cols].copy()

        # Clean team names: strip footnote brackets like '[a]' and '(C)'
        df["team_name"] = df["team_name"].astype(str).str.replace(r"\(.*?\)|\[.*?\]", "", regex=True).str.strip()

        # Filter out invalid rows (like summary footers)
        df = df[df["team_name"].str.len() > 2].reset_index(drop=True)

        os.makedirs(os.path.join("data", "raw"), exist_ok=True)
        output_csv = os.path.join("data", "raw", "scraped_team_stats.csv")
        df.to_csv(output_csv, index=False, encoding="utf-8")
        logging.info(f"Successfully scraped {len(df)} teams and saved to {output_csv}")

    except Exception as e:
        logging.error(f"Scraping error: {e}", exc_info=True)


if __name__ == "__main__":
    scrape_premier_league_table()