import os
import sys
import json
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

RAW_MATCHES_PATH = os.path.join("data", "raw", "PL_matches.json")
RAW_STATS_PATH = os.path.join("data", "raw", "scraped_team_stats.csv")
OUTPUT_PATH = os.path.join("data", "processed", "clean_matches.csv")


def clean_team_name(name: str) -> str:
    """Normalize team names to ensure clean join across disparate sources."""
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    replacements = {
        "fc": "",
        "afc": "",
        "football club": "",
        "hotspur": "",
        "albion": "",
        "wanderers": "",
        "& hove": "",
        "city": "",
        "united": "utd",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return " ".join(name.split())


def aggregate_pipeline():
    logging.info("Starting aggregation and cleaning pipeline (C3)...")

    # 1. Load API Matches
    if not os.path.exists(RAW_MATCHES_PATH):
        logging.error(f"Missing {RAW_MATCHES_PATH}. Please run extract_api.py first.")
        sys.exit(1)

    with open(RAW_MATCHES_PATH, "r", encoding="utf-8") as f:
        matches_data = json.load(f)

    raw_matches = matches_data.get("matches", [])
    logging.info(f"Loaded {len(raw_matches)} matches from API.")

    records = []
    for m in raw_matches:
        # Filter: keep finished matches with complete score records
        if m.get("status") == "FINISHED" and m.get("score", {}).get("fullTime", {}).get("home") is not None:
            home_team = m["homeTeam"]["name"]
            away_team = m["awayTeam"]["name"]
            home_goals = m["score"]["fullTime"]["home"]
            away_goals = m["score"]["fullTime"]["away"]

            # Target outcome for our ML model
            if home_goals > away_goals:
                outcome = "HOME_WIN"
            elif home_goals < away_goals:
                outcome = "AWAY_WIN"
            else:
                outcome = "DRAW"

            records.append({
                "match_id": m.get("id"),
                "utc_date": m.get("utcDate"),
                "matchday": m.get("matchday"),
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": int(home_goals),
                "away_goals": int(away_goals),
                "outcome": outcome,
                "home_norm": clean_team_name(home_team),
                "away_norm": clean_team_name(away_team)
            })

    df_matches = pd.DataFrame(records)
    logging.info(f"Filtered to {len(df_matches)} finished valid matches.")

    # 2. Load Scraped Table Stats
    if os.path.exists(RAW_STATS_PATH):
        df_stats = pd.read_csv(RAW_STATS_PATH)
        df_stats["team_norm"] = df_stats["team_name"].apply(clean_team_name)

        # Merge home team statistics
        df_merged = df_matches.merge(
            df_stats[["team_norm", "points", "goal_difference"]],
            left_on="home_norm",
            right_on="team_norm",
            how="left"
        ).rename(columns={
            "points": "home_points",
            "goal_difference": "home_goal_diff"
        }).drop(columns=["team_norm"])

        # Merge away team statistics
        df_merged = df_merged.merge(
            df_stats[["team_norm", "points", "goal_difference"]],
            left_on="away_norm",
            right_on="team_norm",
            how="left"
        ).rename(columns={
            "points": "away_points",
            "goal_difference": "away_goal_diff"
        }).drop(columns=["team_norm", "home_norm", "away_norm"])
    else:
        logging.warning("Scraped stats not found, proceeding with API fields only.")
        df_merged = df_matches.drop(columns=["home_norm", "away_norm"])

    # 3. Clean Missing Values and Formatting
    df_merged = df_merged.fillna({
        "home_points": 0,
        "away_points": 0,
        "home_goal_diff": 0,
        "away_goal_diff": 0
    })

    # Save processed dataset
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    df_merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    logging.info(f"Aggregation complete! Saved {len(df_merged)} clean records to {OUTPUT_PATH}")


if __name__ == "__main__":
    aggregate_pipeline()