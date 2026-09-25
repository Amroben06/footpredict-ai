import os
import sys
import re
import logging
import sqlite3
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

DB_PATH = os.path.join("database", "footpredict.db")
PROCESSED_CSV = os.path.join("data", "processed", "clean_matches.csv")


def clean_int(value) -> int:
    """Strip footnotes like '[d]' or non-digit chars and return safe integer."""
    if pd.isna(value):
        return 0
    clean_str = re.sub(r"[^\d\-+]", "", str(value))
    try:
        return int(clean_str)
    except (ValueError, TypeError):
        return 0


def init_database():
    logging.info("Initializing relational database schema (C4)...")
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create tables based on Physical Data Model (MPD)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS teams (
        id_team INTEGER PRIMARY KEY AUTOINCREMENT,
        team_name TEXT UNIQUE NOT NULL,
        points INTEGER DEFAULT 0,
        goal_difference INTEGER DEFAULT 0
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS matches (
        id_match INTEGER PRIMARY KEY,
        utc_date TEXT NOT NULL,
        matchday INTEGER NOT NULL,
        home_team_id INTEGER NOT NULL,
        away_team_id INTEGER NOT NULL,
        home_goals INTEGER NOT NULL,
        away_goals INTEGER NOT NULL,
        outcome TEXT NOT NULL,
        FOREIGN KEY (home_team_id) REFERENCES teams(id_team),
        FOREIGN KEY (away_team_id) REFERENCES teams(id_team)
    );
    """)
    conn.commit()

    # Load cleaned data
    if not os.path.exists(PROCESSED_CSV):
        logging.error(f"Missing {PROCESSED_CSV}. Run pipeline/clean_and_aggregate.py first.")
        conn.close()
        sys.exit(1)

    df = pd.read_csv(PROCESSED_CSV)

    # 1. Populate teams
    teams = pd.concat([
        df[["home_team", "home_points", "home_goal_diff"]].rename(
            columns={"home_team": "name", "home_points": "pts", "home_goal_diff": "gd"}
        ),
        df[["away_team", "away_points", "away_goal_diff"]].rename(
            columns={"away_team": "name", "away_points": "pts", "away_goal_diff": "gd"}
        )
    ]).drop_duplicates(subset=["name"])

    for _, row in teams.iterrows():
        cursor.execute("""
        INSERT OR IGNORE INTO teams (team_name, points, goal_difference)
        VALUES (?, ?, ?)
        """, (row["name"], clean_int(row["pts"]), clean_int(row["gd"])))
    conn.commit()

    # Get team ID lookup map
    cursor.execute("SELECT team_name, id_team FROM teams")
    team_map = dict(cursor.fetchall())

    # 2. Populate matches
    for _, row in df.iterrows():
        cursor.execute("""
        INSERT OR REPLACE INTO matches (
            id_match, utc_date, matchday, home_team_id, away_team_id, home_goals, away_goals, outcome
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            clean_int(row["match_id"]),
            str(row["utc_date"]),
            clean_int(row["matchday"]),
            team_map[row["home_team"]],
            team_map[row["away_team"]],
            clean_int(row["home_goals"]),
            clean_int(row["away_goals"]),
            str(row["outcome"])
        ))
    conn.commit()
    logging.info(f"Database successfully populated at {DB_PATH}")

    # Analytical SQL query for Criterion C2
    logging.info("Executing analytical SQL query (Criterion C2)...")
    query = """
    SELECT 
        t.team_name,
        COUNT(m.id_match) AS matches_played,
        SUM(CASE WHEN (m.home_team_id = t.id_team AND m.outcome = 'HOME_WIN') 
                   OR (m.away_team_id = t.id_team AND m.outcome = 'AWAY_WIN') THEN 1 ELSE 0 END) AS wins,
        t.points,
        t.goal_difference
    FROM teams t
    JOIN matches m ON t.id_team = m.home_team_id OR t.id_team = m.away_team_id
    GROUP BY t.team_name
    ORDER BY t.points DESC
    LIMIT 5;
    """
    sample_df = pd.read_sql_query(query, conn)
    print("\n--- Top 5 Teams by Query (SQL Validation C2) ---")
    print(sample_df.to_string(index=False))
    print("------------------------------------------------\n")

    conn.close()


if __name__ == "__main__":
    init_database()