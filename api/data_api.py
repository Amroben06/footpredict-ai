import os
import sqlite3
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

# Configuration & Security (Criterion C5)
API_KEY_NAME = "X-API-Key"
SERVER_API_KEY = os.getenv("DATA_API_KEY", "simplon_secure_token_2024")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

DB_PATH = os.path.join("database", "footpredict.db")

app = FastAPI(
    title="FootPredict Data Ingestion & Storage API",
    description="REST API providing structured football match and team metrics (Simplon RNCP37827 - C5).",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)


# Security dependency
def get_api_key(api_key: Optional[str] = Security(api_key_header)):
    if api_key == SERVER_API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Unauthorized: Invalid or missing API Key"
    )


# Schemas (Data Validation & OpenAPI specs)
class TeamResponse(BaseModel):
    id_team: int
    team_name: str
    points: int
    goal_difference: int


class MatchResponse(BaseModel):
    id_match: int
    utc_date: str
    matchday: int
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    outcome: str


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint to verify service uptime."""
    return {"status": "healthy", "database": os.path.exists(DB_PATH)}


@app.get("/api/v1/teams", response_model=List[TeamResponse], tags=["Teams"])
def list_teams(auth: str = Depends(get_api_key)):
    """Retrieve all teams with points and goal differences (Protected)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id_team, team_name, points, goal_difference FROM teams ORDER BY points DESC;")
    teams = cursor.fetchall()
    conn.close()
    return [dict(t) for t in teams]


@app.get("/api/v1/matches", response_model=List[MatchResponse], tags=["Matches"])
def list_matches(limit: int = 50, auth: str = Depends(get_api_key)):
    """Retrieve match fixtures and outcomes with joined team names (Protected)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
    SELECT 
        m.id_match,
        m.utc_date,
        m.matchday,
        ht.team_name AS home_team,
        at.team_name AS away_team,
        m.home_goals,
        m.away_goals,
        m.outcome
    FROM matches m
    JOIN teams ht ON m.home_team_id = ht.id_team
    JOIN teams at ON m.away_team_id = at.id_team
    ORDER BY m.matchday ASC
    LIMIT ?;
    """
    cursor.execute(query, (limit,))
    matches = cursor.fetchall()
    conn.close()
    return [dict(m) for m in matches]