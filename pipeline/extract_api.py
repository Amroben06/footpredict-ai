import os
import sys
import json
import logging
import requests
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
BASE_URL = "https://api.football-data.org/v4"

if not API_KEY:
    logging.error("FOOTBALL_DATA_API_KEY not found in environment. Please set it in .env")
    sys.exit(1)

HEADERS = {"X-Auth-Token": API_KEY}


def fetch_competition_data(competition_code: str = "PL"):
    endpoints = {
        "matches": f"{BASE_URL}/competitions/{competition_code}/matches",
        "standings": f"{BASE_URL}/competitions/{competition_code}/standings",
        "teams": f"{BASE_URL}/competitions/{competition_code}/teams"
    }
    
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)

    for resource_name, url in endpoints.items():
        logging.info(f"Extracting {resource_name} for competition '{competition_code}'...")
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            
            if response.status_code == 429:
                logging.warning("Rate limit hit (football-data.org free tier allows 10 calls/min).")
                break
                
            response.raise_for_status()
            payload = response.json()
            
            file_path = os.path.join("data", "raw", f"{competition_code}_{resource_name}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                
            logging.info(f"Saved {resource_name} to {file_path}")

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch {resource_name}: {e}")


if __name__ == "__main__":
    fetch_competition_data("PL")