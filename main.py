import os
import logging
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from google.cloud import bigquery  # NEW: Import BigQuery

# 1 & 2: Lancement & Initialisation
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_metrica_tracking_data():
    """Extracts Metrica Sports tracking data from GitHub (Web/File Source)."""
    logging.info("Démarrage de l'extraction Metrica Sports...")
    url_home = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv"
    url_away = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"
    
    try:
        df_home = pd.read_csv(url_home, skiprows=2)
        df_away = pd.read_csv(url_away, skiprows=2)
        
        os.makedirs("data/raw", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        home_path = f"data/raw/metrica_home_{timestamp}.csv"
        away_path = f"data/raw/metrica_away_{timestamp}.csv"
        
        df_home.to_csv(home_path, index=False)
        df_away.to_csv(away_path, index=False)
        
        logging.info(f"Succès: Données Metrica sauvegardées dans {home_path} et {away_path}")
        return True

    except Exception as exc:
        logging.error(f"Échec de l'extraction Metrica: {exc}")
        return False

def extract_weather_bigdata():
    """Extracts historical UK weather data from BigQuery public datasets."""
    logging.info("Démarrage de l'extraction Big Data...")
    try:
        # Client automatically uses GOOGLE_APPLICATION_CREDENTIALS from .env
        client = bigquery.Client()
        
        query = """
            SELECT id, date, element, value
            FROM `bigquery-public-data.ghcn_d.ghcnd_2023`
            WHERE id LIKE 'UK%'
        """
        
        # --- JURY REQUIREMENT: DRY RUN ESTIMATION ---
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        dry_run_job = client.query(query, job_config=job_config)
        
        bytes_processed = dry_run_job.total_bytes_processed
        mb_processed = bytes_processed / (2048 * 2048)
        logging.info(f"Estimation BigQuery : {mb_processed:.2f} MB seront scannés.")
        
        # Refuse execution if it scans more than 1 GB (2048 MB)
        if mb_processed > 2048:
            raise ValueError(f"Requête annulée: volume trop important ({mb_processed:.2f} MB).")
        
        # --- ACTUAL EXECUTION ---
        query_job = client.query(query)
        df = query_job.to_dataframe()
        
        os.makedirs("data/raw", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"data/raw/weather_bigdata_{timestamp}.csv"
        df.to_csv(filepath, index=False)
        logging.info(f"Succès: {len(df)} lignes sauvegardées dans {filepath}.")
        return True

    except Exception as exc:
        # Fails gracefully without stopping the script
        logging.error(f"Échec de l'extraction Big Data: {exc}")
        return False

# Execution block
if __name__ == "__main__":
    extract_metrica_tracking_data()
    extract_weather_bigdata()  # NEW: Run the Big Data extraction
    logging.info("Fin du script d'extraction.")