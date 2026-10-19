import os
import glob
import logging
import pandas as pd

# Configuration des logs pour le suivi
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_latest_file(pattern):
    """Trouve le fichier le plus récent dans le dossier data/raw/ selon un motif."""
    files = glob.glob(f"data/raw/{pattern}")
    if not files:
        raise FileNotFoundError(f"Aucun fichier trouvé pour le motif: {pattern}")
    return max(files, key=os.path.getctime)

def preprocess_metrica_data():
    """Nettoie et fusionne les données de tracking Metrica."""
    logging.info("Démarrage du nettoyage des données de tracking...")
    
    try:
        # 1. Charger les fichiers les plus récents
        home_file = get_latest_file("metrica_home_*.csv")
        away_file = get_latest_file("metrica_away_*.csv")
        
        df_home = pd.read_csv(home_file)
        df_away = pd.read_csv(away_file)
        
        # 2. Renommer les colonnes pour éviter les conflits lors de la fusion
        # Ajout du préfixe 'Home_' et 'Away_' sur les colonnes des joueurs
        home_cols = {col: f"Home_{col}" for col in df_home.columns if col not in ['Period', 'Frame', 'Time [s]']}
        away_cols = {col: f"Away_{col}" for col in df_away.columns if col not in ['Period', 'Frame', 'Time [s]']}
        
        df_home = df_home.rename(columns=home_cols)
        df_away = df_away.rename(columns=away_cols)
        
        # 3. Fusionner les deux équipes sur le numéro de Frame (image par image)
        df_merged = pd.merge(df_home, df_away, on=['Period', 'Frame', 'Time [s]'], how='inner')
        
        # 4. Traitement des valeurs manquantes (ex: quand la balle sort du terrain)
        # Interpolation linéaire pour boucher les trous de trajectoire de la balle
        if 'Home_Ball_x' in df_merged.columns:
            df_merged['Home_Ball_x'] = df_merged['Home_Ball_x'].interpolate(method='linear')
            df_merged['Home_Ball_y'] = df_merged['Home_Ball_y'].interpolate(method='linear')
            
        logging.info(f"Fusion réussie : {len(df_merged)} frames au total.")
        return df_merged

    except Exception as e:
        logging.error(f"Erreur lors du traitement Metrica : {e}")
        return None

def save_processed_data(df, filename):
    """Sauvegarde le DataFrame nettoyé dans le dossier data/processed/"""
    os.makedirs("data/processed", exist_ok=True)
    filepath = f"data/processed/{filename}"
    df.to_csv(filepath, index=False)
    logging.info(f"Données prêtes pour l'IA sauvegardées dans: {filepath}")

if __name__ == "__main__":
    logging.info("--- DÉBUT DU PIPELINE DE PRÉPARATION ---")
    
    # Étape 1 : Préparation du tracking
    df_tracking = preprocess_metrica_data()
    
    # Étape 2 : Sauvegarde finale si tout s'est bien passé
    if df_tracking is not None:
        save_processed_data(df_tracking, "metrica_tracking_processed.csv")
    
    logging.info("--- FIN DU PIPELINE ---")