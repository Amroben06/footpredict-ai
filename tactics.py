import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_pitch_control(frame_index=500):
    """Calcule et dessine les zones de Voronoï pour une frame donnée du match."""
    logging.info("--- DÉMARRAGE DE L'ANALYSE TACTIQUE (VORONOÏ) ---")
    
    try:
        filepath = "data/processed/metrica_tracking_processed.csv"
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Fichier de données introuvable : {filepath}")
        
        logging.info("Chargement du dataset de tracking...")
        df = pd.read_csv(filepath)
        frame_data = df.iloc[frame_index]
        
        points = []
        colors = []
        columns = df.columns
        
        # 1. Extraire les coordonnées X et Y par paires
        for i in range(len(columns) - 1):
            col = columns[i]
            # Si on trouve un joueur (X), la colonne suivante est son (Y)
            if 'Player' in col and 'Unnamed' not in col:
                x = frame_data[col]
                y = frame_data[columns[i+1]]
                
                if pd.notna(x) and pd.notna(y):
                    points.append([x, y])
                    if 'Home' in col:
                        colors.append('#1f77b4')  # Bleu pour Domicile
                    else:
                        colors.append('#d62728')  # Rouge pour Extérieur
                        
        points = np.array(points)
        
        # Sécurité : vérifier qu'on a bien trouvé des joueurs
        if len(points) < 3:
            raise ValueError(f"Pas assez de joueurs détectés sur la frame {frame_index} pour calculer Voronoï.")
        
        # 2. Calcul mathématique de l'espace (Voronoï)
        logging.info(f"{len(points)} joueurs trouvés. Calcul des polygones...")
        vor = Voronoi(points)
        
        # 3. Rendu visuel du terrain
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_facecolor('#2E8B57')  # Vert pelouse
        
        # Lignes du terrain
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color='white', linewidth=2)
        ax.axvline(0.5, color='white', linewidth=2, linestyle='--')
        
        # Dessiner les frontières de Voronoï
        voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False, line_colors='white', line_width=1.5)
        
        # Placer les joueurs sur la carte
        for (x, y), color in zip(points, colors):
            ax.scatter(x, y, c=color, s=150, edgecolors='white', zorder=3)
            
        # 4. Chercher et placer le ballon
        for i in range(len(columns) - 1):
            if 'Ball' in columns[i] and 'Unnamed' not in columns[i]:
                ball_x = frame_data[columns[i]]
                ball_y = frame_data[columns[i+1]]
                if pd.notna(ball_x) and pd.notna(ball_y):
                    ax.scatter(ball_x, ball_y, c='yellow', s=80, edgecolors='black', zorder=4)
                break

        # Mise en forme finale
        plt.title(f"Contrôle du Terrain (Voronoï) - Frame {frame_index}", fontsize=14, color='black', pad=15)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.gca().invert_yaxis()
        
        # Masquer les graduations proprement pour conserver le fond vert
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        output_path = 'voronoi_tactics_result.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor())
        logging.info(f"Succès ! Carte tactique générée : {output_path}")
        
    except Exception as e:
        logging.error(f"Erreur de calcul tactique : {e}")

if __name__ == "__main__":
    calculate_pitch_control()