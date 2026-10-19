import logging
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans

# Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_jersey_color(image_crop):
    """Extrait la couleur moyenne du torse pour éviter de fausser l'analyse avec la pelouse."""
    h, w, _ = image_crop.shape
    if h == 0 or w == 0:
        return [0, 0, 0]
    
    # Cibler le centre de la moitié supérieure (le maillot)
    torso = image_crop[int(h*0.1):int(h*0.5), int(w*0.3):int(w*0.7)]
    if torso.size == 0:
        return [0, 0, 0]
        
    # Retourner la couleur moyenne [Bleu, Vert, Rouge]
    return cv2.mean(torso)[:3]

def analyze_match_frame():
    logging.info("--- DÉMARRAGE DE L'IA PRO (YOLOv8s + K-MEANS) ---")
    
    try:
        # 1. Modèle 'Small' (plus précis pour le ballon)
        model = YOLO('yolov8s.pt')
        image_path = 'broadcast_frame.png'
        image = cv2.imread(image_path)
        
        # Inférence
        results = model(image_path, conf=0.3)[0]
        
        players, player_colors, balls = [], [], []
        
        # 2. Extraction des joueurs et du ballon
        for box in results.boxes:
            cls = int(box.cls[0]) # Classe (0=Personne, 32=Ballon de sport)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if cls == 0: # Joueur
                crop = image[y1:y2, x1:x2]
                color = get_jersey_color(crop)
                players.append((x1, y1, x2, y2))
                player_colors.append(color)
            elif cls == 32: # Ballon
                balls.append((x1, y1, x2, y2))
                
        # 3. K-Means Clustering : Séparation en 2 équipes
        logging.info(f"{len(players)} joueurs trouvés. Analyse des maillots en cours...")
        if len(player_colors) >= 2:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            teams = kmeans.fit_predict(player_colors)
        else:
            teams = [0] * len(players)
            
        # 4. Rendu visuel personnalisé
        colors = [(255, 50, 50), (50, 255, 50)] # Bleu pour Équipe 0, Vert pour Équipe 1
        
        for (x1, y1, x2, y2), team_id in zip(players, teams):
            color = colors[team_id]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"Team {team_id+1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        for (x1, y1, x2, y2) in balls:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3) # Ballon en Jaune
            cv2.putText(image, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 3)
            
        output_path = 'vision_pro_result.png'
        cv2.imwrite(output_path, image)
        logging.info(f"Succès ! Résultat sauvegardé dans : {output_path}")

    except Exception as e:
        logging.error(f"Erreur : {e}")

if __name__ == "__main__":
    analyze_match_frame()