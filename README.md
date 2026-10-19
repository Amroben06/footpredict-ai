# FootPredict AI

## 1. Présentation du projet
* **Objectifs fonctionnels et techniques :** Automatiser l'extraction et le traitement de données footballistiques (tracking, météo) pour alimenter un modèle de machine learning prédictif.
* **Acteurs :** Projet individuel réalisé dans le cadre du Bachelor 3 Développeur Data & IA.
* **Environnements :** Python 3.11, Google Cloud Platform (BigQuery).
* **Contraintes :** Gestion stricte des erreurs sans interruption du pipeline, volume Big Data > 1 Go.

## 2. Spécifications techniques & Périmètre
* **Langage :** Python
* **Données visées :** Tracking (fichiers Metrica Sports) et Météo (BigQuery). Le périmètre couvre l'extraction depuis les sources jusqu'au jeu de données brutes (`data/raw/`).
* **Bibliothèques :** `pandas`, `google-cloud-bigquery`, `python-dotenv`.

## 3. Installation et Configuration (Test Binôme)
1. Cloner le dépôt GitHub privé.
2. Installer les dépendances :
   ```bash
   python -m pip install -r requirements.txt