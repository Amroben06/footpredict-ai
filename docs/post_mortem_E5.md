\# Cas Pratique 2 (E5) : Monitoring et Post-Mortem d'Incident



\## 1. Stratégie de Monitoring (Production)

Pour assurer la fiabilité de l'application \*FootPredict AI\* en production, les métriques suivantes sont surveillées :

\* \*\*Disponibilité (Uptime)\*\* : Ping régulier sur le point d'accès `/health` de l'API de données (cible : 99.9%).

\* \*\*Performance (Latence)\*\* : Surveillance du temps de réponse de l'API d'inférence (alerte si > 500ms).

\* \*\*Dérive du Modèle (Data Drift)\*\* : Analyse mensuelle de l'exactitude des prédictions (Accuracy) par rapport aux résultats réels de la Premier League. Si la précision chute sous les 35%, un réentraînement est déclenché automatisé via GitHub Actions.



\## 2. Post-Mortem : Résolution de l'Incident de Parsing (Scraping)

\* \*\*Date de l'incident\*\* : 25 Septembre 2026.

\* \*\*Symptôme\*\* : L'API de données renvoyait une erreur `500 Internal Server Error` lors de l'ingestion, et le dashboard affichait des données vides.

\* \*\*Cause racine (Root Cause)\*\* : Modification de la structure HTML du tableau Wikipédia. Les points de pénalité étaient accompagnés de notes de bas de page (ex: `32\[d]`), provoquant un crash lors de la conversion en nombre entier (`ValueError: invalid literal for int()`).

\* \*\*Correction apportée\*\* : Implémentation d'une fonction de nettoyage avec expression régulière (`re.sub(r"\[^\\d\\-+.]", "", str(val))`) dans le script de modélisation pour ignorer les caractères non numériques avant la conversion.

\* \*\*Prévention\*\* : Ajout d'un test unitaire dans `tests/test\_app.py` validant strictement le typage (integer) des points et de la différence de buts avant l'insertion en base.

