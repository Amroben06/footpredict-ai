\# Data Modeling (Merise Method) \& Storage Architecture



\## 1. Conceptual Data Model (CDM / MCD)



\### Entities \& Attributes

\* \*\*TEAM\*\*

&#x20; \* `id\_team` (INT, Primary Key)

&#x20; \* `team\_name` (VARCHAR, Unique)

&#x20; \* `points` (INT)

&#x20; \* `goal\_difference` (INT)



\* \*\*MATCH\*\*

&#x20; \* `id\_match` (INT, Primary Key)

&#x20; \* `utc\_date` (DATETIME)

&#x20; \* `matchday` (INT)

&#x20; \* `home\_goals` (INT)

&#x20; \* `away\_goals` (INT)

&#x20; \* `outcome` (VARCHAR)



\### Associations

\* `PLAYS\_AT\_HOME` : TEAM (1,n) <---> (1,1) MATCH

\* `PLAYS\_AWAY`    : TEAM (1,n) <---> (1,1) MATCH



\---



\## 2. Physical Data Model (PDM / MPD)



```sql

CREATE TABLE teams (

&#x20;   id\_team INTEGER PRIMARY KEY AUTOINCREMENT,

&#x20;   team\_name VARCHAR(100) UNIQUE NOT NULL,

&#x20;   points INTEGER DEFAULT 0,

&#x20;   goal\_difference INTEGER DEFAULT 0

);



CREATE TABLE matches (

&#x20;   id\_match INTEGER PRIMARY KEY,

&#x20;   utc\_date TIMESTAMP NOT NULL,

&#x20;   matchday INTEGER NOT NULL,

&#x20;   home\_team\_id INTEGER NOT NULL REFERENCES teams(id\_team),

&#x20;   away\_team\_id INTEGER NOT NULL REFERENCES teams(id\_team),

&#x20;   home\_goals INTEGER NOT NULL,

&#x20;   away\_goals INTEGER NOT NULL,

&#x20;   outcome VARCHAR(20) NOT NULL

);

\### Step 2: Add the GDPR Compliance Document (Criterion C4)



The grid requires a register of personal data processing and retention/purging rules.



Create `docs/rgpd.md`:



```cmd

notepad docs\\rgpd.md

\# Registre de Traitement \& Conformité RGPD



\## 1. Contexte et Finalité du Traitement

\* \*\*Projet\*\* : FootPredict AI - Outil prédictif d'analyse de performances sportives.

\* \*\*Finalité\*\* : Analyse statistique des rencontres et aide à la décision sportive.

\* \*\*Base Légale\*\* : Intérêt légitime (Article 6(1)(f) du RGPD). Données issues du domaine public (statistiques officielles de ligues professionnelles).



\## 2. Catégories de Données Traitées

\* Noms des clubs professionnels et statistiques de match (scores, journées, dates UTC).

\* Données personnelles sensibles exclues (aucune donnée médicale, privée ou de coordonnées).



\## 3. Durée de Conservation et Procédure de Purge

\* \*\*Conservation active\*\* : 3 saisons sportives pour l'entraînement et l'évaluation des modèles d'IA.

\* \*\*Procédure de purge\*\* : Script automatisé annuel purgeant toute métrique nominative non exploitée au-delà du cycle réglementaire.

