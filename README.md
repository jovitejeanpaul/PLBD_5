# AQUA-SENS — Systeme Intelligent de Surveillance de la Qualite de l'Eau

> Systeme embarque multi-filtre avec IA sur Raspberry Pi 4 pour le diagnostic, la prevision et le traitement automatise de la qualite de l'eau.

**Ecole Centrale Casablanca — Projet Learning By Doing (PLBD), 1ere annee cycle ingenieur**

---

## Vue d'ensemble

AQUA-SENS est un systeme concu pour les contextes a ressources limitees (zones rurales, points d'eau sans laboratoire). Il combine des capteurs bas cout, de l'intelligence artificielle embarquee et une filtration automatisee pour surveiller et traiter l'eau en continu.

| Module | Description | Statut |
|---|---|---|
| **Diagnostic temps reel** | Classification potable/non potable a partir de 4 capteurs | Operationnel |
| **Prevision 24h (LSTM)** | Anticipe l'evolution des parametres sur 24 heures | Operationnel |
| **Filtration automatisee** | Active le filtre adapte selon le diagnostic | Operationnel |
| **Moteur d'alertes** | Alertes preventives basees sur la norme NM 03.7.001 | Operationnel |
| **Interface web** | Dashboard temps reel avec historiques et gestion | Operationnel |
| **Explicabilite SHAP** | Importance de chaque capteur dans la decision du modele | Operationnel |

---

## Architecture

```
                  Bassine (source d'eau)
                         |
            +-----------+-----------+
            |           |           |
        pH Meter    TDS Meter    Turbidite
        (analog)    Gravity V1   TSW-20M
            |        (EC+TDS)       |
            +-----+-----+-----+----+
                  |           |
              ADS1115       DS18B20
              (I2C)        (1-Wire)
                  |           |
            +-----+-----------+-----+
            |   Raspberry Pi 4      |
            |                       |
            |  sensor_inference.py  |  --> Diagnostic (CatBoost)
            |  filter_controller.py |  --> GPIO (3 pompes)
            |  app/main.py (FastAPI)|  --> Interface web
            |  prediction/ (LSTM)   |  --> Prevision 24h
            +-----------+-----------+
                        |
               Interface Web (port 8000)
               Diagnostic | Prevision | Historiques
```

### Capteurs

| Capteur | Parametre | Interface | Pin |
|---|---|---|---|
| pH Meter V1.1 | pH [0-14] | ADS1115 A2 | I2C |
| Gravity TDS Meter V1.0 | Conductivite [uS/cm] + TDS [mg/L] | ADS1115 A1 | I2C |
| TSW-20M | Turbidite [NTU] | ADS1115 A3 | I2C |
| DS18B20 | Temperature [C] | 1-Wire | GPIO 4 |

### Filtres

| Pompe | Filtre | Usage | GPIO |
|---|---|---|---|
| 1 | Sediments | Particules en suspension, sable | PIN 23 |
| 2 | Charbon compresse | Micro-particules fines, metaux lourds | PIN 24 |
| 3 | Charbon actif | Chlore, pesticides, composes organiques | PIN 25 |

---

## Structure du projet

```
PLBD_5/
  src/
    config.py                # Configuration centrale (seuils, features, chemins)
    data_processing.py       # Pipeline de traitement des donnees Kaggle
    tuning.py                # Grid Search hyperparametres (8 modeles)
    train_model.py           # Entrainement, CV, evaluation, sauvegarde
    threshold_classifier.py  # Wrapper modele + seuil de decision
    sensor_inference.py      # Lecture capteurs + diagnostic temps reel
    filter_controller.py     # Controle des pompes GPIO
    explainability.py        # SHAP global (PC) + resume JSON (Pi)
  prediction/
    model.py                 # Architecture LSTM (PyTorch)
    1_generate_data.py       # Generation dataset synthetique 120 jours
    2_preprocess.py          # Normalisation + sequences supervisees
    3_train_model.py         # Entrainement LSTM avec early stopping
    4_evaluate.py            # Metriques + integration diagnostic
    alerte_engine.py         # Alertes preventives NM 03.7.001
    export_onnx.py           # Export LSTM en ONNX (deploiement Pi)
  app/
    main.py                  # Backend FastAPI (WebSocket + REST)
    auth.py                  # Authentification JWT (login/register/roles)
    database.py              # SQLite (utilisateurs, historiques, config)
    security.py              # Hashing bcrypt, JWT encode/decode
    static/
      index.html             # Dashboard SPA (diagnostic, prevision, historiques)
      login.html             # Page de connexion/inscription
  tests/                     # 203 tests unitaires
  outputs/
    models/                  # Modeles .joblib + scaler
    reports/                 # Rapports CSV/JSON/TXT
    figures/                 # Figures matplotlib
```

---

## Installation

### PC (entrainement)

```bash
git clone https://github.com/jovitejeanpaul/PLBD_5.git
cd PLBD_5
pip install -r requirements.txt
```

### Raspberry Pi (deploiement)

```bash
git clone https://github.com/jovitejeanpaul/PLBD_5.git
cd PLBD_5
pip install -r requirements-pi.txt
pip install "bcrypt==4.0.1"   # compatibilite passlib

# Creer le fichier .env
cat > .env <<'EOF'
SECRET_KEY=votre-cle-secrete-ici
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
EOF
```

---

## Demarrage rapide

### 1. Pipeline ML (sur PC)

```bash
cd src/

# Tuning des hyperparametres (8 modeles, Grid Search)
python tuning.py

# Entrainement + evaluation + sauvegarde des top modeles
python train_model.py

# Explicabilite SHAP
python explainability.py
```

### 2. Pipeline LSTM (sur PC)

```bash
cd prediction/

# Generer le dataset synthetique
python 1_generate_data.py

# Preprocesser (normalisation + sequences)
python 2_preprocess.py

# Entrainer le LSTM
python 3_train_model.py

# Evaluer
python 4_evaluate.py
```

### 3. Lancer l'interface web

```bash
# Sur PC (mode mock, capteurs simules)
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Sur Raspberry Pi (capteurs reels)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Identifiants par defaut :
- **Admin** : `admin` / `Admin@2026`
- **Operateur** : `operateur` / `Oper@2026`

---

## Pipeline ML — Diagnostic

### Donnees

- **Source** : Kaggle Water Potability (3276 echantillons)
- **Features** : pH, Solids (TDS), Conductivity, Turbidity
- **Cible** : Potability (0=Potable, 1=Non potable)
- **Nettoyage** : ecretage IQR, imputation pH par mediane conditionnelle, bornes physiques strictes (TRAINING_BOUNDS)

### Modeles evalues

| Modele | Composite | ROC-AUC | G-mean | F1 |
|---|---|---|---|---|
| CatBoost | 0.5717 | 0.6301 | — | 0.7812 |
| Random Forest | 0.5634 | 0.6337 | — | 0.7759 |
| LightGBM | 0.5601 | 0.6244 | — | 0.7712 |
| Gradient Boosting | 0.5547 | 0.6357 | — | 0.7471 |
| XGBoost | 0.5332 | 0.5937 | — | 0.7548 |

Le modele actif est selectionnable depuis l'interface (admin).

### Metriques

- **Optimisation Grid Search** : G-mean (equilibre recall des deux classes)
- **Score composite** : gmean (25%) + fbeta (20%) + pr_auc (20%) + mcc (20%) + roc_auc (15%)
- **Seuil de decision** : 0.5 en production (configurable)

---

## Pipeline LSTM — Prevision

- **Architecture** : LSTM bi-couche (hidden=64, dropout=0.2)
- **Entree** : 24 mesures horaires x 5 features (pH, TDS, Conductivite, Turbidite, Temperature)
- **Sortie** : 24 heures de prevision x 5 features
- **Buffer** : mediane des 5 dernieres minutes stockee toutes les heures
- **Alertes** : norme NM 03.7.001, regles combinees (contamination microbiologique, intrusion agricole, acidification, proliferation bacterienne)

---

## Filtration automatisee

Un seul filtre active par cycle, selon la priorite :

| Priorite | Condition | Filtre |
|---|---|---|
| P1 | Turbidite > 5 NTU | Sediments |
| P2 | pH hors [6.5, 8.5] ou contexte chimique | Charbon actif |
| P3 | Turbidite 2-5 NTU | Charbon compresse |
| P4 | Non potable sans regle P1-P3 | Charbon actif (defaut) + explication SHAP |
| — | Conductivite > 2700 uS/cm | Recommandation operateur (pas de pompe) |

Activation manuelle par l'admin (bouton Demarrer/Stopper, duree configurable, defaut 5 min).

---

## Interface web

### Pages

- **Accueil** : fonctionnalites, performances des modeles, etat du systeme, configuration admin, gestion utilisateurs
- **Diagnostic temps reel** : 5 cartes capteurs, badge potabilite, confiance, SHAP, recommandation filtration avec demarrer/stopper
- **Prevision 24h** : graphiques par parametre, alertes preventives (CRITICAL/WARNING/INFO)
- **Historiques** : tableaux + graphiques d'evolution, export CSV

### Roles

| Fonctionnalite | Admin | Operateur |
|---|---|---|
| Diagnostic temps reel | Oui | Oui |
| Prevision et alertes | Oui | Oui |
| Historiques et export CSV | Oui | Oui |
| Activer/stopper filtration | Oui | Non |
| Changer de modele | Oui | Non |
| Configuration systeme | Oui | Non |
| Gerer les utilisateurs | Oui | Non |

### Securite

- JWT en cookie httpOnly (30 min d'expiration)
- Hashing bcrypt des mots de passe
- WebSocket protege par cookie JWT
- Inscription libre (role operateur) + creation admin

---

## Technologies

| Composant | Technologie |
|---|---|
| Micro-controleur | Raspberry Pi 4 (ARM64, Bullseye, Python 3.9) |
| ML | scikit-learn, CatBoost, XGBoost, LightGBM |
| Deep Learning | PyTorch (LSTM), ONNX Runtime (fallback) |
| Explicabilite | SHAP |
| Backend | FastAPI, uvicorn, WebSocket |
| Frontend | HTML/CSS/JS, Chart.js |
| Base de donnees | SQLite |
| Auth | python-jose (JWT), passlib/bcrypt |
| GPIO | RPi.GPIO |
| Capteurs | Adafruit ADS1115 (I2C), DS18B20 (1-Wire) |

---

## Licence

MIT
