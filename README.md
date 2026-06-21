# 💧 PLBD — Système de Détection de la Potabilité de l'Eau

> Système embarqué de traitement et de prédiction de la qualité de l'eau,  
> combinant capteurs bas coût, vision par ordinateur et modèles ML explicables.

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Contexte et contraintes](#2-contexte-et-contraintes)
3. [Architecture du système](#3-architecture-du-système)
4. [Structure du projet](#4-structure-du-projet)
5. [Installation](#5-installation)
6. [Démarrage rapide](#6-démarrage-rapide)
7. [Pipeline ML](#7-pipeline-ml)
8. [Choix techniques](#8-choix-techniques)
9. [Résultats et performances](#9-résultats-et-performances)
10. [Limites connues](#10-limites-connues)
11. [Feuille de route](#11-feuille-de-route)
12. [Contribuer](#12-contribuer)

---

## 1. Vue d'ensemble

PLBD est un système intelligent de surveillance et de traitement de la qualité de l'eau conçu pour des **contextes à ressources limitées** — zones rurales, points d'eau sans infrastructure de laboratoire, ou sites nécessitant une surveillance continue à faible coût.

Le système remplit trois fonctions complémentaires :

| Fonction | Description | Statut |
|---|---|---|
| **Détection temps réel** | Classifie l'eau comme potable ou non potable à partir de 4 capteurs | ✅ Opérationnel |
| **Prédiction de turbidité** | Estime la turbidité (NTU) à partir d'une image caméra | 🔄 En développement |
| **Filtration automatisée** | Sélectionne et active les filtres adaptés à la qualité détectée | 🔄 En développement |
| **Prévision temporelle** | Anticipe la qualité de l'eau dans les prochaines heures | 📋 Planifié |
| **Interface web** | Visualisation des mesures, alertes et prédictions en temps réel | 📋 Planifié |

---

## 2. Contexte et contraintes

### Problème métier

La vérification de la potabilité de l'eau requiert normalement une analyse physico-chimique complète en laboratoire — coûteuse, lente, et inaccessible dans de nombreux contextes. Ce projet vise à construire un **outil de triage à faible coût** permettant d'identifier les eaux suspectes avant toute analyse approfondie.

### Contrainte budgétaire — sélection des features

Le dataset contient **9 variables physico-chimiques**. Seules **4** sont utilisées pour la modélisation, contraintes par le coût des capteurs de mesure :

| Variable | Capteur | Coût | Statut |
|---|---|---|---|
| `ph` | Électrode pH | 💲 Faible | ✅ Retenu |
| `Solids` (TDS) | Conductimètre TDS | 💲 Faible | ✅ Retenu |
| `Conductivity` | Conductimètre EC | 💲 Faible | ✅ Retenu |
| `Turbidity` | Turbidimètre / Caméra | 💲 Faible | ✅ Retenu |
| `Hardness` | Titrimètre + réactifs | 💲💲 Moyen | ❌ Écarté |
| `Chloramines` | Spectrophotomètre | 💲💲💲 Élevé | ❌ Écarté |
| `Sulfate` | Chromatographe ionique | 💲💲💲 Élevé | ❌ Écarté |
| `Organic_carbon` | Analyseur TOC | 💲💲💲💲 Très élevé | ❌ Écarté |
| `Trihalomethanes` | GC-MS | 💲💲💲💲 Très élevé | ❌ Écarté |

> ⚠️ **Limite fondamentale** : Les variables les plus prédictives (Sulfate, Organic Carbon, Trihalomethanes) sont précisément celles hors budget. La corrélation entre les 4 features retenues et la cible est faible (~0.08–0.12). Ce système est un **outil de triage**, non un substitut à une analyse de laboratoire complète.

### Convention de labellisation

```
Potability = 1  →  Non potable  (classe dangereuse — majoritaire ~61%)
Potability = 0  →  Potable      (classe minoritaire ~39%)
```

Cette convention place la classe dangereuse en tant que **classe positive**, ce qui oriente naturellement les métriques sklearn (recall, F1, précision) vers la détection de l'eau non potable.

---

## 3. Architecture du système

```
┌─────────────────────────────────────────────────────────────────┐
│                       COUCHE ACQUISITION                         │
│                                                                   │
│   Capteurs physiques              Caméra embarquée               │
│   (ph, TDS, Conductivity)         (image eau)                    │
└──────────────┬────────────────────────────┬─────────────────────┘
               │                            │
               ▼                            ▼
┌──────────────────────┐        ┌───────────────────────┐
│  Séries temporelles  │        │  Modèle Vision (CNN)  │
│  des mesures         │        │  Image → Turbidité    │
└──────────┬───────────┘        └──────────┬────────────┘
           │                               │
           └─────────────┬─────────────────┘
                         │
           ┌─────────────▼──────────────────┐
           │          COUCHE ML             │
           │                                │
           │  ┌──────────────────────────┐  │
           │  │  Classifieur (actuel)    │  │
           │  │  Potabilité temps réel   │  │
           │  │  ThresholdClassifier     │  │
           │  └──────────────────────────┘  │
           │                                │
           │  ┌──────────────────────────┐  │
           │  │  Modèle temporel         │  │
           │  │  Prévision t+n           │  │
           │  └──────────────────────────┘  │
           └─────────────┬──────────────────┘
                         │
           ┌─────────────▼──────────────────┐
           │       COUCHE DÉCISION          │
           │   Système multi-filtres        │
           │   (règles + ML)                │
           └─────────────┬──────────────────┘
                         │
           ┌─────────────▼──────────────────┐
           │       INTERFACE WEB            │
           │  Mesures temps réel            │
           │  Alertes + Prévisions          │
           │  Explications SHAP             │
           └────────────────────────────────┘
```

---

## 4. Structure du projet

```
PLBD/
│
├── 📁 src/                          # Code source
│   ├── config.py                    # Configuration centrale (source de vérité)
│   ├── data/
│   │   ├── data_processing.py       # Pipeline prétraitement
│   │   └── feature_engineering.py  # GMM features, transforms
│   ├── models/
│   │   ├── train_model.py           # Entraînement + ThresholdClassifier
│   │   ├── tuning.py                # Grid Search + threshold tuning
│   │   ├── forecasting.py           # Prévision temporelle (à venir)
│   │   └── vision.py                # Turbidité par image (à venir)
│   ├── evaluation/
│   │   ├── metrics.py               # Métriques, score composite, rapport
│   │   ├── plots.py                 # ROC, confusion matrix, CV comparison
│   │   └── explainability.py        # SHAP — global + individuel
│   └── api/
│       ├── main.py                  # FastAPI endpoints
│       ├── schemas.py               # Pydantic models
│       └── inference.py             # Chargement modèle + predict
│
├── 📁 app/                          # Interface web
│
├── 📁 data/
│   ├── raw/                         # ⚠️ Jamais modifié — source de vérité
│   │   └── water_potability.csv
│   ├── processed/                   # Sorties data_processing
│   └── external/                    # Données météo, géo, temporelles
│
├── 📁 models/                       # Modèles sérialisés (.joblib)
│   └── runs/                        # Versionné par timestamp
│       └── YYYY-MM-DD_HH-MM/
│           ├── model_1_rf.joblib    # ThresholdClassifier (seuil intégré)
│           ├── model_2_xgb.joblib
│           └── scaler.joblib
│
├── 📁 outputs/
│   ├── figures/
│   │   ├── tuning/                  # Heatmaps Grid Search
│   │   └── evaluation/              # ROC, confusion matrix, SHAP plots
│   └── reports/
│       ├── best_params.json         # Meilleurs hyperparamètres + seuils
│       ├── tuning_report.csv        # Toutes les combinaisons testées
│       ├── evaluation_report.csv    # Métriques CV + test
│       └── model_summary.txt        # Rapport lisible synthèse
│
├── 📁 notebooks/
│   └── water_potability_eda.ipynb   # EDA complète + analyses clustering
│
├── 📁 tests/
│   ├── test.py                      # 75 tests unitaires
│   └── conftest.py                  # Fixtures partagées pytest
│
├── 📁 .github/
│   └── workflows/
│       └── ci.yml                   # Tests automatiques CI/CD
│
├── .env.example                     # Template credentials (Kaggle, etc.)
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── Makefile
└── README.md
```

---

## 5. Installation

### Prérequis

- Python 3.10+
- pip

### Installation des dépendances

```bash
# Cloner le projet
git clone https://github.com/votre-org/PLBD.git
cd PLBD

# Environnement virtuel (recommandé)
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# Dépendances core
pip install -r requirements.txt

# Dépendances dev (tests, linting)
pip install -r requirements-dev.txt
```

### Configuration des credentials Kaggle

```bash
# 1. Obtenir la clé API sur https://www.kaggle.com/settings → Create New Token
# 2. Copier le template
cp .env.example .env

# 3. Renseigner vos credentials dans .env
KAGGLE_USERNAME=votre_username
KAGGLE_KEY=votre_api_key
```

### Configuration pour un nouveau membre de l'équipe

```bash
python -c "from src.data.data_processing import setup_kaggle_credentials; setup_kaggle_credentials()"
```

---

## 6. Démarrage rapide

```bash
# Téléchargement automatique du dataset + pipeline complet
python src/models/tuning.py           # ~20-40 min (Grid Search)
python src/models/train_model.py      # ~5-10 min (entraînement + évaluation)

# Sans SHAP (plus rapide)
python src/models/train_model.py --no-shap

# Avec hyperparamètres par défaut (sans tuning préalable)
python src/models/train_model.py --no-tuning

# Lancer les tests
python -m pytest tests/ -v

# Explorer les données
jupyter notebook notebooks/water_potability_eda.ipynb
```

---

## 7. Pipeline ML

### Vue d'ensemble

```
CSV brut
   │
   ▼
raw_data_processing()          # Nettoyage, outliers, imputation, relabélisation
   │
   ▼
preprocess_for_ml()            # Split stratifié + RobustScaler
   │
   ▼
SMOTETomek                     # Rééquilibrage train set uniquement
   │
   ▼
tuning.py                      # Grid Search (G-mean scorer)
   │  └── threshold tuning     # Seuil optimal par modèle (Fbeta β=2)
   │
   ▼
train_model.py
   │  ├── cross_validate()     # StratifiedKFold (10 folds)
   │  ├── ThresholdClassifier  # Seuil intégré au modèle
   │  └── save (.joblib)
   │
   ▼
evaluation/
   ├── metrics.py              # Score composite (5 métriques pondérées)
   ├── plots.py                # ROC, PR, confusion matrix, feature importance
   └── explainability.py       # SHAP global + individuel + interactions
```

### Gestion du déséquilibre des classes

Le dataset est déséquilibré (~61% Non potable / ~39% Potable). Trois stratégies complémentaires sont appliquées :

| Stratégie | Où | Effet |
|---|---|---|
| `SMOTETomek` | Train set uniquement | Sur-échantillonnage synthétique + nettoyage frontière |
| `class_weight="balanced"` | Tous les estimateurs | Pénalise davantage les erreurs sur la classe minoritaire |
| `G-mean scorer` | GridSearchCV | Empêche le biais vers la classe majoritaire |

### ThresholdClassifier — seuil de décision intégré

```python
# Le problème sans wrapper
model = joblib.load("model.joblib")
model.predict(X)   # ← utilise 0.5 par défaut, pas le seuil optimal ❌

# La solution
model = joblib.load("model.joblib")  # ThresholdClassifier
model.predict(X)        # ← seuil optimal appliqué automatiquement ✓
model.threshold         # ← 0.32 (exemple) ✓
model.predict_proba(X)  # ← probabilités brutes inchangées ✓
```

### Score composite de classement

Les modèles sont classés selon un score composite pondéré :

```
Score = 0.25 × G-mean   +  0.20 × Fbeta(β=2)
      + 0.20 × PR-AUC   +  0.20 × MCC
      + 0.15 × ROC-AUC
```

| Métrique | Poids | Justification |
|---|---|---|
| G-mean | 0.25 | Équilibre recall₀ × recall₁ — anti-biais classe majoritaire |
| Fbeta(β=2) | 0.20 | Priorité recall Non potable (santé publique) |
| PR-AUC | 0.20 | Robustesse au déséquilibre |
| MCC | 0.20 | Vision globale des 4 quadrants de la matrice de confusion |
| ROC-AUC | 0.15 | Discrimination générale |

---

## 8. Choix techniques

### Standardisation — RobustScaler

Le `RobustScaler` est privilégié sur `StandardScaler` et `MinMaxScaler` car il centre sur la **médiane** et scale sur l'**IQR** — insensible aux outliers résiduels, critiques dans les distributions de Conductivity et Solids.

### Scorer GridSearch — G-mean

Le G-mean (`√(recall₁ × recall₀)`) est utilisé pour le Grid Search car il vaut **0** dès qu'un modèle prédit tout dans une classe, ce que Fbeta seul ne garantit pas :

```
Modèle "prédit tout Non potable" :
  Fbeta(β=2) ≈ 0.89  ← score excellent pour un modèle inutile ❌
  G-mean     = 0.00  ← détecté et pénalisé ✓
```

### Modèles évalués

| Modèle | Particularité |
|---|---|
| Logistic Regression | Baseline linéaire |
| SVM (RBF) | Efficace en faible dimension |
| Random Forest | Robuste, feature importance fiable |
| Extra Trees | Plus aléatoire que RF |
| Gradient Boosting | Boosting séquentiel classique |
| XGBoost | `scale_pos_weight` pour déséquilibre |
| LightGBM | Ultra-rapide sur histogrammes |
| CatBoost | `auto_class_weights="Balanced"` |

### Explainabilité SHAP

SHAP est intégré dans `evaluation/explainability.py` pour trois niveaux d'analyse :

| Analyse | Figure | Usage |
|---|---|---|
| **Global** | Summary plot + Beeswarm | Importance et direction des features |
| **Interaction** | Dependence plot | Effets croisés entre variables |
| **Local** | Waterfall + Force plot | Explication d'une prédiction individuelle |

L'explication locale est particulièrement importante pour l'interface web : l'opérateur comprend **pourquoi** l'eau est classée non potable et peut choisir le filtre approprié.

---

## 9. Résultats et performances

> Les résultats ci-dessous sont indicatifs et varient selon le run.  
> Les rapports complets sont générés dans `outputs/reports/` à chaque run.

### Métriques test (top modèles)

| Modèle | G-mean | ROC-AUC | PR-AUC | MCC | Fbeta(β=2) | Seuil |
|---|---|---|---|---|---|---|
| Random Forest | ~0.47 | ~0.64 | ~0.75 | ~0.23 | ~0.85 | 0.32 |
| XGBoost | ~0.46 | ~0.65 | ~0.74 | ~0.21 | ~0.83 | 0.24 |
| LightGBM | ~0.45 | ~0.64 | ~0.73 | ~0.18 | ~0.77 | 0.25 |

### Interprétation

Ces performances reflètent la **contrainte budgétaire** documentée en section 2. Les 4 features retenues ont une corrélation individuelle faible avec la cible (~0.08–0.12). Le ROC-AUC de ~0.64 est proche du plafond atteignable avec ce sous-ensemble de variables.

Le modèle complet utilisant les 9 features atteint un ROC-AUC de ~0.75, confirmant un gap de ~0.11 dû à la contrainte budgétaire.

---

## 10. Limites connues

| Limite | Impact | Mitigation |
|---|---|---|
| Faible corrélation features → cible | ROC-AUC plafonné ~0.65 | Documenté et assumé |
| Dataset statique (pas temporel) | Pas de saisonnalité | Prévu en v2 |
| Dataset non géolocalisé | Modèle non contextualisé | Données externes prévues |
| Turbidité par caméra non validée | Vision en développement | Dataset en constitution |
| `contamination` IF ≤ 0.5 | Non potable majoritaire → paradigme anomalie inadapté | IsolationForest comme signal complémentaire uniquement |

---

## 11. Feuille de route

### v1.0 — Actuel ✅
- Pipeline ML complet (data_processing → tuning → train → evaluation)
- ThresholdClassifier avec seuil intégré
- 8 modèles optimisés par Grid Search + G-mean scorer
- SMOTETomek + class_weight pour le déséquilibre
- Explainabilité SHAP (global + local)
- 75 tests unitaires
- Notebook EDA complet avec analyses de clustering

### v1.5 — En cours 🔄
- Prédiction de turbidité par image caméra (CNN léger — MobileNetV2)
- Système multi-filtres (règles de décision basées sur la classification)
- API FastAPI pour l'inférence temps réel

### v2.0 — Planifié 📋
- Prévision temporelle de la qualité de l'eau (LSTM ou Temporal Fusion Transformer)
- Interface web temps réel (React + WebSocket)
- Alertes automatiques
- Explications SHAP en temps réel sur l'interface
- Versionnement des runs avec horodatage

### v3.0 — Vision long terme 🔭
- Collecte de données géolocalisées
- Adaptation du modèle par bassin versant
- Boucle de rétroaction (mesure après filtration → réentraînement)
- Déploiement sur Raspberry Pi / ESP32

---

## 12. Contribuer

### Workflow

```bash
# 1. Créer une branche
git checkout -b feature/nom-de-la-feature

# 2. Développer + tester
python -m pytest tests/ -v

# 3. Vérifier la qualité du code
flake8 src/ --max-line-length=100
black src/ --check

# 4. Pull Request → review obligatoire
```

### Conventions

- **Langue** : code en anglais, commentaires/docstrings en français
- **Docstrings** : format NumPy (Parameters / Returns / Examples)
- **Tests** : toute nouvelle fonction doit avoir au moins 3 tests unitaires
- **config.py** : tout nouveau paramètre global passe par `config.py`
- **Data leakage** : tout fit (scaler, imputer, resampler) se fait uniquement sur le train set

### Variables d'environnement

```bash
# .env.example — copier en .env (non versionné)
KAGGLE_USERNAME=
KAGGLE_KEY=
```

---

## Dépendances principales

```
pandas>=2.0        numpy>=1.24        scikit-learn>=1.3
xgboost>=2.0       lightgbm>=4.0      catboost>=1.2
imbalanced-learn>=0.11    shap>=0.44
matplotlib>=3.7    seaborn>=0.12      scipy>=1.10
joblib>=1.3        fastapi>=0.100     uvicorn>=0.23
python-dotenv>=1.0 kaggle>=1.5
```

---

<div align="center">

**PLBD** — Système de Détection de la Potabilité de l'Eau  
Licence MIT

</div>
