"""
config.py
=========
Fichier de configuration centrale du projet Water Potability.

Rôle
----
Source unique de vérité pour tous les paramètres transversaux :
chemins d'accès, métriques de comparaison, seuils de sélection de modèles,
et structure des répertoires de sortie.

Tous les autres modules (data_processing, tuning, train_model) importent
depuis ce fichier — ne jamais dupliquer ces constantes ailleurs.

Usage
-----
    from config import PATHS, METRICS, MODEL_SELECTION, RANDOM_STATE
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.metrics import fbeta_score, make_scorer, recall_score
from typing import Tuple, Optional
# ===========================================================================
# REPRODUCTIBILITÉ
# ===========================================================================

RANDOM_STATE: int = 42
"""Graine aléatoire partagée par tous les modules pour la reproductibilité."""

# ===========================================================================
# DONNÉES
# ===========================================================================

DATA = {
    # Répertoire contenant le CSV brut téléchargé depuis Kaggle
    "raw_dir":      Path("data/raw"),

    # Répertoire pour les données après pipeline data_processing
    "processed_dir": Path("data/processed"),

    # Nom du CSV brut
    "raw_filename": "water_potability.csv",

    # Nom du CSV traité sauvegardé après run_pipeline()
    "processed_filename": "water_potability_processed.csv",

    # Fraction réservée au test final (hors CV)
    "test_size": 0.20,

    # Fraction réservée à la validation lors du tuning (GridSearch interne)
    "val_size": 0.15,
}

DATA["raw_path"]       = DATA["raw_dir"]       / DATA["raw_filename"]
DATA["processed_path"] = DATA["processed_dir"] / DATA["processed_filename"]

# ===========================================================================
# CHEMINS DE SORTIE
# ===========================================================================

PATHS = {
    # Racine des artefacts générés
    "outputs":        Path("outputs"),

    # Modèles sérialisés (.joblib)
    "models":         Path("outputs/models"),

    # Rapports texte / CSV / JSON
    "reports":        Path("outputs/reports"),

    # Figures matplotlib / seaborn
    "figures":        Path("outputs/figures"),

    # Sous-dossier dédié aux figures de tuning
    "figures_tuning": Path("outputs/figures/tuning"),

    # Sous-dossier dédié aux figures d'évaluation finale
    "figures_eval":   Path("outputs/figures/evaluation"),

    # Fichier JSON des meilleurs hyperparamètres (produit par tuning.py)
    "best_params":    Path("outputs/reports/best_params.json"),

    # Rapport CSV complet du grid search
    "tuning_report":  Path("outputs/reports/tuning_report.csv"),

    # Rapport CSV de l'évaluation finale (CV + test)
    "eval_report":    Path("outputs/reports/evaluation_report.csv"),

    # Rapport texte lisible synthétisant les top modèles
    "summary_txt":    Path("outputs/reports/model_summary.txt"),
}


def ensure_dirs() -> None:
    """
    Crée tous les répertoires de sortie s'ils n'existent pas encore.

    À appeler en début de script dans tuning.py et train_model.py.

    Examples
    --------
    >>> from config import ensure_dirs
    >>> ensure_dirs()   # idempotent, sans effet si les dossiers existent déjà
    """
    all_dirs = list(PATHS.values()) + [DATA["raw_dir"], DATA["processed_dir"]]
    for p in all_dirs:
        if p.suffix == "":          # c'est un répertoire (pas un fichier)
            p.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# FEATURES & CIBLE
# ===========================================================================

FEATURES: list[str] = ["ph", "Solids", "Conductivity", "Turbidity"]
TARGET:   str       = "Potability"

# Sémantique des classes après relabélisation dans run_pipeline()
# 1 = Non potable  (cas dangereux — classe majoritaire : ~61 %)
# 0 = Potable      (classe minoritaire : ~39 %)
# Les métriques sklearn ciblant la classe 1 mesurent ainsi directement
# la capacité du modèle à détecter l'eau non potable.
CLASS_LABELS: dict[int, str] = {0: "Potable", 1: "Non potable"}
POSITIVE_CLASS: int = 1   # classe d'intérêt pour recall, F1, précision"

# ===========================================================================
# MÉTRIQUES DE COMPARAISON
# ===========================================================================

# ---------------------------------------------------------------------------
# Scorers personnalisés
# ---------------------------------------------------------------------------

# Fbeta (beta=2) : donne 2× plus de poids au recall qu'à la précision.
# Utilisé dans le score composite de classement final uniquement.
FBETA_SCORER = make_scorer(fbeta_score, beta=2)


def _gmean_score(y_true, y_pred) -> float:
    """
    G-mean : moyenne géométrique des recalls de chaque classe.

        G-mean = sqrt(recall_classe_1 × recall_classe_0)

    Propriété clé : vaut 0 si le modèle prédit tout dans une seule classe,
    quelle que soit cette classe. Force donc l'équilibre entre les deux recalls
    et protège contre le biais de classe majoritaire — là où Fbeta échoue.

    Exemples :
    - Prédit tout Non potable  → sqrt(1.0 × 0.0) = 0.0  (modèle dégénéré détecté)
    - Prédit tout Potable      → sqrt(0.0 × 1.0) = 0.0  (idem)
    - recall_1=0.80, recall_0=0.75 → sqrt(0.80 × 0.75) ≈ 0.775
    """
    r1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    r0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    return float(np.sqrt(r1 * r0))


# G-mean scorer : utilisé comme métrique d'optimisation du GridSearchCV.
# Empêche le biais vers la classe majoritaire tout en restant sensible
# au déséquilibre des classes.
GMEAN_SCORER = make_scorer(_gmean_score)

METRICS = {
    # Métrique principale pour GridSearchCV : G-mean
    # Force l'équilibre des recalls des deux classes — protège contre
    # le biais de classe majoritaire que Fbeta seul ne détecte pas.
    "primary": "gmean",

    # Métriques calculées en CV et sur le test set
    "secondary": ["roc_auc", "f1", "precision", "recall", "accuracy", "mcc", "pr_auc"],

    # Scoring dict pour cross_validate (sklearn scoring strings + custom)
    "scoring": {
        "roc_auc":   "roc_auc",
        "f1":        "f1",
        "precision": "precision",
        "recall":    "recall",
        "accuracy":  "accuracy",
        "fbeta":     FBETA_SCORER,
        "gmean":     GMEAN_SCORER,
    },

    # Poids du score composite pour le classement final des modèles.
    # Somme = 1.0.
    # - gmean   : garantit l'équilibre entre les deux recalls (anti-biais)
    # - fbeta   : reward supplémentaire sur le recall classe Non potable
    # - pr_auc  : robustesse au déséquilibre
    # - mcc     : vision globale des 4 quadrants de la matrice de confusion
    # - roc_auc : discrimination globale
    "composite_weights": {
        "gmean":    0.25,   # équilibre recall_0 × recall_1 — anti-biais classe majoritaire
        "fbeta":    0.20,   # recall-oriented classe Non potable (beta=2)
        "pr_auc":   0.20,   # robustesse au déséquilibre des classes
        "mcc":      0.20,   # équilibre sur les 4 quadrants de la matrice de confusion
        "roc_auc":  0.15,   # discrimination globale
    },
}

# ===========================================================================
# RÉÉQUILIBRAGE DES CLASSES
# ===========================================================================

REBALANCING = {
    # Stratégie : SMOTETomek (SMOTE sur-échantillonne la classe minoritaire +
    # Tomek Links supprime les paires ambiguës à la frontière de décision).
    # Appliqué UNIQUEMENT sur le train set, jamais sur le test set.
    "strategy":        "smotetomek",

    # k-voisins utilisés par SMOTE pour générer des échantillons synthétiques
    "smote_k_neighbors": 5,

    # Activer/désactiver le rééquilibrage (False = désactivé pour comparaison)
    "enabled": False,
}

# ===========================================================================
# TUNING DU SEUIL DE DÉCISION
# ===========================================================================

THRESHOLD_TUNING = {
    # Activer la recherche du seuil optimal sur le val set (ou CV)
    "enabled": True,

    # Grille de seuils à tester (de 0.20 à 0.80, pas de 0.01)
    "thresholds": None,   # None = auto : np.arange(0.20, 0.80, 0.01)

    # Métrique optimisée pour choisir le seuil : "fbeta", "f1" ou "recall"
    "optimize_metric": "fbeta",

    # Valeur beta pour fbeta lors du tuning de seuil
    "beta":1,
}

# ===========================================================================
# VALIDATION CROISÉE
# ===========================================================================

CV = {
    # Nombre de folds pour la validation croisée stratifiée
    "n_splits": 10,

    # Shuffle avant split (recommandé pour les datasets non temporels)
    "shuffle": True,
}

# ===========================================================================
# GRID SEARCH
# ===========================================================================

GRID_SEARCH = {
    # Nombre de folds internes pour le GridSearchCV
    "cv_splits": 5,

    # Métrique optimisée par GridSearchCV : gmean 
    "scoring": GMEAN_SCORER,

    # Nombre de jobs parallèles (-1 = tous les cœurs disponibles)
    "n_jobs": -1,

    # Affichage de la progression (0 = silencieux, 2 = verbeux)
    "verbose": 1,

    # Retourner les scores de toutes les combinaisons testées
    "return_train_score": True,
}

# ===========================================================================
# SÉLECTION DES MODÈLES
# ===========================================================================

MODEL_SELECTION = {
    # Nombre de modèles conservés dans le « top » (sauvegardés + tracés)
    "top_n": 5,

    # Seuil minimum de score composite pour qu'un modèle soit considéré acceptable
    "min_composite": 0.55,
    #Seuil minimun de roc_auc pour selectionner les modèles acceptables
    "min_roc_auc": 0.70,
    # Seuil d'écart-type maximum acceptable en CV (stabilité)
    "max_std_gmean": 0.06,

    # Format de sérialisation des modèles
    "model_format": ".joblib",

    # Préfixe des fichiers modèle
    "model_prefix": "model_",
}

PHYSICAL_BOUNDS: dict[str, Tuple[float, float]] = {
    "ph":           (0.0,  14.0),
    "Solids":       (0.0,  10000.0),   
    "Conductivity": (0.0,  15000.0),   
    "Turbidity":    (0.0,  1000.0),   
    "Temperature":  (-10.0,100.0)  
}

TDS_EC_FACTOR=0.67

# ===========================================================================
# PARAMÈTRES D'AFFICHAGE DES FIGURES
# ===========================================================================

PLOT = {
    "dpi":     150,
    "figsize": (10, 7),
    "style":   "seaborn-v0_8-whitegrid",

    # Palette de couleurs pour les courbes ROC (une couleur par modèle)
    "palette": [
        "#2E86AB",   # bleu acier
        "#E84855",   # rouge vif
        "#3BB273",   # vert émeraude
        "#F18F01",   # orange ambré
        "#7B2D8B",   # violet
        "#FF6B6B",   # corail
        "#4ECDC4",   # turquoise
        "#95E1D3",   # menthe
    ],
}

# ===========================================================================
# REGISTRE DES MODÈLES ACTIFS
# ===========================================================================
# Dictionnaire mis à jour dynamiquement par train_model.py après l'entraînement.
# Structure attendue :
#   MODEL_REGISTRY[model_name] = {
#       "roc_auc_mean": float,
#       "roc_auc_std":  float,
#       "f1_mean":      float,
#       "params":       dict,
#       "path":         Path,
#       "rank":         int,       # 1 = meilleur
#   }
MODEL_REGISTRY: dict = {}
