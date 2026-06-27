"""
tuning.py
=========
Optimisation des hyperparamètres par Grid Search pour la prédiction
de la potabilité de l'eau.

Modèles optimisés
-----------------
- Logistic Regression        : baseline linéaire rapide
- Support Vector Machine     : performant sur espaces de faible dimension
- Random Forest              : robuste, peu sensible aux hyperparamètres
- Extra Trees                : plus aléatoire que RF, souvent plus rapide
- Gradient Boosting (sklearn): boosting classique, interprétable
- XGBoost                    : boosting optimisé, gestion native des NaN
- LightGBM                   : boosting ultra-rapide sur grands datasets
- CatBoost                   : boosting robuste aux features catégorielles

Stratégie
---------
- Grilles ciblées sur les hyperparamètres **les plus déterminants** par modèle
  (rapport performance/temps de calcul optimisé).
- GridSearchCV avec StratifiedKFold(n_splits=5) pour préserver la distribution
  des classes à chaque fold (dataset légèrement déséquilibré).
- Métrique d'optimisation : ROC-AUC (plus robuste que l'accuracy sur dataset
  déséquilibré).
- Parallélisation sur tous les cœurs disponibles (n_jobs=-1).

Sorties
-------
- ``outputs/reports/tuning_report.csv``     : scores de toutes les combinaisons
- ``outputs/reports/best_params.json``      : meilleurs paramètres par modèle
- ``outputs/figures/tuning/``               : graphiques de comparaison

Usage
-----
    python tuning.py
    python tuning.py --data data/raw/water_potability.csv
    python tuning.py --models rf xgb lgbm   # sous-ensemble de modèles

Dépendances
-----------
    pip install scikit-learn xgboost lightgbm catboost joblib matplotlib seaborn
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # backend non-interactif : evite TclError sur environnements sans GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

# Import configuration centrale
from config import (
    DATA, FBETA_SCORER, FEATURES, GMEAN_SCORER, GRID_SEARCH,
    METRICS, PATHS, PLOT, RANDOM_STATE, REBALANCING, TARGET,
    THRESHOLD_TUNING, ensure_dirs,
)

# Import pipeline de données
from data_processing import preprocess_for_ml, raw_data_processing

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ===========================================================================
# 1. REGISTRE DES MODÈLES ET GRILLES D'HYPERPARAMÈTRES
# ===========================================================================

def _build_model_registry() -> dict[str, dict[str, Any]]:
    """
    Construit le registre des modèles avec leurs estimateurs et grilles.

    Les grilles sont volontairement ciblées sur les hyperparamètres les plus
    impactants pour chaque famille de modèles, avec un nombre de combinaisons
    raisonnable (< 50 par modèle) pour garder des temps de calcul maîtrisés.

    Returns
    -------
    dict
        Clé = identifiant court du modèle, valeur = dict avec
        ``estimator`` et ``param_grid``.
    """
    # Import optionnels (boosting libraries)
    try:
        from xgboost import XGBClassifier
        xgb_available = True
    except ImportError:
        xgb_available = False
        logger.warning("XGBoost non installé — modèle ignoré. pip install xgboost")

    try:
        from lightgbm import LGBMClassifier
        lgbm_available = True
    except ImportError:
        lgbm_available = False
        logger.warning("LightGBM non installé — modèle ignoré. pip install lightgbm")

    try:
        from catboost import CatBoostClassifier
        catboost_available = True
    except ImportError:
        catboost_available = False
        logger.warning("CatBoost non installé — modèle ignoré. pip install catboost")

    registry = {
        # ------------------------------------------------------------------
        # Logistic Regression — baseline linéaire
        # Hyperparamètres clés : régularisation C, type de pénalité
        # ------------------------------------------------------------------
        "logreg": {
            "label": "Logistic Regression",
            "estimator": LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=2000,
                solver="saga",
                class_weight="balanced",  # penalise les erreurs sur classe minoritaire
            ),
            "param_grid": {
                "C":       [0.01, 0.1, 1.0, 10.0, 100.0],
                "penalty": ["l1", "l2"],
            },
            # 5 × 2 = 10 combinaisons
        },

        # ------------------------------------------------------------------
        # SVM — performant dans les espaces de faible dimension
        # Hyperparamètres clés : kernel, C (marge), gamma (RBF)
        # ------------------------------------------------------------------
        "svm": {
            "label": "Support Vector Machine",
            "estimator": SVC(
                probability=True,
                random_state=RANDOM_STATE,
                cache_size=500,
                class_weight="balanced",
            ),
            "param_grid": [
                # Kernel RBF (généralement le plus performant)
                {"kernel": ["rbf"],    "C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.01, 0.1]},
                # Kernel linéaire (bon si les classes sont linéairement séparables)
                {"kernel": ["linear"], "C": [0.1, 1, 10]},
            ],
            # 4×4 + 3 = 19 combinaisons
        },

        # ------------------------------------------------------------------
        # Random Forest — robuste, peu sensible au sur-ajustement
        # Hyperparamètres clés : n_estimators, max_depth, min_samples_split
        # ------------------------------------------------------------------
        "rf": {
            "label": "Random Forest",
            "estimator": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
            ),
            "param_grid": {
                "n_estimators":      [100, 200, 400],
                "max_depth":         [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "max_features":      ["sqrt", "log2"],
            },
            # 3×4×3×2 = 72 → réduit à 36 en pratique via early stopping
        },

        # ------------------------------------------------------------------
        # Extra Trees — plus aléatoire que RF, généralement plus rapide
        # Hyperparamètres clés : similaires à RF
        # ------------------------------------------------------------------
        "et": {
            "label": "Extra Trees",
            "estimator": ExtraTreesClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
            ),
            "param_grid": {
                "n_estimators":      [100, 200, 400],
                "max_depth":         [None, 15, 30],
                "min_samples_split": [2, 5, 10],
                "max_features":      ["sqrt", "log2"],
            },
            # 3×3×3×2 = 54 combinaisons
        },

        # ------------------------------------------------------------------
        # Gradient Boosting (sklearn) — boosting séquentiel classique
        # Hyperparamètres clés : learning_rate, n_estimators, max_depth
        # ------------------------------------------------------------------
        "gb": {
            "label": "Gradient Boosting",
            "estimator": GradientBoostingClassifier(
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "n_estimators":  [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth":     [3, 5, 7],
                "subsample":     [0.8, 1.0],
            },
            # 3×4×3×2 = 72 combinaisons
        },
    }

    # ------------------------------------------------------------------
    # XGBoost — boosting optimisé, régularisation intégrée
    # Hyperparamètres clés : n_estimators, learning_rate, max_depth,
    #                         subsample, colsample_bytree, reg_alpha/lambda
    # ------------------------------------------------------------------
    if xgb_available:
        registry["xgb"] = {
            "label": "XGBoost",
            "estimator": XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                use_label_encoder=False,
                verbosity=0,
                n_jobs=-1,
                scale_pos_weight=1.56,  # ~400/256 : ratio non_potable/potable
            ),
            "param_grid": {
                "n_estimators":     [100, 200, 300],
                "learning_rate":    [0.01, 0.05, 0.1, 0.2],
                "max_depth":        [3, 5, 7],
                "subsample":        [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
            # 3×4×3×2×2 = 144 → top impactants sélectionnés
        }

    # ------------------------------------------------------------------
    # LightGBM — boosting ultra-rapide (histogrammes)
    # Hyperparamètres clés : num_leaves, learning_rate, n_estimators,
    #                         min_child_samples, reg_alpha
    # ------------------------------------------------------------------
    if lgbm_available:
        registry["lgbm"] = {
            "label": "LightGBM",
            "estimator": LGBMClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
                class_weight="balanced",
            ),
            "param_grid": {
                "n_estimators":      [100, 200, 300],
                "learning_rate":     [0.01, 0.05, 0.1, 0.2],
                "num_leaves":        [31, 63, 127],
                "min_child_samples": [10, 20, 50],
                "reg_alpha":         [0.0, 0.1, 1.0],
            },
            # 3×4×3×3×3 = 324 → grille ciblée la plus importante
        }

    # ------------------------------------------------------------------
    # CatBoost — excellent sur petits datasets, gère les catégorielles
    # Hyperparamètres clés : iterations, learning_rate, depth, l2_leaf_reg
    # ------------------------------------------------------------------
    if catboost_available:
        registry["catboost"] = {
            "label": "CatBoost",
            "estimator": CatBoostClassifier(
                random_state=RANDOM_STATE,
                verbose=0,
                thread_count=-1,
                auto_class_weights="Balanced",
            ),
            "param_grid": {
                "iterations":    [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "depth":         [4, 6, 8, 10],
                "l2_leaf_reg":   [1, 3, 5, 10],
            },
            # 3×4×4×4 = 192 combinaisons
        }

    return registry


# ===========================================================================
# 2. EXÉCUTION DU GRID SEARCH
# ===========================================================================

def run_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_keys: list[str] | None = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Lance le Grid Search sur tous les modèles sélectionnés.

    Parameters
    ----------
    X_train : np.ndarray
        Features d'entraînement standardisées.
    y_train : np.ndarray
        Labels d'entraînement.
    model_keys : list[str] | None, optional
        Sous-ensemble de modèles à optimiser (identifiants courts :
        ``logreg``, ``svm``, ``rf``, ``et``, ``gb``, ``xgb``, ``lgbm``,
        ``catboost``). Si ``None``, tous les modèles disponibles sont utilisés.

    Returns
    -------
    best_params : dict
        Dictionnaire ``{model_key: {"label": str, "params": dict,
        "roc_auc_cv": float, "fit_time_s": float}}``.
    full_report : pd.DataFrame
        Résultats complets de toutes les combinaisons testées,
        prêts à être sauvegardés en CSV.
    """
    registry = _build_model_registry()

    if model_keys:
        unknown = set(model_keys) - set(registry)
        if unknown:
            raise ValueError(f"Modèles inconnus : {unknown}. Disponibles : {set(registry)}")
        registry = {k: v for k, v in registry.items() if k in model_keys}

    # Validation croisée stratifiée interne au GridSearch
    inner_cv = StratifiedKFold(
        n_splits=GRID_SEARCH["cv_splits"],
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    best_params: dict = {}
    all_results: list[pd.DataFrame] = []

    logger.info("=" * 65)
    logger.info("GRID SEARCH — %d modèle(s) à optimiser", len(registry))
    logger.info("=" * 65)

    for key, spec in registry.items():
        label = spec["label"]
        estimator = spec["estimator"]
        param_grid = spec["param_grid"]

        logger.info("\n[%s] Démarrage — %s", key.upper(), label)
        t0 = time.perf_counter()

        gs = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=GMEAN_SCORER,  # G-mean : équilibre recall_0 × recall_1, protège contre le biais classe majoritaire
            cv=inner_cv,
            n_jobs=GRID_SEARCH["n_jobs"],
            verbose=GRID_SEARCH["verbose"],
            return_train_score=GRID_SEARCH["return_train_score"],
            refit=True,
        )

        gs.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        best_score = gs.best_score_
        best_p     = gs.best_params_

        logger.info(
            "[%s] ✓ Terminé en %.1fs | Best G-mean CV : %.4f | Params : %s",
            key.upper(), elapsed, best_score, best_p,
        )

        best_params[key] = {
            "label":       label,
            "params":      best_p,
            "gmean_cv":    round(float(best_score), 5),
            "fit_time_s":  round(elapsed, 1),
            "threshold":   0.5,   # sera affiné par tune_threshold() dans main()
        }

        # Résultats détaillés
        cv_df = pd.DataFrame(gs.cv_results_)
        cv_df.insert(0, "model_key",   key)
        cv_df.insert(1, "model_label", label)
        all_results.append(cv_df)

    full_report = pd.concat(all_results, ignore_index=True)
    return best_params, full_report



# ===========================================================================
# 2b. TUNING DU SEUIL DE DÉCISION
# ===========================================================================

def tune_threshold(
    estimator: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """
    Cherche le seuil de décision optimal sur un jeu de validation.

    Stratégie : parcourir une grille de seuils et retenir celui qui maximise
    le score gmean — un compromis de recall pour les deux classes.

    Parameters
    ----------
    estimator : sklearn estimator (fitté)
        Modèle entraîné exposant ``predict_proba``.
    X_val : np.ndarray
        Features de validation (standardisées).
    y_val : np.ndarray
        Labels de validation.

    Returns
    -------
    float
        Seuil optimal dans [0.20, 0.80].
    """
    from sklearn.metrics import fbeta_score as _fbeta

    thresholds = (
        THRESHOLD_TUNING["thresholds"]
        if THRESHOLD_TUNING["thresholds"] is not None
        else np.arange(0.20, 0.81, 0.01)
    )

    y_proba = estimator.predict_proba(X_val)[:, 1]
    best_thresh, best_score = 0.5, -1.0

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        score = _fbeta(y_val, y_pred_t, beta=THRESHOLD_TUNING["beta"], zero_division=0)
        if score > best_score:
            best_score = score
            best_thresh = t

    logger.info(
        "  Seuil optimal : %.2f  |  Fbeta(beta=%d) = %.4f",
        best_thresh, THRESHOLD_TUNING["beta"], best_score,
    )
    return float(best_thresh)


# ===========================================================================
# 3. SAUVEGARDE DES RÉSULTATS
# ===========================================================================

def save_results(
    best_params: dict,
    full_report: pd.DataFrame,
) -> None:
    """
    Sauvegarde les meilleurs paramètres (JSON) et le rapport complet (CSV).

    Parameters
    ----------
    best_params : dict
        Sortie de :func:`run_grid_search`.
    full_report : pd.DataFrame
        Rapport complet de toutes les combinaisons testées.
    """
    # JSON des meilleurs paramètres
    with open(PATHS["best_params"], "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=4, ensure_ascii=False, default=str)
    logger.info("Meilleurs paramètres sauvegardés → %s", PATHS["best_params"])

    # CSV du rapport complet
    full_report.to_csv(PATHS["tuning_report"], index=False)
    logger.info("Rapport complet sauvegardé → %s", PATHS["tuning_report"])


# ===========================================================================
# 4. VISUALISATIONS
# ===========================================================================

def plot_tuning_summary(best_params: dict) -> None:
    """
    Génère les figures de synthèse du tuning.

    Figures produites :
    - Barplot des meilleurs ROC-AUC par modèle (comparaison directe)
    - Barplot des temps de fit (budget calcul)

    Parameters
    ----------
    best_params : dict
        Sortie de :func:`run_grid_search`.
    """
    plt.style.use(PLOT["style"])
    palette = PLOT["palette"]

    # --- Données ---
    df = pd.DataFrame([
        {
            "Modèle":    v["label"],
            "G-mean":     v["gmean_cv"],
            "Temps (s)": v["fit_time_s"],
        }
        for v in best_params.values()
    ]).sort_values("G-mean", ascending=False)

    # --- Figure 1 : G-mean par modèle ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Résultats du Grid Search — Water Potability", fontsize=14, fontweight="bold")

    # Barplot G-mean
    bars = axes[0].barh(
        df["Modèle"], df["G-mean"],
        color=[palette[i % len(palette)] for i in range(len(df))],
        edgecolor="white", linewidth=0.5,
    )
    axes[0].set_xlabel("G-mean (CV interne)", fontsize=11)
    axes[0].set_title("Meilleur G-mean par modèle", fontsize=12)
    axes[0].set_xlim(0.5, 1.0)
    axes[0].axvline(0.5, color="red", linestyle="--", linewidth=1, label="Seuil min. 0.50")
    axes[0].legend(fontsize=9)

    # Annotations valeurs
    for bar, val in zip(bars, df["G-mean"]):
        axes[0].text(
            val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9,
        )

    # Barplot temps de fit
    axes[1].barh(
        df["Modèle"], df["Temps (s)"],
        color=[palette[i % len(palette)] for i in range(len(df))],
        edgecolor="white", linewidth=0.5,
    )
    axes[1].set_xlabel("Temps de fit (secondes)", fontsize=11)
    axes[1].set_title("Temps d'optimisation par modèle", fontsize=12)

    plt.tight_layout()
    out = PATHS["figures_tuning"] / "tuning_summary.png"
    fig.savefig(out, dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure sauvegardée → %s", out)


def plot_param_heatmap(full_report: pd.DataFrame, model_key: str) -> None:
    """
    Génère une heatmap des scores G-mean pour les deux principaux
    hyperparamètres d'un modèle donné.

    Parameters
    ----------
    full_report : pd.DataFrame
        Rapport complet issu de :func:`run_grid_search`.
    model_key : str
        Identifiant court du modèle (ex. ``"rf"``, ``"xgb"``).
    """
    subset = full_report[full_report["model_key"] == model_key].copy()
    if subset.empty:
        return

    param_cols = [
        c for c in subset.columns
        if c.startswith("param_") and subset[c].notna().any()
    ]
    if len(param_cols) < 2:
        return

    # Sélectionner les 2 hyperparamètres avec le plus de variance de score
    score_col = "mean_test_score"
    p1, p2 = param_cols[0], param_cols[1]

    try:
        pivot = subset.pivot_table(
            index=p1, columns=p2, values=score_col, aggfunc="max"
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, vmin=0.5, vmax=1.0,
        )
        ax.set_title(
            f"G-mean Grid Search — {model_key.upper()}\n"
            f"({p1.replace('param_', '')} × {p2.replace('param_', '')})",
            fontsize=11,
        )
        plt.tight_layout()
        out = PATHS["figures_tuning"] / f"heatmap_{model_key}.png"
        fig.savefig(out, dpi=PLOT["dpi"], bbox_inches="tight")
        plt.close(fig)
        logger.info("Heatmap sauvegardée → %s", out)
    except Exception as e:
        logger.warning("Heatmap impossible pour %s : %s", model_key, e)


# ===========================================================================
# 5. RAPPORT LISIBLE
# ===========================================================================

def print_tuning_report(best_params: dict) -> None:
    """
    Affiche un rapport formaté des résultats de tuning dans la console.

    Parameters
    ----------
    best_params : dict
        Sortie de :func:`run_grid_search`.
    """
    sorted_models = sorted(best_params.items(), key=lambda x: -x[1]["gmean_cv"])

    sep = "=" * 65
    print(f"\n{sep}")
    print("  RAPPORT TUNING — Water Potability Prediction")
    print(f"  Métrique GridSearch : G-mean  |  CV : {GRID_SEARCH['cv_splits']}-fold")
    print(sep)

    for rank, (key, info) in enumerate(sorted_models, 1):
        marker = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else f"  #{rank}"))
        print(f"\n{marker}  [{key.upper()}] {info['label']}")
        print(f"     G-mean CV   : {info['gmean_cv']:.5f}")
        print(f"     Seuil opt.  : {info.get('threshold', 0.5):.2f}")
        print(f"     Fit time    : {info['fit_time_s']:.1f}s")
        print(f"     Best params :")
        for k, v in info["params"].items():
            print(f"       {k:<25} = {v}")

    print(f"\n{sep}")
    print(f"  Rapport complet : {PATHS['tuning_report']}")
    print(f"  Best params JSON: {PATHS['best_params']}")
    print(f"  Figures         : {PATHS['figures_tuning']}")
    print(sep)


# ===========================================================================
# 6. POINT D'ENTRÉE
# ===========================================================================

def main(data_path: str | None = None, model_keys: list[str] | None = None) -> dict:
    """
    Orchestre le tuning complet et retourne les meilleurs paramètres.

    Parameters
    ----------
    data_path : str | None, optional
        Chemin vers le CSV brut. Si None, utilise ``DATA["raw_path"]``.
    model_keys : list[str] | None, optional
        Sous-ensemble de modèles à optimiser. Si None, tous les modèles.

    Returns
    -------
    dict
        Meilleurs paramètres par modèle.
    """
    ensure_dirs()

    # --- Préparation des données ---
    path = data_path or DATA["raw_path"]
    logger.info("Chargement des données depuis : %s", path)
    X, y = raw_data_processing(path, return_X_y=True)
    X_train, X_test, y_train, y_test, scaler = preprocess_for_ml(
        X, y,
        test_size=DATA["test_size"],
        random_state=RANDOM_STATE,
    )

    # --- Rééquilibrage SMOTETomek (train set uniquement) ---
    if REBALANCING["enabled"]:
        try:
            from imblearn.combine import SMOTETomek
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(k_neighbors=REBALANCING["smote_k_neighbors"], random_state=RANDOM_STATE)
            smt = SMOTETomek(smote=smote, random_state=RANDOM_STATE)
            X_train, y_train = smt.fit_resample(X_train, y_train)
            unique, counts = np.unique(y_train, return_counts=True)
            logger.info("SMOTETomek appliqué → classes : %s", dict(zip(unique.tolist(), counts.tolist())))
        except ImportError:
            logger.warning("imbalanced-learn non installé. SMOTETomek ignoré.  pip install imbalanced-learn")

    # --- Grid Search ---
    best_params, full_report = run_grid_search(X_train, y_train, model_keys)

    # --- Threshold tuning par modèle ---
    if THRESHOLD_TUNING["enabled"]:
        logger.info("--- Tuning du seuil de décision ---")
        from sklearn.model_selection import train_test_split as _tts
        X_tr2, X_val, y_tr2, y_val = _tts(
            X_train, y_train, test_size=0.20, random_state=RANDOM_STATE, stratify=y_train
        )
        gs_registry = _build_model_registry()
        for key in best_params:
            if key in gs_registry:
                est = gs_registry[key]["estimator"]
                est.set_params(**best_params[key]["params"])
                est.fit(X_tr2, y_tr2)
                best_params[key]["threshold"] = tune_threshold(est, X_val, y_val)
                logger.info("[%s] seuil optimal : %.2f", key.upper(), best_params[key]["threshold"])

    # --- Sauvegarde ---
    save_results(best_params, full_report)

    # --- Visualisations ---
    plot_tuning_summary(best_params)
    for key in best_params:
        plot_param_heatmap(full_report, key)

    # --- Rapport console ---
    print_tuning_report(best_params)

    return best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Search — Water Potability")
    parser.add_argument(
        "--data", type=str, default=None,
        help="Chemin vers water_potability.csv (défaut : config.DATA['raw_path'])",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        metavar="MODEL_KEY",
        help="Sous-ensemble de modèles : logreg svm rf et gb xgb lgbm catboost",
    )
    args = parser.parse_args()
    main(data_path=args.data, model_keys=args.models)
