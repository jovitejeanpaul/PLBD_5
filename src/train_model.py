"""
train_model.py
==============
Pipeline complet d'entraînement, de validation croisée et d'évaluation
finale pour la prédiction de la potabilité de l'eau.

Workflow
--------
1. Chargement des données brutes via ``data_processing.run_pipeline``
2. Prétraitement ML : split stratifié + RobustScaler
3. Chargement des meilleurs hyperparamètres produits par ``tuning.py``
4. Entraînement avec validation croisée StratifiedKFold (10 folds)
5. Classement des modèles selon ROC-AUC moyen sur CV
6. Évaluation finale des top-N modèles sur le test set
7. Sauvegarde des modèles + scaler (joblib)
8. Génération des figures (ROC curves, matrices de confusion, feature importance)
9. Production du rapport de synthèse (CSV + TXT)

Méthode de validation croisée : StratifiedKFold
-----------------------------------------------
- Préserve le ratio des classes à chaque fold (essentiel sur dataset déséquilibré)
- 10 folds → bonne estimation de la variance de généralisation
- Toutes les métriques sont moyennées sur les 10 folds avec leur écart-type

Usage
-----
    # Utilise les best_params produits par tuning.py
    python train_model.py

    # Forcer le rechargement des données depuis un chemin spécifique
    python train_model.py --data data/raw/water_potability.csv

    # Sauter le tuning (utiliser des params par défaut)
    python train_model.py --no-tuning

Dépendances
-----------
    pip install scikit-learn xgboost lightgbm catboost joblib matplotlib seaborn
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")  # backend non-interactif : evite TclError sur environnements sans GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC

from config import (
    CALIBRATION, CLASS_LABELS, CV, DATA, FBETA_SCORER, FEATURES, GMEAN_SCORER,
    METRICS, MODEL_SELECTION, PATHS, PLOT, POSITIVE_CLASS, RANDOM_STATE,
    REBALANCING, TARGET, THRESHOLD_TUNING, ensure_dirs,
)
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
# 0. WRAPPER — MODÈLE AVEC SEUIL INTÉGRÉ
# ===========================================================================

from threshold_classifier import ThresholdClassifier  # noqa: E402


# ===========================================================================
# 1. CONSTRUCTION DES MODÈLES À PARTIR DES BEST PARAMS
# ===========================================================================

def _base_estimators() -> dict[str, Any]:
    """
    Retourne les estimateurs de base (sans hyperparamètres tunés).
    Utilisé comme fallback si best_params.json n'existe pas.
    """
    estimators: dict[str, Any] = {
        "logreg": LogisticRegression(random_state=RANDOM_STATE, max_iter=2000, solver="saga", class_weight="balanced"),
        "svm":    SVC(probability=True, random_state=RANDOM_STATE, class_weight="balanced"),
        "rf":     RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
        "et":     ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
        "gb":     GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    try:
        from xgboost import XGBClassifier
        estimators["xgb"] = XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss",
            verbosity=0, n_jobs=-1,
            scale_pos_weight=1.56,
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        estimators["lgbm"] = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, class_weight="balanced")
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier
        estimators["catboost"] = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, auto_class_weights="Balanced")
    except ImportError:
        pass
    return estimators


def build_models(best_params_path: Path | None = None) -> dict[str, Any]:
    """
    Construit les estimateurs sklearn instanciés avec les meilleurs hyperparamètres.

    Si ``best_params_path`` est fourni et valide, chaque estimateur est
    instancié avec les paramètres issus du Grid Search. Sinon, les valeurs
    par défaut sont utilisées (avec un avertissement).

    Parameters
    ----------
    best_params_path : Path | None
        Chemin vers ``best_params.json`` produit par ``tuning.py``.

    Returns
    -------
    dict[str, estimator]
        Dictionnaire ``{model_key: estimateur_sklearn_instancié}``.
    """
    base = _base_estimators()

    if best_params_path is None or not best_params_path.exists():
        logger.warning(
            "best_params.json introuvable (%s). "
            "Utilisation des hyperparamètres par défaut. "
            "Lancez d'abord : python tuning.py",
            best_params_path,
        )
        return base

    with open(best_params_path, encoding="utf-8") as f:
        best_params: dict = json.load(f)

    models: dict[str, Any] = {}
    for key, estimator in base.items():
        if key in best_params:
            params = best_params[key].get("params", {})
            try:
                estimator.set_params(**params)
                models[key] = estimator
                logger.info("[%s] Hyperparamètres chargés : %s", key.upper(), params)
            except Exception as e:
                logger.warning("[%s] set_params() échoué (%s). Params par défaut utilisés.", key, e)
                models[key] = estimator
        else:
            models[key] = estimator
            logger.warning("[%s] Absent de best_params.json — valeurs par défaut.", key.upper())

    return models


# ===========================================================================
# 2. VALIDATION CROISÉE
# ===========================================================================

def cross_validate_models(
    models: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> pd.DataFrame:
    """
    Évalue tous les modèles par validation croisée StratifiedKFold.

    La validation croisée stratifiée préserve la proportion de chaque classe
    dans chaque fold — indispensable sur ce dataset légèrement déséquilibré.
    Les métriques sont calculées sur les folds de *validation* uniquement
    (prévention du data leakage).

    Parameters
    ----------
    models : dict[str, estimator]
        Modèles à évaluer.
    X_train : np.ndarray
        Features d'entraînement standardisées.
    y_train : np.ndarray
        Labels d'entraînement.

    Returns
    -------
    pd.DataFrame
        Tableau des métriques moyennes ± écart-type pour chaque modèle,
        trié par ROC-AUC décroissant.
    """
    cv = StratifiedKFold(
        n_splits=CV["n_splits"],
        shuffle=CV["shuffle"],
        random_state=RANDOM_STATE,
    )

    results = []
    logger.info("=" * 65)
    logger.info("VALIDATION CROISÉE — StratifiedKFold(%d)", CV["n_splits"])
    logger.info("=" * 65)

    for key, model in models.items():
        logger.info("\n[%s] Évaluation en cours…", key.upper())

        try:
            # Ajouter gmean au scoring dict pour cross_validate
            scoring_cv = dict(METRICS["scoring"])
            scoring_cv["gmean"] = GMEAN_SCORER
            scores = cross_validate(
                model,
                X_train, y_train,
                cv=cv,
                scoring=scoring_cv,
                n_jobs=-1,
                return_train_score=False,
                error_score="raise",
            )

            row = {"model_key": key}
            all_metrics = list(METRICS["scoring"].keys()) + ["gmean"]
            for metric in all_metrics:
                fold_scores = scores[f"test_{metric}"]
                row[f"{metric}_mean"] = round(float(fold_scores.mean()), 5)
                row[f"{metric}_std"]  = round(float(fold_scores.std()),  5)

            results.append(row)

            logger.info(
                "[%s] ✓  G-mean : %.4f ± %.4f  |  ROC-AUC : %.4f ± %.4f  |  F1 : %.4f ± %.4f  |  Acc : %.4f ± %.4f",
                key.upper(),
                row.get("gmean_mean", 0), row.get("gmean_std", 0),
                row["roc_auc_mean"], row["roc_auc_std"],
                row["f1_mean"],      row["f1_std"],
                row["accuracy_mean"],row["accuracy_std"],
            )

        except Exception as e:
            logger.error("[%s] Échec de la CV : %s", key.upper(), e)

    df = pd.DataFrame(results)
    df = df.sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ===========================================================================
# 3. SÉLECTION DES TOP-N MODÈLES
# ===========================================================================

def select_top_models(
    cv_results: pd.DataFrame,
    models: dict[str, Any],
    top_n: int = MODEL_SELECTION["top_n"],
) -> dict[str, Any]:
    """
    Sélectionne les N meilleurs modèles selon la métrique primaire (ROC-AUC).

    Filtre également les modèles dont le ROC-AUC moyen est inférieur au
    seuil défini dans :data:`config.MODEL_SELECTION`.

    Parameters
    ----------
    cv_results : pd.DataFrame
        Sortie de :func:`cross_validate_models`.
    models : dict
        Tous les modèles disponibles.
    top_n : int
        Nombre de modèles à conserver.

    Returns
    -------
    dict[str, estimator]
        Sous-ensemble des ``top_n`` meilleurs modèles.
    """
    top_keys = cv_results.head(top_n)["model_key"].tolist()
    min_auc = MODEL_SELECTION["min_roc_auc"]

    filtered = []
    for key in top_keys:
        row = cv_results[cv_results["model_key"] == key].iloc[0]
        if row["roc_auc_mean"] >= min_auc:
            filtered.append(key)
        else:
            logger.warning(
                "[%s] Écarté : ROC-AUC %.4f < seuil min %.2f",
                key.upper(), row["roc_auc_mean"], min_auc,
            )

    if not filtered:
        logger.warning("Aucun modèle ne dépasse le seuil ROC-AUC. Conservation du top-%d.", top_n)
        filtered = top_keys

    logger.info("Top-%d modèles sélectionnés : %s", len(filtered), filtered)
    return {k: models[k] for k in filtered if k in models}


# ===========================================================================
# 4. ENTRAÎNEMENT FINAL & ÉVALUATION SUR LE TEST SET
# ===========================================================================

def train_and_evaluate(
    top_models: dict[str, Any],
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    cv_results: pd.DataFrame,
    best_params_thresholds: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Entraîne les top modèles sur le train set complet et évalue sur le test set.

    Parameters
    ----------
    top_models : dict
        Modèles sélectionnés par :func:`select_top_models`.
    X_train, X_test : np.ndarray
        Features standardisées.
    y_train, y_test : np.ndarray
        Labels.
    cv_results : pd.DataFrame
        Métriques CV pour enrichir le rapport final.

    Returns
    -------
    eval_report : pd.DataFrame
        Métriques CV et test pour chaque modèle.
    fitted_models : dict[str, fitted_estimator]
        Modèles entraînés sur le train set complet.
    """
    fitted_models: dict[str, Any] = {}
    eval_rows = []

    logger.info("=" * 65)
    logger.info("ENTRAÎNEMENT FINAL & ÉVALUATION TEST")
    logger.info("=" * 65)

    for key, model in top_models.items():

        # Récupérer le seuil optimal AVANT l'entraînement
        threshold = 0.5
        if best_params_thresholds and key in best_params_thresholds:
            threshold = best_params_thresholds[key]

        logger.info("\n[%s] Entraînement sur le train set complet…", key.upper())
        logger.info("  Seuil intégré au modèle : %.3f  (ThresholdClassifier)", threshold if best_params_thresholds and key in (best_params_thresholds or {}) else 0.5)


        # Calibration des probabilités (optionnelle)
        if CALIBRATION["enabled"]:
            logger.info("  Calibration des probabilités (%s, cv=%d)…",
                        CALIBRATION["method"], CALIBRATION["cv"])
            calibrated = CalibratedClassifierCV(
                model,
                method=CALIBRATION["method"],
                cv=CALIBRATION["cv"],
            )
            calibrated.fit(X_train, y_train)
            estimator_to_wrap = calibrated
        else:
            model.fit(X_train, y_train)
            estimator_to_wrap = model

        wrapped = ThresholdClassifier(estimator=estimator_to_wrap, threshold=threshold)
        wrapped.classes_ = np.array([0, 1])
        fitted_models[key] = wrapped

        # Prédictions cohérentes avec le seuil intégré
        y_proba = wrapped.predict_proba(X_test)[:, 1]
        y_pred  = wrapped.predict(X_test)   # utilise threshold automatiquement

        # Métriques test complètes
        from sklearn.metrics import fbeta_score as _fbeta
        from sklearn.metrics import recall_score as _recall
        import numpy as _np
        _r1 = _recall(y_test, y_pred, pos_label=1, zero_division=0)
        _r0 = _recall(y_test, y_pred, pos_label=0, zero_division=0)
        _gmean_val = float(_np.sqrt(_r1 * _r0))
        from sklearn.metrics import brier_score_loss
        test_metrics = {
            "threshold":      round(threshold, 3),
            "calibrated":     CALIBRATION["enabled"],
            "gmean_test":     round(_gmean_val, 5),
            "roc_auc_test":   round(roc_auc_score(y_test, y_proba), 5),
            "pr_auc_test":    round(average_precision_score(y_test, y_proba), 5),
            "mcc_test":       round(matthews_corrcoef(y_test, y_pred), 5),
            "brier_score":    round(brier_score_loss(y_test, y_proba), 5),
            "fbeta_test":     round(_fbeta(y_test, y_pred, beta=2, zero_division=0), 5),
            "f1_test":        round(f1_score(y_test, y_pred, zero_division=0), 5),
            "precision_test": round(precision_score(y_test, y_pred, zero_division=0), 5),
            "recall_test":    round(recall_score(y_test, y_pred, zero_division=0), 5),
            "accuracy_test":  round(accuracy_score(y_test, y_pred), 5),
        }

        logger.info(
            "[%s] TEST  seuil=%.2f | Gmean=%.4f | AUC=%.4f | PR-AUC=%.4f | MCC=%.4f | Fbeta=%.4f | F1=%.4f | Recall=%.4f",
            key.upper(), test_metrics["threshold"], test_metrics["gmean_test"],
            test_metrics["roc_auc_test"], test_metrics["pr_auc_test"], test_metrics["mcc_test"],
            test_metrics["fbeta_test"], test_metrics["f1_test"], test_metrics["recall_test"],
        )

        # Rapport de classification
        report_str = classification_report(y_test, y_pred, target_names=[CLASS_LABELS[0], CLASS_LABELS[1]])
        logger.info("[%s] Rapport de classification :\n%s", key.upper(), report_str)

        # Fusion CV + test
        cv_row = cv_results[cv_results["model_key"] == key]
        cv_dict = cv_row.drop(columns=["model_key"]).iloc[0].to_dict() if not cv_row.empty else {}

        eval_rows.append({
            "model_key": key,
            **cv_dict,
            **test_metrics,
        })

    eval_report = pd.DataFrame(eval_rows)

    return eval_report, fitted_models


# ===========================================================================
# 5. SAUVEGARDE DES MODÈLES
# ===========================================================================

def save_models(
    fitted_models: dict[str, Any],
    scaler: Any,
    eval_report: pd.DataFrame,
) -> None:
    """
    Sérialise les modèles entraînés et le scaler avec joblib.

    Convention de nommage :
        ``outputs/models/model_{rank}_{key}.joblib``
        ``outputs/models/scaler.joblib``

    Parameters
    ----------
    fitted_models : dict
        Modèles entraînés.
    scaler : sklearn transformer
        RobustScaler fitté sur le train set.
    eval_report : pd.DataFrame
        Utilisé pour déterminer le rang de chaque modèle.
    """
    # Nettoyer les anciens fichiers model_*.joblib avant d'écrire les nouveaux
    for old in PATHS["models"].glob("model_*.joblib"):
        old.unlink()
        logger.info("Ancien modèle supprimé : %s", old.name)

    # Scaler
    scaler_path = PATHS["models"] / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler sauvegardé → %s", scaler_path)

    # Modèles classés par score composite
    ranked_keys = eval_report["model_key"].tolist()

    for rank, key in enumerate(ranked_keys, 1):
        if key not in fitted_models:
            continue
        filename = f"model_{rank}_{key}{MODEL_SELECTION['model_format']}"
        path = PATHS["models"] / filename
        joblib.dump(fitted_models[key], path)
        logger.info("Modèle #%d [%s] sauvegardé → %s", rank, key.upper(), path)


# ===========================================================================
# 6. VISUALISATIONS
# ===========================================================================

def plot_roc_curves(
    fitted_models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    eval_report: pd.DataFrame,
) -> None:
    """
    Trace les courbes ROC de tous les top modèles sur un même graphe.

    Parameters
    ----------
    fitted_models : dict
        Modèles entraînés.
    X_test, y_test : np.ndarray
        Données de test.
    eval_report : pd.DataFrame
        Pour récupérer le ROC-AUC de chaque modèle.
    """
    plt.style.use(PLOT["style"])
    fig, ax = plt.subplots(figsize=PLOT["figsize"])

    palette = PLOT["palette"]

    for i, (key, model) in enumerate(fitted_models.items()):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        rank = eval_report[eval_report["model_key"] == key]["rank"].values
        rank_label = f"#{rank[0]}" if len(rank) > 0 else ""
        ax.plot(
            fpr, tpr,
            color=palette[i % len(palette)],
            lw=2,
            label=f"{rank_label} {key.upper()} (AUC = {roc_auc:.4f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aléatoire (AUC = 0.50)")
    ax.set_xlabel("Taux de Faux Positifs (FPR)", fontsize=12)
    ax.set_ylabel("Taux de Vrais Positifs (TPR)", fontsize=12)
    ax.set_title("Courbes ROC — Comparaison des modèles (Test set)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.01])

    plt.tight_layout()
    out = PATHS["figures_eval"] / "roc_curves.png"
    fig.savefig(out, dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curves → %s", out)


def plot_confusion_matrices(
    fitted_models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    thresholds: dict | None = None,
) -> None:
    """
    Génère les matrices de confusion de tous les top modèles.

    Utilise le seuil de décision optimal par modèle (issu du threshold tuning)
    pour garantir la cohérence avec le rapport de classification.

    Parameters
    ----------
    fitted_models : dict
        Modèles entraînés.
    X_test, y_test : np.ndarray
        Données de test.
    thresholds : dict | None
        Seuils optimaux par modèle ``{model_key: float}``.
        Si None, utilise 0.5 (comportement sklearn par défaut).
    """
    n = len(fitted_models)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    plt.suptitle("Matrices de confusion — Test set", fontsize=14, fontweight="bold", y=1.01)

    for ax, (key, model) in zip(axes, fitted_models.items()):
        # ThresholdClassifier.predict() applique automatiquement le seuil optimal
        # Plus besoin de le passer séparément — il est encapsulé dans le modèle.
        threshold = getattr(model, "threshold", 0.5)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=[CLASS_LABELS[0], CLASS_LABELS[1]])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{key.upper()}  (seuil = {threshold:.2f})",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Prédit", fontsize=10)
        ax.set_ylabel("Réel", fontsize=10)

    # Masquer les axes vides
    for ax in axes[len(fitted_models):]:
        ax.set_visible(False)

    plt.tight_layout()
    out = PATHS["figures_eval"] / "confusion_matrices.png"
    fig.savefig(out, dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close(fig)
    logger.info("Matrices de confusion → %s", out)


def plot_feature_importance(
    fitted_models: dict[str, Any],
    feature_names: list[str],
) -> None:
    """
    Trace les importances des features pour les modèles arborescents.

    Seuls les modèles exposant ``feature_importances_`` sont traités
    (RandomForest, ExtraTrees, GradientBoosting, XGBoost, LightGBM, CatBoost).

    Parameters
    ----------
    fitted_models : dict
        Modèles entraînés.
    feature_names : list[str]
        Noms des features dans l'ordre du DataFrame X.
    """
    importances_data = {}

    for key, model in fitted_models.items():
        # ThresholdClassifier délègue feature_importances_ et coef_
        # via les propriétés définies dans la classe wrapper
        if model.feature_importances_ is not None:
            importances_data[key] = model.feature_importances_
        elif model.coef_ is not None:
            importances_data[key] = np.abs(model.coef_[0])

    if not importances_data:
        logger.info("Aucun modèle n'expose feature_importances_. Figure ignorée.")
        return

    n = len(importances_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    axes = [axes] if n == 1 else axes

    plt.suptitle("Importance des features", fontsize=14, fontweight="bold")
    palette = PLOT["palette"]

    for ax, ((key, imps), color) in zip(axes, zip(importances_data.items(), palette)):
        indices = np.argsort(imps)[::-1]
        sorted_names = [feature_names[i] for i in indices]
        sorted_imps  = imps[indices]

        ax.barh(sorted_names[::-1], sorted_imps[::-1], color=color, edgecolor="white")
        ax.set_title(key.upper(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Importance", fontsize=10)

    plt.tight_layout()
    out = PATHS["figures_eval"] / "feature_importance.png"
    fig.savefig(out, dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance → %s", out)


def plot_cv_comparison(cv_results: pd.DataFrame) -> None:
    """
    Barplot comparatif des métriques CV (mean ± std) pour tous les modèles.

    Parameters
    ----------
    cv_results : pd.DataFrame
        Sortie de :func:`cross_validate_models`.
    """
    metrics_to_plot = ["roc_auc", "f1", "accuracy", "precision", "recall"]
    available = [m for m in metrics_to_plot if f"{m}_mean" in cv_results.columns]

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 6))
    axes = [axes] if n == 1 else axes
    palette = PLOT["palette"]

    fig.suptitle(
        f"Validation croisée StratifiedKFold({CV['n_splits']}) — Toutes métriques",
        fontsize=13, fontweight="bold",
    )

    for ax, metric in zip(axes, available):
        means = cv_results[f"{metric}_mean"]
        stds  = cv_results[f"{metric}_std"]
        keys  = cv_results["model_key"]

        bars = ax.barh(
            keys, means,
            xerr=stds,
            color=[palette[i % len(palette)] for i in range(len(keys))],
            edgecolor="white",
            capsize=4,
            error_kw={"elinewidth": 1.5},
        )
        ax.set_title(metric.replace("_", " ").upper(), fontsize=11)
        ax.set_xlim(0.4, 1.0)
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8)

        for bar, val, std in zip(bars, means, stds):
            ax.text(
                min(val + std + 0.01, 0.98),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8,
            )

    plt.tight_layout()
    out = PATHS["figures_eval"] / "cv_comparison.png"
    fig.savefig(out, dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close(fig)
    logger.info("CV comparison → %s", out)


def plot_calibration_diagram(
    fitted_models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Diagramme de fiabilité (reliability diagram) pour évaluer la calibration
    des probabilités de chaque modèle.

    Un modèle parfaitement calibré suit la diagonale : quand il dit p=0.6,
    60% des échantillons sont effectivement positifs.
    """
    from sklearn.calibration import calibration_curve

    n = len(fitted_models)
    fig, axes = plt.subplots(1, min(n, 3), figsize=(6 * min(n, 3), 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Diagramme de fiabilité — Calibration des probabilités",
                 fontsize=13, fontweight="bold")
    palette = PLOT["palette"]

    for ax, ((key, model), color) in zip(axes, zip(fitted_models.items(), palette)):
        y_proba = model.predict_proba(X_test)[:, 1]

        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="uniform")

        ax.plot(prob_pred, prob_true, "o-", color=color, lw=2, label=key.upper())
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Calibration parfaite")
        ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.15, color=color)

        ax.set_xlabel("Probabilité prédite", fontsize=10)
        ax.set_ylabel("Proportion réelle", fontsize=10)
        ax.set_title(f"{key.upper()}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

        from sklearn.metrics import brier_score_loss
        brier = brier_score_loss(y_test, y_proba)
        ax.text(0.05, 0.92, f"Brier: {brier:.4f}", transform=ax.transAxes,
                fontsize=9, color=color, fontweight="bold")

    for ax in axes[len(fitted_models):]:
        ax.set_visible(False)

    plt.tight_layout()
    out = PATHS["figures_eval"] / "calibration_diagram.png"
    fig.savefig(out, dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close(fig)
    logger.info("Calibration diagram → %s", out)


# ===========================================================================
# 7. RAPPORT SYNTHÈSE
# ===========================================================================

def write_summary_report(
    cv_results: pd.DataFrame,
    eval_report: pd.DataFrame,
    best_params_path: Path,
) -> None:
    """
    Écrit un rapport texte lisible synthétisant les résultats.

    Contient :
    - Configuration du run (date, random state, CV strategy)
    - Tableau CV de tous les modèles
    - Performances test des top-N modèles
    - Meilleurs hyperparamètres

    Parameters
    ----------
    cv_results : pd.DataFrame
        Sortie de :func:`cross_validate_models`.
    eval_report : pd.DataFrame
        Sortie de :func:`train_and_evaluate`.
    best_params_path : Path
        Chemin vers le JSON des meilleurs paramètres.
    """
    import datetime

    lines = []
    sep = "=" * 70

    lines += [
        sep,
        "  RAPPORT D'ÉVALUATION — Water Potability Prediction",
        f"  Généré le : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Random state : {RANDOM_STATE}",
        f"  CV strategy  : StratifiedKFold(n_splits={CV['n_splits']}, shuffle={CV['shuffle']})",
        f"  Métrique principale : {METRICS['primary'].upper()}",
        sep, "",
    ]

    # --- Section CV ---
    lines += ["  VALIDATION CROISÉE (Train set)", "-" * 40]
    cv_display = cv_results[
        ["rank", "model_key", "roc_auc_mean", "roc_auc_std",
         "f1_mean", "f1_std", "accuracy_mean", "accuracy_std"]
    ].copy()
    cv_display.columns = ["Rang", "Modèle", "AUC moy.", "AUC std",
                          "F1 moy.", "F1 std", "Acc moy.", "Acc std"]
    lines.append(cv_display.to_string(index=False))
    lines.append("")

    # --- Section test ---
    lines += ["  ÉVALUATION FINALE (Test set — données jamais vues)", "-" * 40]
    cols_test = [c for c in ["model_key", "composite_score", "roc_auc_test", "pr_auc_test",
                  "mcc_test", "fbeta_test", "recall_test", "f1_test", "accuracy_test", "threshold"]
                  if c in eval_report.columns]
    test_display = eval_report[cols_test].copy()
    test_display.columns = [c.replace("_test","").replace("_"," ").title() for c in cols_test]
    lines.append(test_display.to_string(index=False))
    lines.append("")

    # --- Best params ---
    if best_params_path.exists():
        with open(best_params_path, encoding="utf-8") as f:
            best_params = json.load(f)

        lines += ["  MEILLEURS HYPERPARAMÈTRES (Grid Search)", "-" * 40]
        for key, info in best_params.items():
            lines.append(f"\n  [{key.upper()}] {info.get('label', key)}")
            lines.append(f"    ROC-AUC CV : {info.get('roc_auc_cv', 'N/A')}")
            for p, v in info.get("params", {}).items():
                lines.append(f"    {p:<30} = {v}")

    lines += ["", sep, "  Figures   : " + str(PATHS["figures_eval"]), sep]

    report_text = "\n".join(lines)
    print(f"\n{report_text}")

    # Sauvegarde fichier texte
    PATHS["summary_txt"].write_text(report_text, encoding="utf-8")
    logger.info("Rapport synthèse → %s", PATHS["summary_txt"])

    # Sauvegarde CSV évaluation
    eval_report.to_csv(PATHS["eval_report"], index=False)
    logger.info("Rapport CSV → %s", PATHS["eval_report"])


# ===========================================================================
# 8. PIPELINE PRINCIPAL
# ===========================================================================

def main(
    data_path: str | None = None,
    best_params_path: Path | None = None,
    top_n: int = MODEL_SELECTION["top_n"],
) -> dict[str, Any]:
    """
    Orchestre le pipeline complet d'entraînement et d'évaluation.

    Parameters
    ----------
    data_path : str | None
        Chemin vers le CSV brut. Si None, utilise ``DATA["raw_path"]``.
    best_params_path : Path | None
        Chemin vers ``best_params.json``. Si None, utilise ``PATHS["best_params"]``.
    top_n : int
        Nombre de modèles à conserver. Par défaut : ``MODEL_SELECTION["top_n"]`` = 3.

    Returns
    -------
    dict
        ``{"fitted_models": ..., "eval_report": ..., "cv_results": ..., "scaler": ...}``
    """
    ensure_dirs()
    bp_path = best_params_path or PATHS["best_params"]

    # 1. Données
    path = data_path or DATA["raw_path"]
    logger.info("Chargement des données : %s", path)
    X, y = raw_data_processing(path, return_X_y=True)
    X_train, X_test, y_train, y_test, scaler = preprocess_for_ml(
        X, y, test_size=DATA["test_size"], random_state=RANDOM_STATE,
    )

    # 1b. Rééquilibrage SMOTETomek (train set uniquement)
    if REBALANCING["enabled"]:
        try:
            from imblearn.combine import SMOTETomek
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(k_neighbors=REBALANCING["smote_k_neighbors"], random_state=RANDOM_STATE)
            smt = SMOTETomek(smote=smote, random_state=RANDOM_STATE)
            X_train, y_train = smt.fit_resample(X_train, y_train)
            unique, counts = np.unique(y_train, return_counts=True)
            logger.info("SMOTETomek → distribution train : %s", dict(zip(unique.tolist(), counts.tolist())))
        except ImportError:
            logger.warning("imbalanced-learn non installé. SMOTETomek ignoré. pip install imbalanced-learn")

    # 2. Construction des modèles
    models = build_models(bp_path)

    # 3. Validation croisée
    cv_results = cross_validate_models(models, X_train, y_train)

    # 4. Sélection top-N
    top_models = select_top_models(cv_results, models, top_n)

    # 5. Entraînement final + évaluation test
    # Charger les seuils optimaux depuis best_params si disponibles
    thresholds = {}
    if bp_path and bp_path.exists():
        import json as _json
        with open(bp_path) as _f:
            _bp = _json.load(_f)
        thresholds = {k: v.get("threshold", 0.5) for k, v in _bp.items()}

    eval_report, fitted_models = train_and_evaluate(
        top_models, X_train, X_test, y_train, y_test, cv_results,
        best_params_thresholds=thresholds,
    )

    # Classement composite pondéré (selon METRICS["composite_weights"])
    weights = METRICS["composite_weights"]
    def _composite(row):
        score = 0.0
        mapping = {
            "gmean":   "gmean_test",
            "roc_auc": "roc_auc_test",
            "pr_auc":  "pr_auc_test",
            "fbeta":   "fbeta_test",
            "mcc":     "mcc_test",
        }
        for metric, col in mapping.items():
            if metric in weights and col in row.index:
                score += weights[metric] * row[col]
        return round(score, 5)

    eval_report["composite_score"] = eval_report.apply(_composite, axis=1)
    eval_report = eval_report.sort_values("composite_score", ascending=False).reset_index(drop=True)

    if "rank" not in eval_report.columns:
        eval_report.insert(0, "rank", range(1, len(eval_report) + 1))
    else:
        eval_report["rank"] = range(1, len(eval_report) + 1)

    # 6. Sauvegarde modèles + scaler
    save_models(fitted_models, scaler, eval_report)

    # 7. Figures
    plot_roc_curves(fitted_models, X_test, y_test, eval_report)
    plot_confusion_matrices(fitted_models, X_test, y_test, thresholds=thresholds)
    plot_feature_importance(fitted_models, FEATURES)
    plot_cv_comparison(cv_results)
    plot_calibration_diagram(fitted_models, X_test, y_test)

    # 8. Rapport synthèse
    write_summary_report(cv_results, eval_report, bp_path)

    logger.info("Pipeline terminé. Tous les artefacts dans : %s", PATHS["outputs"])

    return {
        "fitted_models": fitted_models,
        "eval_report":   eval_report,
        "cv_results":    cv_results,
        "scaler":        scaler,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train & Evaluate — Water Potability")
    parser.add_argument(
        "--data", type=str, default=None,
        help="Chemin vers water_potability.csv",
    )
    parser.add_argument(
        "--best-params", type=str, default=None,
        help="Chemin vers best_params.json (produit par tuning.py)",
    )
    parser.add_argument(
        "--top-n", type=int, default=MODEL_SELECTION["top_n"],
        help=f"Nombre de modèles à conserver (défaut : {MODEL_SELECTION['top_n']})",
    )
    parser.add_argument(
        "--no-tuning", action="store_true",
        help="Ignorer best_params.json et utiliser les hyperparamètres par défaut",
    )
    args = parser.parse_args()

    bp = None if args.no_tuning else (Path(args.best_params) if args.best_params else None)
    main(data_path=args.data, best_params_path=bp, top_n=args.top_n)
