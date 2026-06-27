"""
explainability.py
==================
Explicabilité SHAP du meilleur modèle de potabilité de l'eau.

Architecture offline / embarquée
---------------------------------
Ce module suit une séparation stricte entre deux contextes d'exécution :

**PC (offline)**
    1. Charger le modèle rank-1 et le scaler sauvegardés par train_model.py.
    2. Calculer les SHAP values globales sur le jeu de test.
    3. Sauvegarder un résumé ultra-léger (``shap_summary.json``) :
       importance moyenne par feature + métadonnées du modèle.
    4. Générer les figures (bar plot, beeswarm).

**Raspberry Pi (temps réel)**
    • Charger ``shap_summary.json`` une seule fois au démarrage.
    • Diffuser l'importance globale des features instantanément, sans
      aucun calcul SHAP à la volée (trop coûteux pour un SBC ARM).
    • La fonction ``load_shap_summary`` est la seule à appeler côté Pi.

Sélection automatique de l'explainer
--------------------------------------
+----------------------------+-------------------------------+-----------+
| Modèle interne             | Explainer utilisé             | Vitesse   |
+============================+===============================+===========+
| Tree (RF, ET, GB, XGB…)   | ``shap.TreeExplainer``        | très rapide |
| Linéaire (LogReg, Ridge…) | ``shap.LinearExplainer``      | rapide    |
| Boîte noire (SVM, KNN…)   | ``shap.KernelExplainer``      | lent (!)  |
|                            | (background = kmeans, k=50)   |           |
+----------------------------+-------------------------------+-----------+

Pour les modèles boîte noire, ``KernelExplainer`` peut prendre plusieurs
minutes sur PC. Ne jamais l'appeler sur le Raspberry Pi.

Usage rapide
-------------
    # PC — générer le résumé
    from src.explainability import main as explain_main
    explain_main()

    # Raspberry Pi — consommer le résumé
    from src.models_training.explainability import load_shap_summary
    summary = load_shap_summary()   # dict JSON, chargé en <1 ms
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional, List

import joblib
import numpy as np
import pandas as pd
from train_model import ThresholdClassifier
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports conditionnels (SHAP et matplotlib ne sont pas requis côté Pi)
# ---------------------------------------------------------------------------
try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    logger.warning(
        "Le package 'shap' n'est pas installé. "
        "Les fonctions de calcul SHAP ne sont pas disponibles. "
        "Sur le Raspberry Pi, utilisez uniquement load_shap_summary()."
    )

try:
    import matplotlib
    matplotlib.use("Agg")   # backend non-interactif (compatible Pi headless)
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Imports projet
# ---------------------------------------------------------------------------
try:
    from config import FEATURES, PATHS, PLOT, TARGET
except ImportError:
    # Fallback minimal pour les environnements sans config.py sur le path
    FEATURES = ["ph", "Solids", "Conductivity", "Turbidity"]
    TARGET = "Potability"
    PATHS = {
        "models":         Path("outputs/models"),
        "figures_eval":   Path("outputs/figures/evaluation"),
        "reports":        Path("outputs/reports"),
    }
    PLOT = {"dpi": 150, "palette": ["#2E86AB", "#E84855", "#3BB273", "#F18F01"]}

# Chemin par défaut du résumé JSON (léger, embarquable sur le Pi)
_DEFAULT_SUMMARY_PATH = PATHS["reports"] / "shap_summary.json"


# ===========================================================================
# SÉLECTION DE L'EXPLAINER
# ===========================================================================

def get_explainer(
    estimator,
    X_background: np.ndarray,
    *,
    kernel_background_k: int = 50,
):
    """
    Sélectionne et instancie l'explainer SHAP le plus rapide pour
    ``estimator``.

    Parameters
    ----------
    estimator :
        Estimateur sklearn déjà fitté (l'estimateur **brut**, pas le
        ``ThresholdClassifier`` wrapper — utiliser
        ``threshold_clf.estimator`` pour l'extraire).
    X_background : np.ndarray, shape (n_samples, n_features)
        Données d'arrière-plan pour ``LinearExplainer`` et
        ``KernelExplainer``. Pour ``TreeExplainer`` elles sont ignorées
        (mais requises par signature pour l'uniformité).
    kernel_background_k : int, default 50
        Nombre de centroids kmeans utilisés comme background pour
        ``KernelExplainer``. Réduire (ex. 20) pour accélérer sur petits
        jeux de données.

    Returns
    -------
    explainer
        Instance SHAP prête à l'emploi.

    Raises
    ------
    ImportError
        Si ``shap`` n'est pas installé.
    """
    if not _SHAP_AVAILABLE:
        raise ImportError(
            "Le package 'shap' est requis pour les calculs d'explicabilité. "
            "Installez-le avec : pip install shap"
        )

    estimator_type = type(estimator).__name__

    # — Modèles à base d'arbres décisionnels —
    _tree_types = (
        "DecisionTreeClassifier", "RandomForestClassifier",
        "ExtraTreesClassifier", "GradientBoostingClassifier",
        "XGBClassifier", "LGBMClassifier", "CatBoostClassifier",
        "HistGradientBoostingClassifier",
    )
    if estimator_type in _tree_types:
        logger.info("TreeExplainer sélectionné pour %s", estimator_type)
        return _shap.TreeExplainer(estimator)

    # — Modèles linéaires —
    _linear_types = (
        "LogisticRegression", "LogisticRegressionCV",
        "RidgeClassifier", "SGDClassifier",
        "LinearSVC",
    )
    if estimator_type in _linear_types:
        logger.info("LinearExplainer sélectionné pour %s", estimator_type)
        return _shap.LinearExplainer(estimator, X_background)

    # — Boîte noire (SVM RBF, KNN, etc.) : KernelExplainer avec background kmeans —
    logger.warning(
        "KernelExplainer sélectionné pour %s. "
        "Ce calcul peut être lent. Ne pas exécuter sur Raspberry Pi.",
        estimator_type,
    )
    background = _shap.kmeans(X_background, kernel_background_k)
    return _shap.KernelExplainer(estimator.predict_proba, background)


# ===========================================================================
# CALCUL DES SHAP VALUES GLOBALES
# ===========================================================================

def compute_global_shap(
    estimator,
    X_test: np.ndarray,
    X_background: np.ndarray,
    feature_names: Optional[List[str]] = None,
    *,
    kernel_background_k: int = 50,
) -> Optional[dict]:
    """
    Calcule les SHAP values sur ``X_test`` et retourne un résumé global.

    Le résumé contient, pour chaque feature :
    - ``mean_abs_shap`` : importance moyenne |SHAP| sur le test set
      (utilisée pour le ranking des features)
    - ``shap_values`` : matrice complète (shape n × p), conservée pour
      les figures beeswarm/scatter

    Parameters
    ----------
    estimator :
        Estimateur brut fitté (pas le ``ThresholdClassifier``).
    X_test : np.ndarray, shape (n_samples, n_features)
        Jeu de test **après standardisation** (comme fourni par
        ``preprocess_for_ml``).
    X_background : np.ndarray
        Données d'arrière-plan (typiquement X_train).
    feature_names : Optional[List[str]]
        Noms des colonnes (par défaut ``FEATURES`` de config.py).
    kernel_background_k : int
        Transmis à ``get_explainer``.

    Returns
    -------
    dict avec les clés :
        ``"shap_values"``   : np.ndarray (n × p)
        ``"mean_abs_shap"`` : dict {feature: float}
        ``"feature_names"`` : list[str]
        ``"n_samples"``     : int
        ``"fit_time_s"``    : float
        ``"explainer_type"``: str
    """
    if not _SHAP_AVAILABLE:
        raise ImportError("Le package 'shap' est requis.")

    names = feature_names or FEATURES

    t0 = time.perf_counter()
    explainer = get_explainer(estimator, X_background, kernel_background_k=kernel_background_k)
    explainer_type = type(explainer).__name__

    shap_values = explainer.shap_values(X_test)

    # TreeExplainer (shap < 0.45) renvoie une liste [shap_class0, shap_class1].
    # TreeExplainer (shap ≥ 0.45) renvoie directement un ndarray de shape
    # (n, p, n_classes) pour la classification multi-classes — on prend la
    # dernière classe (classe positive = Non potable, index 1 en binaire).
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    elapsed = time.perf_counter() - t0

    mean_abs = {
        name: float(np.abs(shap_values[:, i]).mean())
        for i, name in enumerate(names)
    }

    logger.info(
        "SHAP calculé en %.1f s (%s, %d échantillons). "
        "Feature la plus importante : %s (%.4f)",
        elapsed,
        explainer_type,
        len(X_test),
        max(mean_abs, key=mean_abs.get),
        max(mean_abs.values()),
    )

    return {
        "shap_values":    shap_values,
        "mean_abs_shap":  mean_abs,
        "feature_names":  names,
        "n_samples":      int(len(X_test)),
        "fit_time_s":     round(elapsed, 3),
        "explainer_type": explainer_type,
    }


# ===========================================================================
# SÉRIALISATION — RÉSUMÉ JSON LÉGER
# ===========================================================================

def save_shap_summary(
    shap_result: Optional[dict],
    model_name: str,
    model_rank: int = 1,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Sauvegarde un résumé SHAP ultra-léger en JSON, embarquable sur le
    Raspberry Pi.

    Seuls les scalaires sont sérialisés (pas la matrice ``shap_values``
    entière), ce qui maintient la taille du fichier sous 1 Ko.

    Parameters
    ----------
    shap_result :
        Sortie de :func:`compute_global_shap`.
    model_name : str
        Clé du modèle (ex. ``"rf"``, ``"logreg"``).
    model_rank : int
        Rang du modèle dans le classement (1 = meilleur).
    output_path : Optional[Path]
        Chemin de sortie. Si ``None``, utilise
        ``outputs/reports/shap_summary.json``.

    Returns
    -------
    Path
        Chemin du fichier JSON créé.
    """
    path = Path(output_path) if output_path else _DEFAULT_SUMMARY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    # Classement décroissant des features par importance
    ranked = sorted(
        shap_result["mean_abs_shap"].items(),
        key=lambda kv: kv[1],
        reverse=True,
    )

    summary = {
        "model_key":      model_name,
        "model_rank":     model_rank,
        "explainer_type": shap_result["explainer_type"],
        "n_samples":      shap_result["n_samples"],
        "fit_time_s":     shap_result["fit_time_s"],
        # Ranking des features (index 0 = plus importante)
        "feature_ranking": [name for name, _ in ranked],
        # Importances brutes (arrondies à 6 décimales)
        "mean_abs_shap":  {name: round(v, 6) for name, v in ranked},
    }

    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Résumé SHAP sauvegardé → %s", path)
    return path


def load_shap_summary(path: Optional[Path] = None) -> Optional[dict]:
    """
    Charge le résumé SHAP depuis le fichier JSON.

    **Fonction principale côté Raspberry Pi.**  Elle est la seule à
    appeler en production embarquée : aucun calcul SHAP, aucune
    dépendance à ``shap`` ou ``matplotlib``.

    Parameters
    ----------
    path : Optional[Path]
        Chemin du JSON. Si ``None``, utilise le chemin par défaut
        ``outputs/reports/shap_summary.json``.

    Returns
    -------
    dict
        Résumé chargé, avec les clés ``feature_ranking``,
        ``mean_abs_shap``, ``model_key``, ``explainer_type``, etc.

    Raises
    ------
    FileNotFoundError
        Si le fichier JSON n'existe pas.
    """
    p = Path(path) if path else _DEFAULT_SUMMARY_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Résumé SHAP introuvable : {p}\n"
            "Générez-le d'abord sur PC avec : "
            "python -m src.models_training.explainability"
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    logger.debug("Résumé SHAP chargé depuis %s", p)
    return data


# ===========================================================================
# VISUALISATIONS (PC uniquement)
# ===========================================================================

def plot_shap_bar(
    shap_result: Optional[dict],
    model_name: str,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Bar plot de l'importance globale moyenne |SHAP| (une barre par feature).

    Parameters
    ----------
    shap_result :
        Sortie de :func:`compute_global_shap`.
    model_name : str
        Nom lisible du modèle (utilisé dans le titre).
    output_path : Optional[Path]
        Chemin de sauvegarde de la figure PNG.
        Si ``None``, utilise ``outputs/figures/evaluation/shap_bar.png``.

    Returns
    -------
    Optional[Path]
        Chemin du fichier PNG créé, ou ``None`` si matplotlib n'est pas
        disponible.
    """
    if not _MPL_AVAILABLE:
        logger.warning("matplotlib non disponible — figure SHAP bar ignorée.")
        return None

    path = Path(output_path) if output_path else (
        PATHS["figures_eval"] / "shap_bar.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    names = shap_result["feature_names"]
    importances = [shap_result["mean_abs_shap"][n] for n in names]
    ranked_idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    palette = PLOT.get("palette", ["#2E86AB"] * len(names))
    ax.barh(
        [names[i] for i in ranked_idx],
        [importances[i] for i in ranked_idx],
        color=[palette[j % len(palette)] for j, _ in enumerate(ranked_idx)],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Importance SHAP moyenne |φ|")
    ax.set_title(f"Importances SHAP globales — {model_name}")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=PLOT.get("dpi", 150))
    plt.close(fig)
    logger.info("Bar plot SHAP sauvegardé → %s", path)
    return path


def plot_shap_beeswarm(
    shap_result: Optional[dict],
    model_name: str,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Beeswarm plot SHAP (distribution des SHAP values par feature).

    Chaque point = un échantillon de test. La couleur encode la valeur
    de la feature (rouge = élevée, bleu = faible), l'axe X la valeur SHAP
    (impact positif = pousse vers Non potable).

    Parameters
    ----------
    shap_result :
        Sortie de :func:`compute_global_shap`.
    model_name : str
        Nom lisible du modèle (utilisé dans le titre).
    output_path : Optional[Path]
        Chemin de sauvegarde de la figure PNG.

    Returns
    -------
    Optional[Path]
    """
    if not _SHAP_AVAILABLE:
        logger.warning("shap non disponible — beeswarm ignoré.")
        return None
    if not _MPL_AVAILABLE:
        logger.warning("matplotlib non disponible — beeswarm ignoré.")
        return None

    path = Path(output_path) if output_path else (
        PATHS["figures_eval"] / "shap_beeswarm.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    shap_values = shap_result["shap_values"]
    names = shap_result["feature_names"]

    # Trier par importance décroissante pour que les features les plus
    # importantes apparaissent en haut du graphe
    order = np.argsort([shap_result["mean_abs_shap"][n] for n in names])[::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 1.2)))
    _shap.summary_plot(
        shap_values[:, order],
        feature_names=[names[i] for i in order],
        plot_type="dot",
        show=False,
        plot_size=None,
    )
    plt.title(f"Beeswarm SHAP — {model_name}", pad=10)
    plt.tight_layout()
    fig.savefig(path, dpi=PLOT.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    logger.info("Beeswarm SHAP sauvegardé → %s", path)
    return path


# ===========================================================================
# PIPELINE PRINCIPAL (PC)
# ===========================================================================

def main(
    model_path: Optional[Path] = None,
    data_path: Optional[Path] = None,
    summary_output: Optional[Path] = None,
) -> Optional[dict]:
    """
    Pipeline complet d'explicabilité SHAP — à exécuter sur PC.

    Étapes
    ------
    1. Charger le modèle rank-1 (``model_1_*.joblib``) et le scaler.
    2. Charger et prétraiter les données.
    3. Sélectionner l'explainer adapté au type de modèle.
    4. Calculer les SHAP values globales sur le test set.
    5. Sauvegarder le résumé JSON léger (embarquable sur Pi).
    6. Générer les figures (bar plot + beeswarm).

    Parameters
    ----------
    model_path : Optional[Path]
        Chemin du fichier ``.joblib`` du modèle à expliquer.
        Si ``None``, cherche ``outputs/models/model_1_*.joblib``
        (le rank-1 sauvegardé par train_model.py).
    data_path : Optional[Path]
        Chemin du CSV traité (sortie de data_processing.py).
        Si ``None``, utilise ``data/processed/water_potability_processed.csv``.
    summary_output : Optional[Path]
        Chemin de sortie du JSON. Si ``None``, chemin par défaut.

    Returns
    -------
    dict avec les clés :
        ``"summary_path"``  : Path du JSON généré
        ``"shap_result"``   : sortie brute de compute_global_shap
        ``"model_name"``    : str
        ``"bar_path"``      : Path de la figure bar (ou None)
        ``"beeswarm_path"`` : Path de la figure beeswarm (ou None)
    """
    if not _SHAP_AVAILABLE:
        raise ImportError(
            "Le package 'shap' est requis pour générer le résumé SHAP. "
            "Installez-le sur PC avec : pip install shap"
        )

    # ------------------------------------------------------------------
    # 1. Trouver et charger le modèle rank-1 + scaler
    # ------------------------------------------------------------------
    if model_path is None:
        models_dir = PATHS["models"]
        candidates = sorted(models_dir.glob("model_1_*.joblib"))
        if not candidates:
            raise FileNotFoundError(
                f"Aucun modèle rank-1 trouvé dans {models_dir}. "
                "Lancez d'abord train_model.py."
            )
        model_path = candidates[0]

    logger.info("Chargement du modèle : %s", model_path)
    threshold_clf = joblib.load(model_path)

    # Extraire le nom du modèle depuis le nom de fichier (ex. model_1_rf.joblib → rf)
    model_name = model_path.stem.split("_", 2)[-1]  # "rf", "logreg", etc.

    scaler_path = PATHS["models"] / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    if scaler is None:
        logger.warning("Scaler non trouvé — les données ne seront pas standardisées.")

    # ------------------------------------------------------------------
    # 2. Charger et prétraiter les données
    # ------------------------------------------------------------------
    from data_processing import preprocess_for_ml, raw_data_processing

    try:
        from config import DATA
        default_data_path = DATA["raw_path"]
    except (ImportError, KeyError):
        default_data_path = Path("data/processed/water_potability_processed.csv")

    dp = Path(data_path) if data_path else default_data_path
    X, y = raw_data_processing(dp, return_X_y=True)

    X_train, X_test, y_train, y_test, fitted_scaler = preprocess_for_ml(
        X, y, scaler=scaler,
    )

    # ------------------------------------------------------------------
    # 3–4. Calcul SHAP
    # ------------------------------------------------------------------
    estimator = threshold_clf.estimator
    shap_result = compute_global_shap(
        estimator, X_test, X_background=X_train,
    )

    # ------------------------------------------------------------------
    # 5. Sauvegarder le résumé JSON léger
    # ------------------------------------------------------------------
    summary_path = save_shap_summary(
        shap_result,
        model_name=model_name,
        model_rank=1,
        output_path=summary_output,
    )

    # ------------------------------------------------------------------
    # 6. Figures
    # ------------------------------------------------------------------
    bar_path = plot_shap_bar(shap_result, model_name)
    beeswarm_path = plot_shap_beeswarm(shap_result, model_name)

    return {
        "summary_path":   summary_path,
        "shap_result":    shap_result,
        "model_name":     model_name,
        "bar_path":       bar_path,
        "beeswarm_path":  beeswarm_path,
    }


# ===========================================================================
# POINT D'ENTRÉE
# ===========================================================================

if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Génère le résumé SHAP du meilleur modèle.")
    parser.add_argument("--model",   type=Path, default=None, help="Chemin du .joblib rank-1")
    parser.add_argument("--data",    type=Path, default=None, help="Chemin du CSV traité")
    parser.add_argument("--output",  type=Path, default=None, help="Chemin du JSON de sortie")
    args = parser.parse_args()

    result = main(model_path=args.model, data_path=args.data, summary_output=args.output)
    print(f"\n✓ Résumé SHAP : {result['summary_path']}")
    print(f"  Bar plot    : {result['bar_path']}")
    print(f"  Beeswarm    : {result['beeswarm_path']}")