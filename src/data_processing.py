"""
data_processing.py
==================
Pipeline de traitement des données pour la prédiction de la potabilité de l'eau.

Dataset source : https://www.kaggle.com/datasets/adityakadiwal/water-potability
Features utilisées : Conductivity, Solids, Turbidity, ph
Variable cible     : Potability (0 = non potable, 1 = potable)

Modules
-------
- optimize_memory       : Réduction de l'empreinte mémoire par downcasting des types.
- load_and_select       : Chargement du CSV et sélection des colonnes pertinentes.
- describe_missing      : Rapport synthétique sur les valeurs manquantes.
- detect_outliers_iqr   : Détection des valeurs aberrantes via la méthode IQR.
- cap_outliers_iqr      : Traitement des outliers par écrêtage (winsorisation).
- impute_ph_by_group    : Imputation de ph par médiane conditionnelle (groupe Potability).
- raw_data_processing   : Orchestration complète du pipeline de traitement brut.
- preprocess_for_ml     : Split stratifié + RobustScaler (prétraitement ML).

Dépendances
-----------
    pip install pandas numpy scikit-learn scipy
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------------
# Configuration du logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Colonnes nécessaires à la modélisation + cible
FEATURES = ["ph", "Solids", "Conductivity", "Turbidity"]
TARGET = "Potability"
ALL_COLS = FEATURES + [TARGET]

# Bornes physiques raisonnables pour chaque feature (domaine eau potable/naturelle)
PHYSICAL_BOUNDS: dict[str, Tuple[float, float]] = {
    "ph":           (0.0,  14.0),
    "Solids":       (0.0,  70_000.0),   # mg/L — TDS eau douce / saumâtre
    "Conductivity": (0.0,  1_500.0),    # µS/cm — eau douce à légèrement minéralisée
    "Turbidity":    (0.0,  100.0),      # NTU
}

# Relation empirique TDS ↔ Conductivity :  TDS (mg/L) ≈ k × EC (µS/cm)
# Le facteur k varie de 0.55 à 0.75 selon la composition ionique ;
# 0.64 est la valeur standard retenue par l'OMS pour l'eau potable.
TDS_EC_FACTOR: float = 0.64


# ===========================================================================
# 1. OPTIMISATION MÉMOIRE
# ===========================================================================

def optimize_memory(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Réduit l'empreinte mémoire d'un DataFrame par downcasting des types numériques.

    Stratégie appliquée :
    - Colonnes entières  → plus petit type int (int8 … int64)
    - Colonnes flottantes → float32 (suffisant pour la précision requise)
    - La colonne cible binaire (0/1) est convertie en int8

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame source.
    verbose : bool, optional
        Si True, affiche le gain mémoire obtenu. Par défaut True.

    Returns
    -------
    pd.DataFrame
        DataFrame avec types optimisés (copie indépendante).

    Examples
    --------
    >>> df_opt = optimize_memory(df)
    Mémoire avant : 2.45 MB  →  après : 0.61 MB  (gain : 75.1 %)
    """
    df = df.copy()
    mem_before = df.memory_usage(deep=True).sum() / 1024 ** 2  # MB

    for col in df.select_dtypes(include=["integer"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in df.select_dtypes(include=["float"]).columns:
        df[col] = df[col].astype(np.float32)

    # Cible binaire → int8
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].astype(np.int8)

    mem_after = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        gain = (1 - mem_after / mem_before) * 100 if mem_before > 0 else 0
        logger.info(
            "Mémoire avant : %.2f MB  →  après : %.2f MB  (gain : %.1f %%)",
            mem_before, mem_after, gain,
        )
    return df


# ===========================================================================
# 2. CHARGEMENT & SÉLECTION DES COLONNES
# ===========================================================================

# Identifiant du dataset sur Kaggle
KAGGLE_DATASET_SLUG: str = "adityakadiwal/water-potability"
KAGGLE_CSV_FILENAME: str = "water_potability.csv"


def _download_from_kaggle(dest_dir: Path) -> Path:
    """
    Télécharge le dataset depuis Kaggle via l'API officielle.

    Nécessite un fichier ``~/.kaggle/kaggle.json`` valide (username + key)
    obtenu sur https://www.kaggle.com/settings → *Create New Token*.

    Parameters
    ----------
    dest_dir : Path
        Répertoire de destination pour le fichier téléchargé.

    Returns
    -------
    Path
        Chemin complet vers le CSV extrait.

    Raises
    ------
    ImportError
        Si le package ``kaggle`` n'est pas installé.
    OSError
        Si l'authentification Kaggle échoue (kaggle.json absent ou invalide).
    """
    try:
        import kaggle  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Le package 'kaggle' est requis pour le téléchargement automatique.\n"
            "Installe-le avec :  pip install kaggle"
        ) from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dest_dir / KAGGLE_CSV_FILENAME

    if csv_path.exists():
        logger.info("Dataset déjà présent localement : %s", csv_path)
        return csv_path

    logger.info("Téléchargement depuis Kaggle : %s → %s", KAGGLE_DATASET_SLUG, dest_dir)
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET_SLUG,
            path=str(dest_dir),
            unzip=True,
            quiet=False,
        )
    except Exception as exc:
        raise OSError(
            f"Échec du téléchargement Kaggle.\n"
            f"Vérifie que ~/.kaggle/kaggle.json existe et est valide.\n"
            f"Détail : {exc}"
        ) from exc

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Le CSV '{KAGGLE_CSV_FILENAME}' est introuvable dans {dest_dir} "
            "après extraction. Vérifie le contenu de l'archive."
        )

    logger.info("Dataset téléchargé avec succès : %s", csv_path)
    return csv_path


def load_and_select(
    filepath: Optional[str | Path] = None,
    optimize: bool = True,
    kaggle_download_dir: str | Path = "./data",
) -> pd.DataFrame:
    """
    Charge le CSV et retourne uniquement les colonnes nécessaires.

    Deux modes de fonctionnement :

    1. **Fichier local** — si ``filepath`` pointe vers un CSV existant,
       celui-ci est chargé directement.
    2. **Téléchargement automatique** — si ``filepath`` est ``None`` ou
       inexistant, le dataset est téléchargé depuis Kaggle via l'API
       officielle (nécessite ``~/.kaggle/kaggle.json``).

    Parameters
    ----------
    filepath : str | Path | None, optional
        Chemin vers ``water_potability.csv``. Si ``None``, déclenche le
        téléchargement automatique depuis Kaggle.
    optimize : bool, optional
        Si True, applique :func:`optimize_memory` après le chargement.
        Par défaut True.
    kaggle_download_dir : str | Path, optional
        Répertoire de destination utilisé uniquement lors du téléchargement
        automatique. Par défaut ``"./data"``.

    Returns
    -------
    pd.DataFrame
        DataFrame filtré sur ``ALL_COLS`` = FEATURES + TARGET.

    Raises
    ------
    FileNotFoundError
        Si ``filepath`` est fourni mais introuvable, ou si le CSV n'est pas
        présent après téléchargement.
    KeyError
        Si des colonnes attendues sont absentes du CSV.
    ImportError
        Si ``kaggle`` n'est pas installé et qu'un téléchargement est requis.
    OSError
        Si l'authentification Kaggle échoue.

    Examples
    --------
    >>> # Chargement depuis un fichier local
    >>> df = load_and_select("data/water_potability.csv")

    >>> # Téléchargement automatique si absent
    >>> df = load_and_select()

    >>> # Répertoire de téléchargement personnalisé
    >>> df = load_and_select(kaggle_download_dir="./datasets/water")
    """
    # --- Résolution du chemin ---
    if filepath is not None:
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(
                "Fichier '%s' introuvable. Tentative de téléchargement depuis Kaggle…",
                filepath,
            )
            filepath = _download_from_kaggle(Path(kaggle_download_dir))
    else:
        # Vérifier d'abord si le CSV est déjà dans le répertoire cible
        local_candidate = Path(kaggle_download_dir) / KAGGLE_CSV_FILENAME
        if local_candidate.exists():
            logger.info("CSV trouvé dans le répertoire par défaut : %s", local_candidate)
            filepath = local_candidate
        else:
            filepath = _download_from_kaggle(Path(kaggle_download_dir))

    # --- Lecture ---
    df = pd.read_csv(filepath)
    logger.info("Dataset chargé : %d lignes × %d colonnes", *df.shape)

    # --- Validation des colonnes ---
    missing_cols = set(ALL_COLS) - set(df.columns)
    if missing_cols:
        raise KeyError(f"Colonnes manquantes dans le CSV : {missing_cols}")

    df = df[ALL_COLS].copy()
    logger.info("Colonnes sélectionnées : %s", ALL_COLS)

    if optimize:
        df = optimize_memory(df)

    return df


# ===========================================================================
# 3. DIAGNOSTIC DES VALEURS MANQUANTES
# ===========================================================================

def describe_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère un rapport sur les valeurs manquantes par colonne.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à analyser.

    Returns
    -------
    pd.DataFrame
        Tableau avec les colonnes ``n_missing``, ``pct_missing`` et ``dtype``
        pour chaque variable présentant au moins une valeur manquante.

    Examples
    --------
    >>> report = describe_missing(df)
    >>> print(report)
    """
    missing = df.isnull().sum()
    pct = missing / len(df) * 100
    report = (
        pd.DataFrame({"n_missing": missing, "pct_missing": pct.round(2), "dtype": df.dtypes})
        .query("n_missing > 0")
        .sort_values("pct_missing", ascending=False)
    )
    logger.info("Valeurs manquantes :\n%s", report.to_string())
    return report


# ===========================================================================
# 4. DÉTECTION & TRAITEMENT DES VALEURS ABERRANTES
# ===========================================================================

def detect_outliers_iqr(
    df: pd.DataFrame,
    cols: Optional[list[str]] = None,
    factor: float = 1.5,
) -> pd.DataFrame:
    """
    Identifie les outliers via la règle IQR (méthode de Tukey).

    Un point *x* est considéré aberrant si :
        x < Q1 - factor × IQR  ou  x > Q3 + factor × IQR

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame source.
    cols : list[str], optional
        Liste des colonnes à analyser. Par défaut : FEATURES.
    factor : float, optional
        Multiplicateur IQR. 1.5 = outliers modérés, 3.0 = outliers extrêmes.
        Par défaut 1.5.

    Returns
    -------
    pd.DataFrame
        Rapport avec colonnes ``Q1``, ``Q3``, ``IQR``, ``lower``, ``upper``,
        ``n_outliers``, ``pct_outliers`` pour chaque colonne analysée.
    """
    cols = cols or FEATURES
    records = []

    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        n_out = ((s < lower) | (s > upper)).sum()
        records.append({
            "column":     col,
            "Q1":         round(float(q1), 4),
            "Q3":         round(float(q3), 4),
            "IQR":        round(float(iqr), 4),
            "lower":      round(float(lower), 4),
            "upper":      round(float(upper), 4),
            "n_outliers": int(n_out),
            "pct_outliers": round(n_out / len(s) * 100, 2),
        })

    report = pd.DataFrame(records).set_index("column")
    logger.info("Rapport outliers IQR (factor=%.1f) :\n%s", factor, report.to_string())
    return report


def cap_outliers_iqr(
    df: pd.DataFrame,
    cols: Optional[list[str]] = None,
    factor: float = 1.5,
    enforce_physical: bool = True,
) -> pd.DataFrame:
    """
    Écrête (winssorise) les valeurs aberrantes aux bornes IQR.

    Pour chaque colonne, les valeurs inférieures à ``lower`` sont remplacées
    par ``lower`` et celles supérieures à ``upper`` par ``upper``.
    Optionnellement, les bornes physiques de :data:`PHYSICAL_BOUNDS` sont
    également appliquées pour éviter des valeurs physiquement impossibles.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame source.
    cols : list[str], optional
        Colonnes à traiter. Par défaut : FEATURES.
    factor : float, optional
        Multiplicateur IQR. Par défaut 1.5.
    enforce_physical : bool, optional
        Si True, clip également selon :data:`PHYSICAL_BOUNDS`. Par défaut True.

    Returns
    -------
    pd.DataFrame
        DataFrame avec outliers écrêtés (copie indépendante).
    """
    df = df.copy()
    cols = cols or FEATURES

    for col in cols:
        if col not in df.columns:
            continue

        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        # Appliquer les bornes physiques si demandé
        if enforce_physical and col in PHYSICAL_BOUNDS:
            phys_low, phys_high = PHYSICAL_BOUNDS[col]
            lower = max(lower, phys_low)
            upper = min(upper, phys_high)

        before = df[col].copy()
        df[col] = df[col].clip(lower=lower, upper=upper)
        n_capped = (df[col] != before).sum()
        logger.info("  [%s] %d valeurs écrêtées → [%.3f, %.3f]", col, n_capped, lower, upper)

    return df


# ===========================================================================
# 5. IMPUTATION PAR MÉDIANE CONDITIONNELLE
# ===========================================================================

def impute_ph_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute les valeurs manquantes de ``ph`` par médiane conditionnelle au groupe ``Potability``.

    Rationale : le pH des eaux potables est généralement plus resserré (6.5–8.5
    selon l'OMS) que celui des eaux non potables. Utiliser la médiane par classe
    préserve cette distribution bimodale potentielle.

    Si ``Potability`` n'est pas disponible, la médiane globale est utilisée
    comme fallback.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant ``ph`` et éventuellement ``Potability``.

    Returns
    -------
    pd.DataFrame
        DataFrame avec ``ph`` imputée (copie indépendante).
    """
    df = df.copy()
    mask = df["ph"].isna()
    n_missing = mask.sum()

    if n_missing == 0:
        logger.info("ph : aucune valeur manquante.")
        return df

    if TARGET in df.columns and df[TARGET].notna().any():
        # Imputation par médiane de groupe
        group_medians = df.groupby(TARGET)["ph"].median()
        for grp, med in group_medians.items():
            idx = mask & (df[TARGET] == grp)
            df.loc[idx, "ph"] = med
            logger.info("  ph groupe %s : %d valeurs imputées (médiane = %.4f)", grp, idx.sum(), med)

        # Fallback pour les lignes sans Potability connue
        remaining = df["ph"].isna()
        if remaining.any():
            global_median = df["ph"].median()
            df.loc[remaining, "ph"] = global_median
            logger.info("  ph fallback global : %d valeurs imputées (médiane = %.4f)", remaining.sum(), global_median)
    else:
        global_median = df["ph"].median()
        df["ph"] = df["ph"].fillna(global_median)
        logger.info("ph : %d valeurs imputées par médiane globale (%.4f)", n_missing, global_median)

    return df



# ===========================================================================
# 6. VÉRIFICATION FINALE
# ===========================================================================

def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Vérifie l'intégrité du DataFrame après traitement.

    Contrôles effectués :
    - Absence de valeurs manquantes dans les features et la cible.
    - Respect des bornes physiques de :data:`PHYSICAL_BOUNDS`.
    - Distribution de la cible (équilibre des classes).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame traité à valider.

    Raises
    ------
    ValueError
        Si des valeurs manquantes persistent ou si des bornes physiques
        sont violées après traitement.
    """
    # Valeurs manquantes résiduelles
    remaining_na = df[ALL_COLS].isnull().sum()
    if remaining_na.any():
        raise ValueError(
            f"Valeurs manquantes résiduelles après pipeline :\n{remaining_na[remaining_na > 0]}"
        )

    # Bornes physiques
    violations = []
    for col, (low, high) in PHYSICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        out_of_bounds = ((df[col] < low) | (df[col] > high)).sum()
        if out_of_bounds:
            violations.append(f"  {col}: {out_of_bounds} valeurs hors [{low}, {high}]")

    if violations:
        raise ValueError("Violations des bornes physiques :\n" + "\n".join(violations))

    # Distribution de la cible
    target_dist = df[TARGET].value_counts(normalize=True).round(3)
    logger.info("Distribution de %s :\n%s", TARGET, target_dist.to_string())
    logger.info("✓ Validation réussie. Shape final : %s", df.shape)


# ===========================================================================
# 7. PIPELINE PRINCIPAL
# ===========================================================================

def raw_data_processing(
    filepath: str | Path,
    cap_factor: float = 1.5,
    return_X_y: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.Series]:
    """
    Orchestre l'ensemble du pipeline de traitement des données brutes.

    Étapes :
    1. Chargement & sélection des colonnes (+ optimisation mémoire)
    2. Rapport sur les valeurs manquantes
    3. Détection des outliers (rapport uniquement)
    4. Écrêtage des outliers IQR + respect des bornes physiques
    5. Imputation pH par médiane conditionnelle au groupe Potability
    6. Relabélisation : 1 = Non potable, 0 = Potable
    7. Validation finale du DataFrame

    Parameters
    ----------
    filepath : str | Path
        Chemin vers ``water_potability.csv``.
    cap_factor : float, optional
        Facteur IQR pour l'écrêtage. Par défaut 1.5.
    return_X_y : bool, optional
        Si True, retourne le tuple ``(X, y)`` prêt pour scikit-learn.
        Si False (défaut), retourne le DataFrame complet.

    Returns
    -------
    pd.DataFrame
        DataFrame traité, ou tuple ``(X, y)`` si ``return_X_y=True``.

    Examples
    --------
    >>> df = raw_data_processing("water_potability.csv")
    >>> X, y = raw_data_processing("water_potability.csv", return_X_y=True)
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> clf = RandomForestClassifier().fit(X, y)
    """
    logger.info("=" * 60)
    logger.info("DÉBUT DU PIPELINE — Water Potability")
    logger.info("=" * 60)

    # Étape 1 : Chargement
    df = load_and_select(filepath, optimize=True)

    # Étape 2 : Diagnostic NaN
    describe_missing(df)

    # Étape 3 : Rapport outliers (sans modification)
    detect_outliers_iqr(df, factor=cap_factor)

    # Étape 4 : Écrêtage outliers
    logger.info("--- Écrêtage des outliers (IQR factor=%.1f) ---", cap_factor)
    df = cap_outliers_iqr(df, factor=cap_factor, enforce_physical=True)

    # Étape 5 : Imputation pH
    logger.info("--- Imputation pH par groupe Potability ---")
    df = impute_ph_by_group(df)

    # Étape 6 : Relabélisation de la cible
    # Convention retenue : 1 = Non potable (cas dangereux, classe majoritaire)
    #                      0 = Potable
    # Raison : les métriques sklearn (F1, recall, précision) ciblent la classe 1
    # par défaut. En faisant de "Non potable" la classe 1, le modèle est évalué
    # naturellement sur sa capacité à détecter l'eau dangereuse.
    logger.info("--- Relabélisation : 1 = Non potable, 0 = Potable ---")
    df[TARGET] = 1 - df[TARGET]
    dist = df[TARGET].value_counts().to_dict()
    logger.info("  Distribution après relabélisation : %s", dist)

    # Étape 7 : Validation
    logger.info("--- Validation ---")
    validate_dataframe(df)

    logger.info("=" * 60)
    logger.info("PIPELINE TERMINÉ")
    logger.info("=" * 60)

    if return_X_y:
        X = df[FEATURES]
        y = df[TARGET]
        return X, y

    return df


# ===========================================================================
# 8. PRÉTRAITEMENT ML — STANDARDISATION & SPLIT
# ===========================================================================

def preprocess_for_ml(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
    scaler=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    """
    Prépare les données pour l'entraînement ML : split stratifié + standardisation.

    Méthode de standardisation : **RobustScaler**
    ---------------------------------------------
    Le RobustScaler centre les données sur la **médiane** et les met à l'échelle
    selon l'**IQR** (intervalle interquartile Q1-Q3), ce qui le rend insensible
    aux valeurs aberrantes résiduelles — une propriété critique pour ce dataset
    dont les distributions de Conductivity et Solids sont fortement asymétriques.

    Comparaison des scalers :
    - ``StandardScaler`` : sensible aux outliers (utilise moyenne/ecart-type).
    - ``MinMaxScaler``   : tres sensible aux extremes (compresse tout vers [0,1]).
    - ``RobustScaler``   : robuste aux outliers, preserve la structure des donnees.

    Procedure
    ---------
    1. Split stratifie train/test (preserve le ratio des classes dans chaque split).
    2. Fit du scaler **uniquement sur le train set** (previent le data leakage).
    3. Transform applique independamment sur train et test.

    Parameters
    ----------
    X : pd.DataFrame
        Features issues de :func:`run_pipeline`.
    y : pd.Series
        Cible binaire (0/1).
    test_size : float, optional
        Fraction reservee au test final. Par defaut 0.20 (80/20 split).
    random_state : int, optional
        Graine aleatoire pour la reproductibilite. Par defaut 42.
    scaler : sklearn transformer, optional
        Scaler custom a utiliser. Si ``None``, utilise ``RobustScaler()``.
        Permet d'injecter un scaler deja fitte (inference en production).

    Returns
    -------
    X_train : np.ndarray
        Features d'entrainement standardisees.
    X_test : np.ndarray
        Features de test standardisees.
    y_train : np.ndarray
        Labels d'entrainement.
    y_test : np.ndarray
        Labels de test.
    fitted_scaler : sklearn transformer
        Scaler fitte sur le train set (a sauvegarder avec joblib pour
        l'inference en production).

    Examples
    --------
    >>> X, y = run_pipeline("water_potability.csv", return_X_y=True)
    >>> X_train, X_test, y_train, y_test, scaler = preprocess_for_ml(X, y)
    >>> import joblib
    >>> joblib.dump(scaler, "outputs/models/scaler.joblib")
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import RobustScaler as _RobustScaler

    # --- Split stratifie ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,          # preserve le ratio potable/non-potable dans chaque split
    )

    logger.info(
        "Split train/test : %d / %d  (stratifie, test_size=%.0f%%)",
        len(X_train), len(X_test), test_size * 100,
    )

    # Distribution des classes apres split
    for name, subset in [("Train", y_train), ("Test", y_test)]:
        dist = subset.value_counts(normalize=True).round(3).to_dict()
        logger.info("  %s -- distribution Potability : %s", name, dist)

    # --- Standardisation RobustScaler ---
    fitted_scaler = scaler if scaler is not None else _RobustScaler()

    # IMPORTANT : fit uniquement sur le train pour eviter tout data leakage
    X_train_scaled = fitted_scaler.fit_transform(X_train)
    X_test_scaled  = fitted_scaler.transform(X_test)

    logger.info(
        "RobustScaler fitte sur le train set. "
        "Centre (mediane) : %s | Echelle (IQR) : %s",
        np.round(fitted_scaler.center_, 3),
        np.round(fitted_scaler.scale_,  3),
    )

    return (
        X_train_scaled,
        X_test_scaled,
        y_train.to_numpy(),
        y_test.to_numpy(),
        fitted_scaler,
    )

# ===========================================================================
# Point d'entrée (usage direct)
# ===========================================================================

if __name__ == "__main__":

    path ="data/raw/water_potability.csv"

    X, y= raw_data_processing(path, return_X_y=True)

    print("\n--- Aperçu des features (5 premières lignes) ---")
    print(X.head().to_string())
    print(f"\nShape X : {X.shape}  |  Shape y : {y.shape}")
    print(f"Classes y : {dict(y.value_counts().sort_index())}")

    X_train, X_test, y_train, y_test, scaler = preprocess_for_ml(X, y)
    print(f"\n--- Prétraitement ML ---")
    print(f"X_train : {X_train.shape} | X_test : {X_test.shape}")
    print(f"Scaler  : {scaler}")
