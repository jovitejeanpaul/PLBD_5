"""
test_data_processing.py
========================
Tests unitaires du module ``src.models_training.data_processing``
(diagnostic immédiat — pipeline de traitement des données de
potabilité de l'eau).

Organisation
------------
Les tests sont regroupés par fonction, dans le même ordre que le
fichier source, pour faciliter la maintenance croisée :

    1. Constantes du module
    2. optimize_memory
    3. load_and_select (+ téléchargement Kaggle simulé)
    4. describe_missing
    5. detect_outliers_iqr
    6. cap_outliers_iqr
    7. impute_ph_by_group
    8. validate_dataframe
    9. raw_data_processing (pipeline complet)
    10. preprocess_for_ml (split + standardisation)

Lancer la suite
----------------
    pytest tests/test_data_processing.py -v
"""

import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.data_processing import (
    ALL_COLS,
    FEATURES,
    KAGGLE_CSV_FILENAME,
    KAGGLE_DATASET_SLUG,
    TRAINING_BOUNDS,
    TARGET,
    TDS_EC_FACTOR,
    cap_outliers_iqr,
    describe_missing,
    detect_outliers_iqr,
    impute_ph_by_group,
    load_and_select,
    optimize_memory,
    preprocess_for_ml,
    raw_data_processing,
    validate_dataframe,
)
import src.data_processing as dp_module

# ===========================================================================
# 1. CONSTANTES DU MODULE
# ===========================================================================

class TestConstants:
    """Vérifie que les constantes structurantes n'ont pas été altérées
    par erreur (régression silencieuse très facile à introduire)."""

    def test_features_and_target(self):
        assert FEATURES == ["ph", "Solids", "Conductivity", "Turbidity"]
        assert TARGET == "Potability"
        assert ALL_COLS == FEATURES + [TARGET]

    def test_physical_bounds_cover_all_features(self):
        # Chaque feature doit avoir une borne physique définie,
        # sans quoi cap_outliers_iqr(enforce_physical=True) l'ignore silencieusement.
        for col in FEATURES:
            assert col in TRAINING_BOUNDS
            low, high = TRAINING_BOUNDS[col]
            assert low < high

    def test_tds_ec_factor_in_oms_range(self):
        # Le facteur OMS standard doit rester dans la plage empirique 0.55-0.75
        assert 0.55 <= TDS_EC_FACTOR <= 0.75

    def test_kaggle_constants_are_strings(self):
        assert isinstance(KAGGLE_DATASET_SLUG, str) and "/" in KAGGLE_DATASET_SLUG
        assert KAGGLE_CSV_FILENAME.endswith(".csv")


# ===========================================================================
# 2. optimize_memory
# ===========================================================================

class TestOptimizeMemory:

    def test_float_columns_downcast_to_float32(self, raw_dataframe_clean):
        out = optimize_memory(raw_dataframe_clean, verbose=False)
        for col in FEATURES:
            assert out[col].dtype == np.float32

    def test_target_downcast_to_int8(self, raw_dataframe_clean):
        out = optimize_memory(raw_dataframe_clean, verbose=False)
        assert out[TARGET].dtype == np.int8

    def test_values_preserved_within_float32_precision(self, raw_dataframe_clean):
        out = optimize_memory(raw_dataframe_clean, verbose=False)
        for col in FEATURES:
            np.testing.assert_allclose(
                out[col].to_numpy(dtype=np.float64),
                raw_dataframe_clean[col].to_numpy(dtype=np.float64),
                rtol=1e-5,
            )
        assert (out[TARGET].to_numpy() == raw_dataframe_clean[TARGET].to_numpy()).all()

    def test_does_not_mutate_input(self, raw_dataframe_clean):
        original_dtype = raw_dataframe_clean["ph"].dtype
        optimize_memory(raw_dataframe_clean, verbose=False)
        # L'appel ne doit pas modifier le DataFrame d'origine (df.copy() en interne)
        assert raw_dataframe_clean["ph"].dtype == original_dtype

    def test_memory_usage_does_not_increase(self, raw_dataframe_clean):
        before = raw_dataframe_clean.memory_usage(deep=True).sum()
        out = optimize_memory(raw_dataframe_clean, verbose=False)
        after = out.memory_usage(deep=True).sum()
        assert after <= before


# ===========================================================================
# 3. load_and_select (+ téléchargement Kaggle simulé)
# ===========================================================================

class TestLoadAndSelect:

    def test_loads_local_file_and_filters_columns(self, raw_csv_path, raw_dataframe_clean):
        # Le CSV contient une colonne supplémentaire non utilisée par le modèle
        df_extra = raw_dataframe_clean.copy()
        df_extra["Hardness"] = 100.0
        df_extra.to_csv(raw_csv_path, index=False)

        out = load_and_select(raw_csv_path, optimize=False)
        assert list(out.columns) == ALL_COLS
        assert "Hardness" not in out.columns
        assert len(out) == len(raw_dataframe_clean)

    def test_missing_required_column_raises_keyerror(self, tmp_path, raw_dataframe_clean):
        df_incomplete = raw_dataframe_clean.drop(columns=["Turbidity"])
        path = tmp_path / "incomplete.csv"
        df_incomplete.to_csv(path, index=False)

        with pytest.raises(KeyError):
            load_and_select(path, optimize=False)

    def test_optimize_false_keeps_original_dtypes(self, raw_csv_path):
        out = load_and_select(raw_csv_path, optimize=False)
        # pandas.read_csv infère du float64 par défaut pour les colonnes décimales
        assert out["ph"].dtype == np.float64

    def test_optimize_true_downcasts(self, raw_csv_path):
        out = load_and_select(raw_csv_path, optimize=True)
        assert out["ph"].dtype == np.float32
        assert out[TARGET].dtype == np.int8

    def test_uses_local_candidate_without_downloading(
        self, tmp_path, raw_dataframe_clean, monkeypatch
    ):
        # Place le CSV au nom Kaggle attendu dans kaggle_download_dir
        local_path = tmp_path / KAGGLE_CSV_FILENAME
        raw_dataframe_clean.to_csv(local_path, index=False)

        def _should_not_be_called(dest_dir):
            raise AssertionError("Le téléchargement Kaggle n'aurait pas dû être déclenché.")

        monkeypatch.setattr(dp_module, "_download_from_kaggle", _should_not_be_called)

        out = load_and_select(filepath=None, optimize=False, kaggle_download_dir=tmp_path)
        assert len(out) == len(raw_dataframe_clean)

    def test_missing_filepath_triggers_download(
        self, tmp_path, raw_dataframe_clean, monkeypatch
    ):
        # Le fichier demandé n'existe pas -> doit déclencher _download_from_kaggle
        requested = tmp_path / "absent.csv"
        fallback_path = tmp_path / "downloaded.csv"
        raw_dataframe_clean.to_csv(fallback_path, index=False)

        calls = []

        def _fake_download(dest_dir):
            calls.append(dest_dir)
            return fallback_path

        monkeypatch.setattr(dp_module, "_download_from_kaggle", _fake_download)

        out = load_and_select(filepath=requested, optimize=False, kaggle_download_dir=tmp_path)
        assert len(calls) == 1
        assert len(out) == len(raw_dataframe_clean)

    def test_none_filepath_without_local_candidate_triggers_download(
        self, tmp_path, raw_dataframe_clean, monkeypatch
    ):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        fallback_path = tmp_path / "downloaded.csv"
        raw_dataframe_clean.to_csv(fallback_path, index=False)

        monkeypatch.setattr(dp_module, "_download_from_kaggle", lambda dest_dir: fallback_path)

        out = load_and_select(filepath=None, optimize=False, kaggle_download_dir=empty_dir)
        assert len(out) == len(raw_dataframe_clean)

    def test_kaggle_not_installed_raises_importerror(
        self, tmp_path, monkeypatch
    ):
        # sys.modules["kaggle"] = None force `import kaggle` à lever ImportError
        monkeypatch.setitem(sys.modules, "kaggle", None)
        empty_dir = tmp_path / "empty2"
        empty_dir.mkdir()

        with pytest.raises(ImportError):
            load_and_select(filepath=None, optimize=False, kaggle_download_dir=empty_dir)


# ===========================================================================
# 4. describe_missing
# ===========================================================================

class TestDownloadFromKaggle:
    """
    Tests de la fonction privée ``_download_from_kaggle``.

    Le module ``kaggle`` n'étant pas installé dans l'environnement de test,
    on injecte un faux module dans ``sys.modules`` exposant ``api.authenticate``
    et ``api.dataset_download_files`` — exactement l'interface utilisée par
    le code source.
    """

    @staticmethod
    def _make_fake_kaggle(monkeypatch, authenticate=None, download=None):
        import types

        fake_api = types.SimpleNamespace(
            authenticate=authenticate or (lambda: None),
            dataset_download_files=download or (lambda *a, **k: None),
        )
        fake_kaggle = types.ModuleType("kaggle")
        fake_kaggle.api = fake_api
        monkeypatch.setitem(sys.modules, "kaggle", fake_kaggle)
        return fake_api

    def test_returns_existing_path_without_calling_api(self, tmp_path, monkeypatch):
        dest_dir = tmp_path / "data"
        dest_dir.mkdir()
        existing = dest_dir / KAGGLE_CSV_FILENAME
        existing.write_text("ph,Solids,Conductivity,Turbidity,Potability\n")

        calls = []
        self._make_fake_kaggle(monkeypatch, authenticate=lambda: calls.append("auth"))

        result = dp_module._download_from_kaggle(dest_dir)
        assert result == existing
        assert calls == []  # l'API Kaggle n'a pas dû être sollicitée

    def test_successful_download_returns_path(self, tmp_path, monkeypatch):
        from pathlib import Path

        dest_dir = tmp_path / "data"

        def _fake_download(dataset, path, unzip, quiet):
            # Simule l'extraction réussie de l'archive Kaggle
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / KAGGLE_CSV_FILENAME).write_text("ph\n7.0\n")

        self._make_fake_kaggle(monkeypatch, download=_fake_download)

        result = dp_module._download_from_kaggle(dest_dir)
        assert result == dest_dir / KAGGLE_CSV_FILENAME
        assert result.exists()

    def test_api_failure_raises_oserror(self, tmp_path, monkeypatch):
        dest_dir = tmp_path / "data"

        def _failing_auth():
            raise Exception("identifiants invalides")

        self._make_fake_kaggle(monkeypatch, authenticate=_failing_auth)

        with pytest.raises(OSError, match="Échec du téléchargement"):
            dp_module._download_from_kaggle(dest_dir)

    def test_missing_csv_after_download_raises_filenotfounderror(self, tmp_path, monkeypatch):
        dest_dir = tmp_path / "data"
        # L'API "réussit" mais n'écrit aucun fichier (archive vide/corrompue)
        self._make_fake_kaggle(monkeypatch)

        with pytest.raises(FileNotFoundError):
            dp_module._download_from_kaggle(dest_dir)


# ===========================================================================
# 4. describe_missing
# ===========================================================================

class TestDescribeMissing:

    def test_reports_only_columns_with_missing_values(self):
        df = pd.DataFrame({
            "A": [1, np.nan, 3, np.nan, 5],   # 2/5 manquants = 40%
            "B": [1, 2, 3, 4, 5],             # 0 manquant
            "C": [np.nan, 2, 3, 4, 5],        # 1/5 manquant = 20%
        })
        report = describe_missing(df)

        assert set(report.index) == {"A", "C"}
        assert "B" not in report.index
        assert report.loc["A", "n_missing"] == 2
        assert report.loc["A", "pct_missing"] == pytest.approx(40.0)
        assert report.loc["C", "n_missing"] == 1
        assert report.loc["C", "pct_missing"] == pytest.approx(20.0)

    def test_sorted_by_pct_missing_descending(self):
        df = pd.DataFrame({
            "A": [1, np.nan, 3, np.nan, 5],   # 40%
            "C": [np.nan, 2, 3, 4, 5],        # 20%
        })
        report = describe_missing(df)
        assert list(report.index) == ["A", "C"]

    def test_no_missing_values_returns_empty_report(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        report = describe_missing(df)
        assert report.empty


# ===========================================================================
# 5. detect_outliers_iqr
# ===========================================================================

class TestDetectOutliersIQR:

    def test_known_iqr_bounds_and_outlier_count(self):
        # 9 valeurs propres + 1 outlier extrême, toutes dans les bornes
        # physiques de Turbidity ([0, 100]) pour isoler le calcul IQR.
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 50]
        df = pd.DataFrame({"Turbidity": values})

        report = detect_outliers_iqr(df, cols=["Turbidity"], factor=1.5)
        row = report.loc["Turbidity"]

        assert row["Q1"] == pytest.approx(3.25)
        assert row["Q3"] == pytest.approx(7.75)
        assert row["IQR"] == pytest.approx(4.5)
        assert row["lower"] == pytest.approx(-3.5)
        assert row["upper"] == pytest.approx(14.5)
        assert row["n_outliers"] == 1
        assert row["pct_outliers"] == pytest.approx(10.0)

    def test_skips_columns_absent_from_dataframe(self):
        df = pd.DataFrame({"ph": [6.5, 7.0, 7.5]})
        report = detect_outliers_iqr(df, cols=["ph", "ColonneInexistante"], factor=1.5)
        assert list(report.index) == ["ph"]

    def test_default_cols_uses_features(self, raw_dataframe_clean):
        report = detect_outliers_iqr(raw_dataframe_clean)
        assert set(report.index) == set(FEATURES)

    def test_no_outliers_when_data_is_uniform(self):
        df = pd.DataFrame({"Turbidity": [5.0] * 20})
        report = detect_outliers_iqr(df, cols=["Turbidity"])
        assert report.loc["Turbidity", "n_outliers"] == 0


# ===========================================================================
# 6. cap_outliers_iqr
# ===========================================================================

class TestCapOutliersIQR:

    def test_caps_value_to_computed_iqr_bound(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 50]
        df = pd.DataFrame({"Turbidity": values})

        out = cap_outliers_iqr(df, cols=["Turbidity"], factor=1.5, enforce_physical=True)

        assert out["Turbidity"].iloc[-1] == pytest.approx(14.5)  # upper bound calculé
        # Les valeurs déjà dans les bornes ne doivent pas être modifiées
        assert (out["Turbidity"].iloc[:-1].to_numpy() == np.array(values[:-1])).all()

    def test_returns_independent_copy(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 50]
        df = pd.DataFrame({"Turbidity": values})
        original = df.copy()

        cap_outliers_iqr(df, cols=["Turbidity"])
        pd.testing.assert_frame_equal(df, original)  # df source intact

    def test_enforce_physical_clips_tighter_than_iqr_bound(self):
        # IQR upper (~18.5) est plus large que la borne physique du ph (14).
        # -> enforce_physical=True doit l'emporter sur l'IQR ; False doit le laisser passer.
        ph_values = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14.5]
        df = pd.DataFrame({"ph": ph_values})

        out_unconstrained = cap_outliers_iqr(df, cols=["ph"], factor=1.5, enforce_physical=False)
        out_constrained = cap_outliers_iqr(df, cols=["ph"], factor=1.5, enforce_physical=True)

        assert out_unconstrained["ph"].iloc[-1] == pytest.approx(14.5)   # bound IQR (~18.5) non atteinte
        assert out_constrained["ph"].iloc[-1] == pytest.approx(14.0)    # borne physique appliquée

    def test_skips_columns_absent_from_dataframe(self):
        df = pd.DataFrame({"ph": [6.5, 7.0, 7.5]})
        out = cap_outliers_iqr(df, cols=["ph", "Inexistant"])
        assert "Inexistant" not in out.columns

    def test_multiple_columns_processed_independently(self, raw_dataframe_dirty):
        out = cap_outliers_iqr(
            raw_dataframe_dirty.drop(columns=["ph"]),  # ph traité séparément (imputation)
            cols=["Solids", "Conductivity", "Turbidity"],
            enforce_physical=True,
        )
        for col in ["Solids", "Conductivity", "Turbidity"]:
            low, high = TRAINING_BOUNDS[col]
            assert out[col].between(low, high).all()


# ===========================================================================
# 7. impute_ph_by_group
# ===========================================================================

class TestImputePhByGroup:

    def test_group_median_imputation(self):
        df = pd.DataFrame({
            "ph":         [6.0, 6.2, 6.4, np.nan, 8.0, 8.4, 8.8, np.nan],
            "Potability": [0,   0,   0,   0,      1,   1,   1,   1],
        })
        out = impute_ph_by_group(df)

        assert out.loc[3, "ph"] == pytest.approx(6.2)  # médiane groupe 0
        assert out.loc[7, "ph"] == pytest.approx(8.4)  # médiane groupe 1
        assert not out["ph"].isna().any()

    def test_fallback_to_global_median_when_target_unknown(self):
        df = pd.DataFrame({
            "ph":         [6.0, 6.2, 6.4, np.nan, 8.0, 8.4, 8.8, np.nan, np.nan],
            "Potability": [0,   0,   0,   0,      1,   1,   1,   1,      np.nan],
        })
        out = impute_ph_by_group(df)

        # Médiane globale calculée après remplissage des groupes connus :
        # [6.0, 6.2, 6.2, 6.4, 8.0, 8.4, 8.4, 8.8] -> médiane = (6.4 + 8.0) / 2 = 7.2
        assert out.loc[8, "ph"] == pytest.approx(7.2)
        assert not out["ph"].isna().any()

    def test_no_target_column_uses_global_median(self):
        df = pd.DataFrame({"ph": [1.0, 2.0, np.nan, 4.0]})
        out = impute_ph_by_group(df)
        assert out.loc[2, "ph"] == pytest.approx(2.0)  # médiane de [1, 2, 4]

    def test_no_missing_values_returns_unchanged(self):
        df = pd.DataFrame({
            "ph":         [6.0, 6.5, 7.0],
            "Potability": [0, 1, 0],
        })
        out = impute_ph_by_group(df)
        assert (out["ph"].to_numpy() == df["ph"].to_numpy()).all()

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({
            "ph":         [6.0, np.nan],
            "Potability": [0, 0],
        })
        impute_ph_by_group(df)
        assert df["ph"].isna().iloc[1]  # l'original garde son NaN


# ===========================================================================
# 8. validate_dataframe
# ===========================================================================

class TestValidateDataframe:

    def test_valid_dataframe_does_not_raise(self, raw_dataframe_clean):
        validate_dataframe(raw_dataframe_clean)  # ne doit lever aucune exception

    def test_raises_on_remaining_missing_values(self, raw_dataframe_clean):
        df = raw_dataframe_clean.copy()
        df.loc[0, "ph"] = np.nan
        with pytest.raises(ValueError, match="manquantes"):
            validate_dataframe(df)

    def test_raises_on_physical_bound_violation(self, raw_dataframe_clean):
        df = raw_dataframe_clean.copy()
        df.loc[0, "ph"] = 15.0  # hors de [0, 14]
        with pytest.raises(ValueError, match="bornes physiques"):
            validate_dataframe(df)


# ===========================================================================
# 9. raw_data_processing — pipeline complet
# ===========================================================================

class TestRawDataProcessing:

    def test_returns_all_columns_without_missing_values(self, raw_csv_path):
        out = raw_data_processing(raw_csv_path)
        assert list(out.columns) == ALL_COLS
        assert not out[ALL_COLS].isnull().any().any()

    def test_relabels_target_correctly(self, raw_csv_path, raw_dataframe_clean):
        out = raw_data_processing(raw_csv_path)
        # Convention Kaggle d'origine : 1 = potable. Après relabélisation : 1 = Non potable.
        expected = 1 - raw_dataframe_clean[TARGET].to_numpy()
        assert (out[TARGET].to_numpy() == expected).all()

    def test_handles_missing_values_and_outliers(self, tmp_path, raw_dataframe_dirty):
        path = tmp_path / "dirty.csv"
        raw_dataframe_dirty.to_csv(path, index=False)

        out = raw_data_processing(path)

        assert not out[ALL_COLS].isnull().any().any()
        for col in FEATURES:
            low, high = TRAINING_BOUNDS[col]
            assert out[col].between(low, high).all()

    def test_return_x_y(self, raw_csv_path):
        X, y = raw_data_processing(raw_csv_path, return_X_y=True)
        assert list(X.columns) == FEATURES
        assert y.name == TARGET
        assert len(X) == len(y)
        assert set(y.unique()).issubset({0, 1})


# ===========================================================================
# 10. preprocess_for_ml — split + standardisation
# ===========================================================================

class TestPreprocessForML:

    @pytest.fixture
    def Xy(self, raw_csv_path):
        return raw_data_processing(raw_csv_path, return_X_y=True)

    def test_split_sizes_and_total(self, Xy):
        X, y = Xy
        X_train, X_test, y_train, y_test, _ = preprocess_for_ml(
            X, y, test_size=0.25, random_state=42,
        )
        assert len(X_train) + len(X_test) == len(X)
        assert abs(len(X_test) - round(0.25 * len(X))) <= 1

    def test_stratification_preserves_class_balance(self, Xy):
        X, y = Xy
        X_train, X_test, y_train, y_test, _ = preprocess_for_ml(
            X, y, test_size=0.25, random_state=42,
        )
        full_ratio = y.mean()
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert train_ratio == pytest.approx(full_ratio, abs=0.05)
        assert test_ratio == pytest.approx(full_ratio, abs=0.08)

    def test_default_scaler_is_robustscaler(self, Xy):
        X, y = Xy
        *_, scaler = preprocess_for_ml(X, y, random_state=42)
        assert isinstance(scaler, RobustScaler)

    def test_custom_scaler_is_used_and_returned(self, Xy):
        X, y = Xy
        custom = StandardScaler()
        *_, scaler = preprocess_for_ml(X, y, random_state=42, scaler=custom)
        assert scaler is custom
        assert hasattr(scaler, "mean_")  # signature StandardScaler, pas RobustScaler

    def test_scaler_fitted_only_on_train_no_leakage(self, Xy):
        from sklearn.model_selection import train_test_split

        X, y = Xy
        X_train, X_test, y_train, y_test, scaler = preprocess_for_ml(
            X, y, test_size=0.25, random_state=42,
        )

        # Reproduction indépendante du même split (même random_state => déterministe)
        X_train_raw, X_test_raw, _, _ = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y,
        )
        expected_center = X_train_raw.median().to_numpy()
        np.testing.assert_allclose(scaler.center_, expected_center, rtol=1e-6)

    def test_outputs_are_numpy_arrays(self, Xy):
        X, y = Xy
        X_train, X_test, y_train, y_test, _ = preprocess_for_ml(X, y, random_state=42)
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)

    def test_reproducible_with_same_random_state(self, Xy):
        X, y = Xy
        out1 = preprocess_for_ml(X, y, random_state=7)
        out2 = preprocess_for_ml(X, y, random_state=7)
        np.testing.assert_array_equal(out1[0], out2[0])  # X_train identique
        np.testing.assert_array_equal(out1[2], out2[2])  # y_train identique
