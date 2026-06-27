"""
test_sensor_inference.py
=========================
Tests unitaires du module ``sensor_inference`` (diagnostic immédiat).

Structure du projet
--------------------
    src/
        sensor_inference.py   ← module testé
        train_model.py        ← ThresholdClassifier
        data_processing.py    ← PHYSICAL_BOUNDS, TDS_EC_FACTOR
        config.py             ← FEATURES, PATHS

Lancer la suite
----------------
    pytest tests/test_sensor_inference.py -v
"""

import sys
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------------
# Résolution du chemin : src/ doit être sur sys.path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Imports du module à tester
# ---------------------------------------------------------------------------
import sensor_inference as si
from sensor_inference import (
    MockSensorReader,
    SensorPipeline,
    get_sensor_reader,
    tds_to_conductivity,
    voltage_to_ph,
    voltage_to_tds,
    voltage_to_turbidity,
)
from threshold_classifier import ThresholdClassifier
from config import FEATURES, PATHS
from data_processing import PHYSICAL_BOUNDS, TDS_EC_FACTOR


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def tmp_outputs(tmp_path, monkeypatch):
    """Redirige PATHS vers un répertoire temporaire."""
    monkeypatch.chdir(tmp_path)
    from config import ensure_dirs
    ensure_dirs()
    return tmp_path


@pytest.fixture
def dummy_model_bundle(tmp_outputs):
    """
    ThresholdClassifier factice (seuil=0.35) + scaler, sauvegardés sur
    disque exactement comme train_model.py le fait.
    """
    rng    = np.random.default_rng(0)
    X      = rng.standard_normal((120, 4)).astype(np.float32)
    y      = rng.integers(0, 2, 120)
    scaler = RobustScaler().fit(X)
    X_s    = scaler.transform(X)
    est    = LogisticRegression(max_iter=500).fit(X_s, y)

    clf = ThresholdClassifier(est, threshold=0.35)
    clf.fit(X_s, y)

    joblib.dump(clf,    PATHS["models"] / "model_1_logreg.joblib")
    joblib.dump(scaler, PATHS["models"] / "scaler.joblib")
    return clf, scaler


@pytest.fixture
def pipeline(dummy_model_bundle):
    """SensorPipeline avec MockSensorReader."""
    clf, scaler = dummy_model_bundle
    return SensorPipeline(clf, scaler,
                          sensor_reader=MockSensorReader(random_state=42),
                          threshold=0.35)


# ===========================================================================
# 1. CONVERSIONS CAPTEURS
# ===========================================================================

class TestVoltageToPh:
    """Formule : pH = 3.5 × V, clampée à [0, 14]."""

    def test_known_value(self):
        assert voltage_to_ph(2.0) == pytest.approx(7.0)

    def test_zero_voltage(self):
        assert voltage_to_ph(0.0) == pytest.approx(0.0)

    def test_max_voltage_clamped_to_14(self):
        assert voltage_to_ph(4.096) == pytest.approx(14.0)

    def test_negative_clamped_to_zero(self):
        assert voltage_to_ph(-1.0) == pytest.approx(0.0)

    def test_linearity(self):
        assert voltage_to_ph(1.0) == pytest.approx(3.5)
        assert voltage_to_ph(3.0) == pytest.approx(10.5)

    def test_returns_float(self):
        assert isinstance(voltage_to_ph(2.0), float)


class TestVoltageToTds:
    """Formule : (133.42·V3 - 255.86·V2 + 857.39·V) * 0.5"""

    def test_zero_voltage_gives_zero(self):
        assert voltage_to_tds(0.0) == pytest.approx(0.0)

    def test_factor_half_applied(self):
        """Regression : facteur 0.5 absent donne valeurs 2x trop grandes."""
        v = 1.5
        with_factor    = (133.42*v**3 - 255.86*v**2 + 857.39*v) * 0.5
        without_factor = (133.42*v**3 - 255.86*v**2 + 857.39*v)
        assert voltage_to_tds(v) == pytest.approx(with_factor,    rel=1e-6)
        assert voltage_to_tds(v) != pytest.approx(without_factor, rel=1e-2)

    def test_known_value_at_2v(self):
        v = 2.0
        expected = (133.42*8 - 255.86*4 + 857.39*2) * 0.5
        assert voltage_to_tds(v) == pytest.approx(expected, rel=1e-5)

    def test_never_negative(self):
        for v in np.linspace(0, 0.1, 10):
            assert voltage_to_tds(v) >= 0.0

    def test_returns_float(self):
        assert isinstance(voltage_to_tds(2.0), float)


class TestVoltageToTurbidity:
    """
    Formule :
        V >= 4.05  ->  0 NTU
        V <  2.5   ->  3000 NTU
        sinon      ->  (4.095 - V) * 1935
    """

    def test_clear_water(self):
        assert voltage_to_turbidity(4.05)  == pytest.approx(0.0)
        assert voltage_to_turbidity(4.096) == pytest.approx(0.0)

    def test_very_turbid_water(self):
        assert voltage_to_turbidity(2.4) == pytest.approx(3000.0)
        assert voltage_to_turbidity(0.0) == pytest.approx(3000.0)

    def test_intermediate_formula(self):
        v = 3.5
        assert voltage_to_turbidity(v) == pytest.approx((4.095 - v) * 1935.0)

    def test_boundary_2_5(self):
        v = 2.5
        assert voltage_to_turbidity(v) == pytest.approx((4.095 - v) * 1935.0)

    def test_inverse_relationship(self):
        assert voltage_to_turbidity(4.0) < voltage_to_turbidity(3.5)
        assert voltage_to_turbidity(3.5) < voltage_to_turbidity(3.0)

    def test_never_negative(self):
        for v in np.linspace(0, 4.096, 50):
            assert voltage_to_turbidity(v) >= 0.0

    def test_returns_float(self):
        assert isinstance(voltage_to_turbidity(3.0), float)


class TestTdsToConductivity:

    def test_uses_tds_ec_factor(self):
        assert tds_to_conductivity(670.0) == pytest.approx(670.0 / TDS_EC_FACTOR)

    def test_zero_tds(self):
        assert tds_to_conductivity(0.0) == pytest.approx(0.0)

    def test_factor_in_oms_range(self):
        assert 0.55 <= TDS_EC_FACTOR <= 0.75

    def test_conductivity_greater_than_tds(self):
        assert tds_to_conductivity(500.0) > 500.0


# ===========================================================================
# 2. MockSensorReader
# ===========================================================================

class TestMockSensorReader:

    def test_returns_all_features(self):
        assert set(MockSensorReader(random_state=0).read_features().keys()) == set(FEATURES)

    def test_values_in_physical_bounds(self):
        reader = MockSensorReader(random_state=7)
        for _ in range(20):
            for feat, val in reader.read_features().items():
                lo, hi = PHYSICAL_BOUNDS[feat]
                assert lo <= val <= hi

    def test_reproducible_with_same_seed(self):
        r1 = MockSensorReader(random_state=42).read_features()
        r2 = MockSensorReader(random_state=42).read_features()
        assert r1 == r2

    def test_different_seeds_give_different_values(self):
        r1 = MockSensorReader(random_state=1).read_features()
        r2 = MockSensorReader(random_state=2).read_features()
        assert r1["ph"] != r2["ph"]

    def test_values_are_floats(self):
        for v in MockSensorReader().read_features().values():
            assert isinstance(v, float)


# ===========================================================================
# 3. get_sensor_reader
# ===========================================================================

class TestGetSensorReader:

    def test_mock_true_returns_mock(self):
        assert isinstance(get_sensor_reader(mock=True), MockSensorReader)

    def test_hw_unavailable_returns_mock(self, monkeypatch):
        monkeypatch.setattr(si, "_HW_AVAILABLE", False)
        assert isinstance(get_sensor_reader(), MockSensorReader)

    def test_accepts_random_state(self):
        assert isinstance(get_sensor_reader(mock=True, random_state=99), MockSensorReader)


# ===========================================================================
# 4. SensorPipeline.run_once
# ===========================================================================

class TestRunOnce:

    def test_result_keys(self, pipeline):
        assert set(pipeline.run_once().keys()) == {
            "timestamp", "raw_values", "potability_now",
            "potability_label", "confidence_proba",
            "out_of_bounds", "inference_time_ms",
        }

    def test_raw_values_contain_all_features(self, pipeline):
        assert set(pipeline.run_once()["raw_values"].keys()) == set(FEATURES)

    def test_potability_now_is_binary(self, pipeline):
        assert pipeline.run_once()["potability_now"] in {0, 1}

    def test_label_matches_class(self, pipeline):
        r = pipeline.run_once()
        expected = "Non potable" if r["potability_now"] == 1 else "Potable"
        assert r["potability_label"] == expected

    def test_confidence_proba_in_range(self, pipeline):
        assert 0.0 <= pipeline.run_once()["confidence_proba"] <= 1.0

    def test_out_of_bounds_is_list(self, pipeline):
        assert isinstance(pipeline.run_once()["out_of_bounds"], list)

    def test_timestamp_is_iso_format(self, pipeline):
        import datetime
        datetime.datetime.fromisoformat(pipeline.run_once()["timestamp"])

    def test_inference_time_ms_positive(self, pipeline):
        assert pipeline.run_once()["inference_time_ms"] >= 0.0

    def test_scaler_applied_before_prediction(self, dummy_model_bundle):
        """Le scaler doit transformer les donnees avant la prediction."""
        clf, scaler = dummy_model_bundle

        class FixedReader:
            def read_features(self):
                return {"ph": 7.0, "Solids": 20000.0,
                        "Conductivity": 420.0, "Turbidity": 4.0}

        p = SensorPipeline(clf, scaler,
                           sensor_reader=FixedReader(), threshold=0.35)
        result = p.run_once()

        # Verification manuelle avec le meme scaler
        x     = np.array([[7.0, 20000.0, 420.0, 4.0]])
        x_s   = scaler.transform(x)
        proba = float(clf.predict_proba(x_s)[0][1])
        assert result["confidence_proba"] == pytest.approx(proba, abs=1e-4)

    def test_out_of_bounds_detected(self, dummy_model_bundle):
        clf, scaler = dummy_model_bundle

        class BadReader:
            def read_features(self):
                return {"ph": 20.0, "Solids": 100.0,
                        "Conductivity": 300.0, "Turbidity": 3.0}

        p = SensorPipeline(clf, scaler, sensor_reader=BadReader())
        assert "ph" in p.run_once()["out_of_bounds"]


# ===========================================================================
# 5. SensorPipeline.run_loop
# ===========================================================================

class TestRunLoop:

    def test_runs_n_iterations(self, pipeline):
        collected = []
        pipeline.run_loop(interval_s=0, max_iterations=3,
                          on_result=collected.append)
        assert len(collected) == 3

    def test_callback_receives_dict(self, pipeline):
        results = []
        pipeline.run_loop(interval_s=0, max_iterations=1,
                          on_result=results.append)
        assert "potability_now" in results[0]

    def test_callback_error_does_not_stop_loop(self, pipeline):
        count = {"n": 0}

        def bad_cb(r):
            count["n"] += 1
            raise RuntimeError("erreur intentionnelle")

        pipeline.run_loop(interval_s=0, max_iterations=3, on_result=bad_cb)
        assert count["n"] == 3


# ===========================================================================
# 6. SensorPipeline.from_saved_models
# ===========================================================================

class TestFromSavedModels:

    def test_loads_model_and_scaler(self, dummy_model_bundle, tmp_outputs):
        p = SensorPipeline.from_saved_models(mock=True)
        assert p.model is not None
        assert p.scaler is not None

    def test_threshold_read_from_threshold_classifier(self, dummy_model_bundle, tmp_outputs):
        """Le seuil doit etre lu depuis ThresholdClassifier.threshold (0.35)."""
        p = SensorPipeline.from_saved_models(mock=True)
        assert p.threshold == pytest.approx(0.35)

    def test_default_threshold_when_no_attribute(self, tmp_outputs):
        """Modele sans attribut threshold -> fallback 0.5."""
        rng    = np.random.default_rng(1)
        X      = rng.standard_normal((60, 4))
        y      = rng.integers(0, 2, 60)
        scaler = RobustScaler().fit(X)
        est    = LogisticRegression(max_iter=500).fit(scaler.transform(X), y)
        joblib.dump(est,    PATHS["models"] / "model_1_logreg.joblib")
        joblib.dump(scaler, PATHS["models"] / "scaler.joblib")
        p = SensorPipeline.from_saved_models(mock=True)
        assert p.threshold == pytest.approx(0.5)

    def test_mock_reader_activated(self, dummy_model_bundle, tmp_outputs):
        p = SensorPipeline.from_saved_models(mock=True)
        assert isinstance(p.reader, MockSensorReader)

    def test_raises_when_no_model(self, tmp_outputs):
        with pytest.raises(FileNotFoundError, match="rank-1"):
            SensorPipeline.from_saved_models(mock=True)

    def test_threshold_used_in_prediction(self, dummy_model_bundle, tmp_outputs):
        """La decision finale doit utiliser le seuil tune, pas 0.5."""
        p      = SensorPipeline.from_saved_models(mock=True)
        result = p.run_once()
        proba  = result["confidence_proba"]
        pred   = result["potability_now"]
        assert pred == (1 if proba >= p.threshold else 0)