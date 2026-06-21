"""
sensor_inference.py
====================
Pipeline temps réel : lecture des capteurs → diagnostic immédiat
(potable / non potable) sur Raspberry Pi.

Flux
-----
    ADS1115 (I2C)
        ├── A0 → TDS       → Solids [mg/L]
        ├── A1 → pH        → ph [0–14]
        └── A3 → Turbidité → Turbidity [NTU]

    Conductivité = Solids / TDS_EC_FACTOR  (dérivée du TDS)

Mode mock
----------
Si les bibliothèques Adafruit sont absentes (PC de développement),
le module bascule automatiquement sur un simulateur de capteurs.

Usage
------
    python src/sensor_inference.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Détection automatique du contexte (Pi vs PC)
# ---------------------------------------------------------------------------
try:
    import board
    import busio
    import adafruit_ads1x15.ads1115 as _ADS
    from adafruit_ads1x15.analog_in import AnalogIn
    _HW_AVAILABLE = True
    logger.info("Bibliothèques Adafruit détectées — mode HARDWARE activé.")
except ImportError:
    _HW_AVAILABLE = False
    logger.warning("Bibliothèques Adafruit absentes — mode MOCK activé.")

# ---------------------------------------------------------------------------
# Imports projet (tous dans src/, imports directs)
# ---------------------------------------------------------------------------
from config import FEATURES, PATHS
from data_processing import PHYSICAL_BOUNDS, TDS_EC_FACTOR

# ThresholdClassifier doit être importé au niveau MODULE (pas dans une fonction)
# pour que pickle puisse le retrouver dans __main__ lors du joblib.load().
from train_model import ThresholdClassifier  # noqa: F401


# ===========================================================================
# CONVERSIONS CAPTEURS → UNITÉS MODÈLE
# ===========================================================================

def voltage_to_tds(v: float) -> float:
    """Tension A0 → TDS en mg/L. Source : tds.py
        TDS = (133.42·V³ − 255.86·V² + 857.39·V) × 0.5
    """
    tds = (133.42 * v**3 - 255.86 * v**2 + 857.39 * v) * 0.5
    return max(0.0, tds)


def voltage_to_ph(v: float) -> float:
    """Tension A1 → pH [0–14]. Source : ph.py
        pH = 3.5 × V
    """
    return float(np.clip(3.5 * v, 0.0, 14.0))


def voltage_to_turbidity(v: float) -> float:
    """Tension A3 → Turbidité en NTU. Source : script turbidité.
        V ≥ 4.05  →  0 NTU
        V < 2.5   →  3000 NTU
        sinon     →  (4.095 − V) × 1935
    """
    if v >= 4.05:
        return 0.0
    if v < 2.5:
        return 3000.0
    return float(max(0.0, (4.095 - v) * 1935.0))


def tds_to_conductivity(tds_mg_l: float) -> float:
    """TDS [mg/L] → Conductivité [µS/cm] via TDS_EC_FACTOR."""
    return tds_mg_l / TDS_EC_FACTOR


# ===========================================================================
# LECTEURS DE CAPTEURS
# ===========================================================================

class ADS1115Reader:
    """Lecture réelle via ADS1115 (I2C). Canaux : A0=TDS · A1=pH · A3=Turbidité"""

    def __init__(self, gain: int = 1, n_samples: int = 5):
        if not _HW_AVAILABLE:
            raise RuntimeError("Bibliothèques Adafruit absentes.")
        i2c = busio.I2C(board.SCL, board.SDA)
        self._ads = _ADS.ADS1115(i2c)
        self._ads.gain = gain
        self._n = n_samples
        self._ch_tds  = AnalogIn(self._ads, 0)   # A0 → TDS
        self._ch_ph   = AnalogIn(self._ads, 1)   # A1 → pH
        self._ch_turb = AnalogIn(self._ads, 3)   # A3 → Turbidité

    def _mean_voltage(self, channel) -> float:
        readings = [channel.voltage for _ in range(self._n)]
        time.sleep(0.01)
        return float(np.mean(readings))

    def read_features(self) -> dict[str, float]:
        """Retourne les 4 features dans les unités du modèle."""
        v_tds  = self._mean_voltage(self._ch_tds)
        v_ph   = self._mean_voltage(self._ch_ph)
        v_turb = self._mean_voltage(self._ch_turb)
        tds = voltage_to_tds(v_tds)
        return {
            "ph":           voltage_to_ph(v_ph),
            "Solids":       tds,
            "Conductivity": tds_to_conductivity(tds),
            "Turbidity":    voltage_to_turbidity(v_turb),
        }


class MockSensorReader:
    """Simulateur de capteurs pour développement sur PC."""

    _MEANS = {"ph": 7.2, "Solids": 18500.0, "Conductivity": 420.0, "Turbidity": 3.8}
    _STD   = {"ph": 0.3, "Solids": 800.0,   "Conductivity": 20.0,  "Turbidity": 0.5}

    def __init__(self, random_state: int | None = None):
        self._rng = np.random.default_rng(random_state)

    def read_features(self) -> dict[str, float]:
        values = {}
        for feat in FEATURES:
            raw = self._rng.normal(self._MEANS[feat], self._STD[feat])
            lo, hi = PHYSICAL_BOUNDS[feat]
            values[feat] = round(float(np.clip(raw, lo, hi)), 4)
        return values


def get_sensor_reader(mock: bool = False, **kwargs):
    """Retourne ADS1115Reader ou MockSensorReader selon le contexte."""
    if mock or not _HW_AVAILABLE:
        return MockSensorReader(**kwargs)
    return ADS1115Reader(**kwargs)


# ===========================================================================
# PIPELINE DE DIAGNOSTIC
# ===========================================================================

class SensorPipeline:
    """
    Pipeline diagnostic : capteurs → scaling → prédiction potabilité.

    Parameters
    ----------
    model :
        ThresholdClassifier chargé depuis model_1_*.joblib.
    scaler :
        RobustScaler chargé depuis scaler.joblib.
    sensor_reader :
        ADS1115Reader ou MockSensorReader.
    threshold : float
        Seuil de décision. Chargé automatiquement depuis best_params.json.
        Présent pour surcharge manuelle uniquement — le ThresholdClassifier
        contient déjà le seuil intégré.
    """

    def __init__(self, model, scaler, sensor_reader=None, threshold: float = 0.5):
        self.model     = model
        self.scaler    = scaler
        self.reader    = sensor_reader or get_sensor_reader()
        self.threshold = threshold

    @classmethod
    def from_saved_models(
        cls,
        models_dir: Path | None = None,
        mock: bool = False,
    ) -> "SensorPipeline":
        """
        Charge les modèles depuis le disque et instancie le pipeline.

        Parameters
        ----------
        models_dir : Path | None
            Répertoire contenant les .joblib. Défaut : outputs/models.
        mock : bool
            Forcer le mode mock (développement PC).
        """
        import joblib

        d = Path(models_dir) if models_dir else PATHS["models"]
        candidates = sorted(d.glob("model_1_*.joblib"))
        if not candidates:
            raise FileNotFoundError(
                f"Aucun modèle rank-1 trouvé dans {d}. "
                "Lancez train_model.py d'abord."
            )
        model  = joblib.load(candidates[0])
        scaler = joblib.load(d / "scaler.joblib")

        # Lire le seuil intégré dans le ThresholdClassifier
        threshold = getattr(model, "threshold", 0.5)
        logger.info(
            "Modèle chargé : %s | seuil=%.3f",
            candidates[0].name, threshold,
        )
        return cls(model, scaler,
                   sensor_reader=get_sensor_reader(mock=mock),
                   threshold=threshold)

    def run_once(self) -> dict[str, Any]:
        """
        Lit les capteurs et retourne le diagnostic immédiat.

        Returns
        -------
        dict
            timestamp         : str ISO-8601
            raw_values        : dict {feature: float}
            potability_now    : int  (0=Potable, 1=Non potable)
            potability_label  : str
            confidence_proba  : float
            out_of_bounds     : list[str]
            inference_time_ms : float
        """
        import datetime

        t0 = time.perf_counter()

        raw = self.reader.read_features()

        out_of_bounds = [
            f for f, v in raw.items()
            if not (PHYSICAL_BOUNDS[f][0] <= v <= PHYSICAL_BOUNDS[f][1])
        ]
        if out_of_bounds:
            logger.warning("Valeurs hors bornes : %s", out_of_bounds)

        x        = np.array([[raw[f] for f in FEATURES]])
        x_scaled = self.scaler.transform(x)
        # ThresholdClassifier.predict() applique déjà le seuil optimal
        pred     = int(self.model.predict(x_scaled)[0])
        proba    = float(self.model.predict_proba(x_scaled)[0][1])
        label    = "Non potable" if pred == 1 else "Potable"

        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = {
            "timestamp":         datetime.datetime.now().isoformat(timespec="seconds"),
            "raw_values":        raw,
            "potability_now":    pred,
            "potability_label":  label,
            "confidence_proba":  round(proba, 4),
            "out_of_bounds":     out_of_bounds,
            "inference_time_ms": round(elapsed_ms, 2),
        }

        logger.info(
            "[%s] pH=%.2f | TDS=%.0f mg/L | Cond=%.1f µS/cm | Turb=%.2f NTU "
            "→ %s (p=%.2f) | %.1f ms",
            result["timestamp"],
            raw["ph"], raw["Solids"], raw["Conductivity"], raw["Turbidity"],
            label, proba, elapsed_ms,
        )
        return result

    def run_loop(
        self,
        interval_s: float = 3600.0,
        max_iterations: int | None = None,
        on_result=None,
    ) -> None:
        """
        Boucle de production : une mesure toutes les interval_s secondes.

        Parameters
        ----------
        interval_s : float
            Intervalle entre deux mesures (défaut : 3600 s = 1 heure).
        max_iterations : int | None
            Nombre max d'itérations (None = infini).
        on_result : callable | None
            Callback on_result(result: dict) après chaque mesure.
        """
        logger.info("Démarrage boucle diagnostic (intervalle=%.0f s).", interval_s)
        iteration = 0
        try:
            while max_iterations is None or iteration < max_iterations:
                result = self.run_once()
                if on_result:
                    try:
                        on_result(result)
                    except Exception as exc:
                        logger.error("Erreur callback : %s", exc)
                iteration += 1
                if max_iterations is None or iteration < max_iterations:
                    time.sleep(interval_s)
        except KeyboardInterrupt:
            logger.info("Boucle interrompue (Ctrl+C).")


# ===========================================================================
# POINT D'ENTRÉE
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    pipeline = SensorPipeline.from_saved_models(mock=not _HW_AVAILABLE)
    pipeline.run_loop(
        interval_s=3600.0,
        on_result=lambda r: print(
            f"[{r['timestamp']}] {r['potability_label']} "
            f"(confiance={r['confidence_proba']:.2f})"
        ),
    )