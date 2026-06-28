"""
sensor_inference.py
====================
Pipeline temps réel : lecture des capteurs → diagnostic immédiat
(potable / non potable) sur Raspberry Pi.

Flux
-----
    ADS1115 (I2C)
        ├── A1 → TDS       → Solids [mg/L]
        ├── A2 → pH        → ph [0–14]
        └── A3 → Turbidité → Turbidity [NTU]

    DS18B20 (1-Wire, GPIO 4)
        └── Temperature [°C]  → utilisée par le modèle de prédiction uniquement

    Conductivité = Solids / TDS_EC_FACTOR  (dérivée du TDS)

Rôle de chaque mesure
----------------------
    Diagnostic potabilité : ph, Solids, Conductivity, Turbidity
    Prédiction 24h        : ph, Solids, Conductivity, Turbidity, Temperature

Mode mock
----------
Si les bibliothèques Adafruit sont absentes (PC de développement),
le module bascule automatiquement sur un simulateur de capteurs.

Usage
------
    python src/sensor_inference.py
"""

from __future__ import annotations

import logging
import os
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
# Imports projet — légers, sans dépendances d'entraînement
# ---------------------------------------------------------------------------

# ThresholdClassifier doit être importé au niveau MODULE (pas dans une fonction)
# pour que pickle puisse le retrouver dans __main__ lors du joblib.load().
from threshold_classifier import ThresholdClassifier  # noqa: F401

# Constantes Pi-compatibles : on tente config.py (complet), sinon valeurs locales
try:
    from config import DEFAULT_THRESHOLD, FEATURES, PATHS, PHYSICAL_BOUNDS, TDS_EC_FACTOR
except ImportError:
    logger.warning("config.py indisponible — utilisation des constantes embarquées.")
    DEFAULT_THRESHOLD = 0.5
    FEATURES = ["ph", "Solids", "Conductivity", "Turbidity"]
    PATHS = {"models": Path("outputs/models")}
    PHYSICAL_BOUNDS = {
        "ph":           (0.0,  14.0),
        "Solids":       (0.0,  10000.0),
        "Conductivity": (0.0,  15000.0),
        "Turbidity":    (0.0,  1000.0),
        "Temperature":  (-10.0, 100.0),
    }
    TDS_EC_FACTOR = 0.67

# Nombre d'échantillons pour la moyenne — harmonisé à 5 pour tous les capteurs
N_SAMPLES = 5

# ===========================================================================
# CAPTEUR DS18B20 — TEMPÉRATURE (1-Wire, GPIO 4)
# ===========================================================================

def _init_ds18b20() -> str | None:
    """
    Active les modules 1-Wire et retourne le chemin du fichier capteur.
    Retourne None si aucun DS18B20 n'est détecté (mode PC/mock).
    """
    os.system("modprobe w1-gpio")
    os.system("modprobe w1-therm")
    base_dir = "/sys/bus/w1/devices/"
    try:
        folders = [d for d in os.listdir(base_dir) if d.startswith("28-")]
        device_file = base_dir + folders[0] + "/w1_slave"
        logger.info("DS18B20 détecté : %s", device_file)
        return device_file
    except (IndexError, FileNotFoundError):
        logger.warning("DS18B20 absent — température simulée en mode mock.")
        return None


_DS18B20_FILE: str | None = _init_ds18b20() if _HW_AVAILABLE else None


def read_temperature_once(device_file: str) -> float:
    """
    Lit une mesure de température depuis le fichier 1-Wire du DS18B20.
    Attend la confirmation 'YES' avant de décoder la valeur.
    """
    def _raw_lines() -> list[str]:
        with open(device_file, "r") as f:
            return f.readlines()

    lines = _raw_lines()
    while lines[0].strip()[-3:] != "YES":
        time.sleep(0.2)
        lines = _raw_lines()

    equals_pos = lines[1].find("t=")
    if equals_pos == -1:
        raise ValueError("Format inattendu du fichier DS18B20.")
    return float(lines[1][equals_pos + 2:]) / 1000.0


def read_temperature_mean(device_file: str, n: int = N_SAMPLES) -> float:
    """
    Moyenne de n lectures DS18B20 pour réduire le bruit.
    Le DS18B20 a une résolution de 0.0625°C — 50 lectures donnent
    une stabilité suffisante pour le modèle de prédiction.
    """
    readings = [read_temperature_once(device_file) for _ in range(n)]
    return round(float(np.mean(readings)), 3)


# ===========================================================================
# CONVERSIONS CAPTEURS → UNITÉS MODÈLE
# ===========================================================================

def voltage_to_conductivity(v: float) -> float:
    """Tension A0 → Conductivité en µS/cm. Capteur Gravity TDS Meter V1.0.

    Le polynôme cubique donne directement la conductivité (EC) :
        EC = 133.42·V³ − 255.86·V² + 857.39·V

    Source : courbe de calibration DFRobot Gravity TDS Meter V1.0.
    """
    ec = 133.42 * v**3 - 255.86 * v**2 + 857.39 * v
    return max(0.0, ec)


def conductivity_to_tds(ec: float) -> float:
    """Conductivité [µS/cm] → TDS [mg/L] via TDS_EC_FACTOR.
        TDS = EC × TDS_EC_FACTOR  (0.67 par défaut)
    """
    return ec * TDS_EC_FACTOR


def voltage_to_ph(v: float, temperature: float = 25.0) -> float:
    """Tension A1 → pH [0–14]. Capteur pH Meter V1.1.

    Formule fabricant : pH = 3.5 × V + 0.5
    Compensation thermique via DS18B20 : ±0.03 pH/°C par rapport à 25°C.

    Parameters
    ----------
    v : float
        Tension lue sur A1 [V].
    temperature : float
        Température de l'eau [°C], fournie par le DS18B20.
    """
    ph_raw = 3.5 * v + 0.5
    ph_compensated = ph_raw + (temperature - 25.0) * 0.03
    return float(np.clip(ph_compensated, 0.0, 14.0))


def voltage_to_turbidity(v: float) -> float:
    """Tension A3 → Turbidité en NTU. Capteur TSW-20M.

    Polynôme quadratique calibré sur 3 points réels mesurés :
        4.196 V → 0.1  NTU  (eau minérale)
        3.764 V → 1.5  NTU  (eau du robinet)
        3.213 V → 100  NTU  (eau troublée)

    Relation inversée : tension basse = eau trouble.
    Domaine valide : V ∈ [2.5, 4.196]
    """
    if v >= 4.196:
        return 0.0
    if v < 2.5:
        return 4550.0
    ntu = 178.5607 * v**2 - 1424.5837 * v + 2833.8397
    return round(float(max(0.0, ntu)), 2)





# ===========================================================================
# LECTEURS DE CAPTEURS
# ===========================================================================

class ADS1115Reader:
    """
    Lecture réelle via ADS1115 (I2C) + DS18B20 (1-Wire).

    Canaux ADS1115 : A0=TDS · A1=pH · A3=Turbidité
    Capteur 1-Wire : DS18B20 → Temperature
    """

    def __init__(self, gain: int = 1, n_samples: int = N_SAMPLES):
        if not _HW_AVAILABLE:
            raise RuntimeError("Bibliothèques Adafruit absentes.")
        i2c = busio.I2C(board.SCL, board.SDA)
        self._ads = _ADS.ADS1115(i2c)
        self._ads.gain = gain
        self._n       = n_samples
        self._ch_tds  = AnalogIn(self._ads, 1)   # A1 → TDS
        self._ch_ph   = AnalogIn(self._ads, 2)   # A2 → pH
        self._ch_turb = AnalogIn(self._ads, 3)   # A3 → Turbidité

        if _DS18B20_FILE is None:
            raise RuntimeError("DS18B20 non détecté. Vérifiez le câblage sur GPIO 4.")

    def _mean_voltage(self, channel) -> float:
        readings = [channel.voltage for _ in range(self._n)]
        time.sleep(0.01)
        return float(np.mean(readings))

    def read_features(self) -> dict[str, float]:
        """Retourne les 5 mesures : 4 features diagnostic + température prédiction."""
        v_tds  = self._mean_voltage(self._ch_tds)
        v_ph   = self._mean_voltage(self._ch_ph)
        v_turb = self._mean_voltage(self._ch_turb)
        temp   = read_temperature_mean(_DS18B20_FILE, self._n)
        ec     = voltage_to_conductivity(v_tds)
        return {
            "ph":           voltage_to_ph(v_ph, temperature=temp),
            "Solids":       conductivity_to_tds(ec),
            "Conductivity": ec,
            "Turbidity":    voltage_to_turbidity(v_turb),
            "Temperature":  temp,
        }


class MockSensorReader:
    """Simulateur de capteurs pour développement sur PC."""

    _MEANS = {
        "ph": 7.2, "Solids": 1000.0, "Conductivity": 1500.0,
        "Turbidity": 1.5, "Temperature": 22.0,
    }
    _STD = {
        "ph": 0.5, "Solids": 200.0, "Conductivity": 300.0,
        "Turbidity": 0.5, "Temperature": 1.0,
    }

    def __init__(self, random_state: int | None = None):
        self._rng = np.random.default_rng(random_state)

    def read_features(self) -> dict[str, float]:
        all_features = FEATURES + ["Temperature"]
        values = {}
        for feat in all_features:
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

    Le seuil de décision est porté par le ThresholdClassifier (dans le
    .joblib). ``from_saved_models`` le force à ``DEFAULT_THRESHOLD``
    (config.py) pour éviter qu'un seuil trop agressif issu du tuning
    ne fausse les résultats en production.

    Parameters
    ----------
    model :
        ThresholdClassifier chargé depuis model_1_*.joblib.
    scaler :
        RobustScaler chargé depuis scaler.joblib.
    sensor_reader :
        ADS1115Reader ou MockSensorReader.
    """

    def __init__(self, model, scaler, sensor_reader=None):
        self.model     = model
        self.scaler    = scaler
        self.reader    = sensor_reader or get_sensor_reader()

    @classmethod
    def from_saved_models(
        cls,
        models_dir: Path | None = None,
        mock: bool = False,
    ) -> "SensorPipeline":
        """
        Charge les modèles depuis le disque et instancie le pipeline.

        Le seuil du ThresholdClassifier est forcé à ``DEFAULT_THRESHOLD``
        (config.py) pour la production.

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

        if hasattr(model, "threshold"):
            model.threshold = DEFAULT_THRESHOLD

        logger.info(
            "Modèle chargé : %s | seuil=%.3f",
            candidates[0].name, getattr(model, "threshold", 0.5),
        )
        return cls(model, scaler,
                   sensor_reader=get_sensor_reader(mock=mock))

    def run_once(self) -> dict[str, Any]:
        """
        Lit tous les capteurs et retourne le diagnostic immédiat.

        Returns
        -------
        dict
            timestamp         : str ISO-8601
            raw_values        : dict {feature: float}  — inclut Temperature
            potability_now    : int   (0=Potable, 1=Non potable)
            potability_label  : str
            confidence_proba  : float
            out_of_bounds     : list[str]
            inference_time_ms : float
        """
        import datetime

        t0  = time.perf_counter()
        raw = self.reader.read_features()

        # Vérification des bornes sur toutes les mesures (y compris température)
        all_bounds = {**PHYSICAL_BOUNDS}
        out_of_bounds = [
            f for f, v in raw.items()
            if f in all_bounds
            and not (all_bounds[f][0] <= v <= all_bounds[f][1])
        ]
        if out_of_bounds:
            logger.warning("Valeurs hors bornes : %s", out_of_bounds)

        # Inférence diagnostic — température exclue (réservée au modèle de prédiction)
        x        = np.array([[raw[f] for f in FEATURES]])
        x_scaled = self.scaler.transform(x)
        pred     = int(self.model.predict(x_scaled)[0])
        proba    = float(self.model.predict_proba(x_scaled)[0][1])
        label    = "Non potable" if pred == 1 else "Potable"
        threshold = getattr(self.model, "threshold", 0.5)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = {
            "timestamp":         datetime.datetime.now().isoformat(timespec="seconds"),
            "raw_values":        raw,
            "potability_now":    pred,
            "potability_label":  label,
            "confidence_proba":  round(proba, 4),
            "threshold":         threshold,
            "out_of_bounds":     out_of_bounds,
            "inference_time_ms": round(elapsed_ms, 2),
        }

        logger.info(
            "[%s] pH=%.2f | TDS=%.0f mg/L | Cond=%.1f µS/cm | "
            "Turb=%.2f NTU | Temp=%.1f °C → %s (p=%.2f) | %.1f ms",
            result["timestamp"],
            raw["ph"], raw["Solids"], raw["Conductivity"],
            raw["Turbidity"], raw["Temperature"],
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
            f"(confiance={r['confidence_proba']:.2f}) | "
            f"Temp={r['raw_values']['Temperature']:.1f} °C"
        ),
    )