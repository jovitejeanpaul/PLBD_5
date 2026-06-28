"""
filter_controller.py
=====================
Contrôle des pompes de filtration à partir des résultats du diagnostic.

Architecture physique
----------------------
    Bassine (source) ← capteurs (pH, turbidité, conductivité/TDS, température)
    Pompe 1 → Filtre à sédiments        (particules grossières)
    Pompe 2 → Filtre à charbon compressé (micro-particules fines)
    Pompe 3 → Filtre à charbon actif     (adsorption chimique/organique)

    Chaque pompe est indépendante — on active UN SEUL filtre par cycle
    (celui qui est le plus adapté au problème détecté).

Mapping GPIO (BCM)
-------------------
    Pompe 1 → PIN 23
    Pompe 2 → PIN 24
    Pompe 3 → PIN 25

Logique de décision (par priorité décroissante)
-------------------------------------------------
    P1  Turbidité > 5 NTU               → Sédiments
    P2  pH hors [6.5, 8.5] ou contexte  → Charbon actif
    P3  Turbidité 2–5 NTU               → Charbon compressé
    P4  Non potable sans règle P1–P3    → Charbon actif (défaut) + explication
    --  Conductivité élevée             → Recommandation opérateur (pas de pompe)
    --  Tout conforme                   → Aucune action
"""

from __future__ import annotations

import atexit
import logging
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# ── Détection GPIO ────────────────────────────────────────────────────────────
try:
    import RPi.GPIO as GPIO
    _GPIO_AVAILABLE = True
    logger.info("RPi.GPIO détecté — mode HARDWARE activé.")
except ImportError:
    _GPIO_AVAILABLE = False
    logger.warning("RPi.GPIO absent — mode MOCK activé (GPIO simulé).")

# ── Configuration des filtres ─────────────────────────────────────────────────
FILTER_CONFIG: dict[int, dict] = {
    1: {
        "pin":    23,
        "name":   "Filtre à sédiments",
        "desc":   "Retient particules, sable, matières en suspension",
        "color":  "#2E86AB",
    },
    2: {
        "pin":    24,
        "name":   "Filtre à charbon compressé",
        "desc":   "Micro-filtration fine, certains métaux lourds, kystes parasitaires",
        "color":  "#1C7293",
    },
    3: {
        "pin":    25,
        "name":   "Filtre à charbon actif",
        "desc":   "Adsorption chlore, pesticides, composés organiques, goût/odeur",
        "color":  "#02C39A",
    },
}

DEFAULT_PUMP_DURATION: float = 300.0   # 5 minutes

# Seuils de décision (NM 03.7.001)
THRESHOLDS = {
    "turbidity_high":       5.0,     # NTU — seuil filtre sédiments
    "turbidity_moderate":   2.0,     # NTU — seuil filtre charbon compressé
    "ph_min":               6.5,
    "ph_max":               8.5,
    "conductivity_warning": 1500.0,  # µS/cm — recommandation surveillance
    "conductivity_critical": 2700.0, # µS/cm — NM 03.7.001 limite max
}


# ── Structures de résultat ───────────────────────────────────────────────────

@dataclass
class FilterAction:
    """Résultat d'une activation de pompe."""
    filter_id:   int
    filter_name: str
    pin:         int
    activated:   bool
    duration_s:  float
    reason:      str
    mock:        bool
    timestamp:   str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "filter_id":   self.filter_id,
            "filter_name": self.filter_name,
            "pin":         self.pin,
            "activated":   self.activated,
            "duration_s":  self.duration_s,
            "reason":      self.reason,
            "mock":        self.mock,
            "timestamp":   self.timestamp,
        }


@dataclass
class FilterDecision:
    """Résultat complet de la décision de filtration."""
    filter_to_activate: int | None
    reason:             str
    recommendations:    list[dict]
    potability:         int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "filter_to_activate": self.filter_to_activate,
            "filter_name":        FILTER_CONFIG[self.filter_to_activate]["name"] if self.filter_to_activate else None,
            "filter_color":       FILTER_CONFIG[self.filter_to_activate]["color"] if self.filter_to_activate else None,
            "reason":             self.reason,
            "recommendations":    self.recommendations,
        }


# ════════════════════════════════════════════════════════════════════════════
# CONTRÔLEUR
# ════════════════════════════════════════════════════════════════════════════

class FilterController:
    """
    Contrôleur des pompes de filtration.

    Active UN SEUL filtre par cycle — celui qui est le plus adapté au
    problème détecté. Les recommandations (conductivité, SHAP) sont
    retournées pour affichage mais ne déclenchent pas de pompe.
    """

    def __init__(
        self,
        pump_duration: float = DEFAULT_PUMP_DURATION,
        mock: bool | None = None,
    ):
        self.pump_duration = pump_duration
        self.mock          = (not _GPIO_AVAILABLE) if mock is None else mock
        self._lock         = threading.Lock()
        self._active_pins: set[int] = set()
        self._stop_event:  threading.Event = threading.Event()
        self._pump_thread: threading.Thread | None = None
        self._running_filter: dict | None = None

        if not self.mock:
            self._setup_gpio()
            atexit.register(self.cleanup)
        else:
            logger.info("FilterController initialisé en mode MOCK.")

    # ── GPIO ─────────────────────────────────────────────────────────────────

    def _setup_gpio(self) -> None:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for fid, cfg in FILTER_CONFIG.items():
            GPIO.setup(cfg["pin"], GPIO.OUT)
            GPIO.output(cfg["pin"], GPIO.LOW)
        logger.info("GPIO initialisé — pins %s configurés en sortie.",
                    [c["pin"] for c in FILTER_CONFIG.values()])

    def cleanup(self) -> None:
        """Éteint toutes les pompes et libère les pins GPIO."""
        if not self.mock and _GPIO_AVAILABLE:
            for cfg in FILTER_CONFIG.values():
                try:
                    GPIO.output(cfg["pin"], GPIO.LOW)
                except Exception:
                    pass
            try:
                GPIO.cleanup()
            except Exception:
                pass
            logger.info("GPIO nettoyé — toutes les pompes éteintes.")

    # ── État de la pompe ────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._pump_thread is not None and self._pump_thread.is_alive()

    @property
    def running_status(self) -> dict | None:
        if not self.is_running or self._running_filter is None:
            return None
        elapsed = time.time() - self._running_filter["start_time"]
        return {
            "filter_id":    self._running_filter["filter_id"],
            "filter_name":  self._running_filter["filter_name"],
            "reason":       self._running_filter["reason"],
            "duration_s":   self._running_filter["duration"],
            "elapsed_s":    round(elapsed, 1),
            "remaining_s":  round(max(0, self._running_filter["duration"] - elapsed), 1),
            "mock":         self.mock,
        }

    # ── Activation d'une pompe (non-bloquante) ───────────────────────────────

    def start_pump(
        self,
        filter_id: int,
        reason:    str = "",
        duration:  float | None = None,
    ) -> FilterAction:
        """
        Démarre une pompe en arrière-plan. Retourne immédiatement.
        La pompe s'arrête automatiquement après ``duration`` secondes
        ou manuellement via ``stop_pump()``.
        """
        if filter_id not in FILTER_CONFIG:
            raise ValueError(f"filter_id doit être 1, 2 ou 3 — reçu : {filter_id}")

        if self.is_running:
            raise RuntimeError(
                f"Pompe {self._running_filter['filter_id']} déjà en cours. "
                "Arrêtez-la d'abord avec stop_pump()."
            )

        cfg  = FILTER_CONFIG[filter_id]
        pin  = cfg["pin"]
        dur  = duration if duration is not None else self.pump_duration
        name = cfg["name"]

        self._stop_event.clear()
        self._running_filter = {
            "filter_id":  filter_id,
            "filter_name": name,
            "pin":        pin,
            "reason":     reason,
            "duration":   dur,
            "start_time": time.time(),
        }

        def _run():
            try:
                logger.info("Pompe %d — %s DÉMARRÉE (pin=%d, durée=%.0fs) [%s]",
                            filter_id, name, pin, dur, reason)

                if not self.mock:
                    with self._lock:
                        GPIO.output(pin, GPIO.HIGH)
                        self._active_pins.add(pin)

                stopped = self._stop_event.wait(timeout=dur)

                if stopped:
                    logger.info("Pompe %d — %s ARRÊTÉE manuellement après %.1fs",
                                filter_id, name, time.time() - self._running_filter["start_time"])
                else:
                    logger.info("Pompe %d — %s TERMINÉE (durée complète %.0fs)",
                                filter_id, name, dur)
            except Exception as e:
                logger.error("Erreur pompe %d : %s", filter_id, e)
            finally:
                if not self.mock and pin in self._active_pins:
                    with self._lock:
                        try:
                            GPIO.output(pin, GPIO.LOW)
                        except Exception:
                            pass
                        self._active_pins.discard(pin)
                self._running_filter = None

        self._pump_thread = threading.Thread(target=_run, daemon=True)
        self._pump_thread.start()

        return FilterAction(
            filter_id   = filter_id,
            filter_name = name,
            pin         = pin,
            activated   = True,
            duration_s  = dur,
            reason      = reason,
            mock        = self.mock,
        )

    def stop_pump(self) -> dict | None:
        """Arrête la pompe en cours. Retourne le statut au moment de l'arrêt."""
        if not self.is_running:
            return None

        status = self.running_status
        self._stop_event.set()
        self._pump_thread.join(timeout=5)
        logger.info("Pompe arrêtée par l'opérateur.")
        return status

    # ── Logique de décision ──────────────────────────────────────────────────

    def decide(
        self,
        diagnostic_result: dict[str, Any],
        source:            str  = "inconnue",
        zone_agricole:     bool = False,
        historique_odeur:  bool = False,
        eau_chloree:       bool = False,
        shap_summary:      dict | None = None,
    ) -> FilterDecision:
        """
        Détermine LE filtre à activer (un seul) et les recommandations.

        Parameters
        ----------
        diagnostic_result : dict
            Sortie de SensorPipeline.run_once().
        source, zone_agricole, historique_odeur, eau_chloree :
            Contexte opérateur (peut être fourni via l'interface web).
        shap_summary : dict | None
            Résumé SHAP (feature_ranking, mean_abs_shap) pour expliquer
            les décisions quand aucune règle à seuil ne se déclenche.

        Returns
        -------
        FilterDecision
        """
        raw        = diagnostic_result.get("raw_values", {})
        potability = diagnostic_result.get("potability_now")

        ph           = raw.get("ph", 7.0)
        turbidity    = raw.get("Turbidity", 0.0)
        conductivity = raw.get("Conductivity", 0.0)

        recommendations: list[dict] = []
        filter_id:  int | None = None
        reason:     str = ""

        # ── Recommandations conductivité (pas de pompe) ──────────────────
        if conductivity > THRESHOLDS["conductivity_critical"]:
            recommendations.append({
                "level":   "critical",
                "message": f"Minéralisation excessive ({conductivity:.0f} µS/cm > {THRESHOLDS['conductivity_critical']:.0f} µS/cm NM)",
                "detail":  "Aucun filtre disponible ne traite la minéralisation. "
                           "Traitement par osmose inverse ou changement de source recommandé.",
            })
        elif conductivity > THRESHOLDS["conductivity_warning"]:
            recommendations.append({
                "level":   "warning",
                "message": f"Minéralisation élevée ({conductivity:.0f} µS/cm)",
                "detail":  "Surveillance renforcée de la source recommandée. "
                           "Risque de dépassement du seuil NM 03.7.001 (2700 µS/cm).",
            })

        # ── Eau potable → pas de filtration ──────────────────────────────
        if potability == 0:
            return FilterDecision(
                filter_to_activate=None,
                reason="Eau potable — aucune filtration nécessaire.",
                recommendations=recommendations,
                potability=potability,
            )

        # ── P1 : Turbidité haute → Sédiments ────────────────────────────
        if turbidity > THRESHOLDS["turbidity_high"]:
            filter_id = 1
            reason = f"Turbidité élevée ({turbidity:.1f} NTU > seuil NM {THRESHOLDS['turbidity_high']:.0f} NTU)"

        # ── P2 : pH hors norme ou contexte chimique → Charbon actif ─────
        elif (not (THRESHOLDS["ph_min"] <= ph <= THRESHOLDS["ph_max"])
              or eau_chloree or zone_agricole or historique_odeur
              or source.lower() in ["réseau", "reseau", "rivière", "riviere", "fleuve"]):

            filter_id = 3
            raisons = []
            if not (THRESHOLDS["ph_min"] <= ph <= THRESHOLDS["ph_max"]):
                raisons.append(f"pH hors norme ({ph:.2f})")
            if eau_chloree:
                raisons.append("eau chlorée")
            if zone_agricole:
                raisons.append("zone agricole (risque pesticides/nitrates)")
            if historique_odeur:
                raisons.append("historique d'odeurs")
            if source.lower() in ["réseau", "reseau", "rivière", "riviere", "fleuve"]:
                raisons.append(f"source '{source}'")
            reason = "Risque chimique/organique : " + ", ".join(raisons)

        # ── P3 : Turbidité modérée → Charbon compressé ──────────────────
        elif turbidity > THRESHOLDS["turbidity_moderate"]:
            filter_id = 2
            reason = f"Turbidité modérée ({turbidity:.1f} NTU) — micro-filtration fine recommandée"

        # ── P4 : Non potable sans règle P1–P3 → Charbon actif (défaut) ──
        else:
            filter_id = 3
            reason = "Eau non potable (combinaison de paramètres) — charbon actif par précaution"

            # Enrichir avec SHAP si disponible
            if shap_summary and "feature_ranking" in shap_summary:
                top_feature = shap_summary["feature_ranking"][0]
                shap_vals = shap_summary.get("mean_abs_shap", {})
                top_val = shap_vals.get(top_feature, 0)
                recommendations.append({
                    "level":   "info",
                    "message": f"Le capteur le plus influent est « {top_feature} » (SHAP = {top_val:.4f})",
                    "detail":  "Aucun paramètre individuel ne dépasse un seuil critique, "
                               "mais la combinaison des mesures est jugée à risque par le modèle. "
                               "Un prélèvement pour analyse en laboratoire est recommandé.",
                })

        return FilterDecision(
            filter_to_activate=filter_id,
            reason=reason,
            recommendations=recommendations,
            potability=potability,
        )

    # ── Pipeline complet : décision + activation ─────────────────────────────

    def decide_and_activate(
        self,
        diagnostic_result: dict[str, Any],
        duration:          float | None = None,
        **context,
    ) -> tuple[FilterDecision, FilterAction | None]:
        """
        Point d'entrée principal : décide puis démarre la pompe si nécessaire.
        Non-bloquant — la pompe tourne en arrière-plan.

        Returns
        -------
        tuple[FilterDecision, FilterAction | None]
        """
        decision = self.decide(diagnostic_result, **context)

        action = None
        if decision.filter_to_activate is not None:
            action = self.start_pump(
                decision.filter_to_activate,
                reason=decision.reason,
                duration=duration,
            )
            logger.info("Filtration démarrée : %s", decision.reason)
        else:
            logger.info("Pas de filtration : %s", decision.reason)

        return decision, action
