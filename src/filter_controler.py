"""
filter_controller.py
=====================
Contrôle automatique des pompes de filtration à partir des résultats
du diagnostic immédiat.

Mapping GPIO (BCM)
-------------------
    Pompe 1 → PIN 23 → Filtre à sédiments       (turbidité élevée)
    Pompe 2 → PIN 24 → Filtre à charbon compressé (minéralisation élevée)
    Pompe 3 → PIN 25 → Filtre à charbon actif    (risque chimique/organique)

Mode mock
----------
Si le module RPi.GPIO n'est pas disponible (PC de développement),
toutes les activations sont simulées — aucun import de GPIO requis.

Usage
------
    from filter_controller import FilterController

    ctrl = FilterController()
    actions = ctrl.decide_and_activate(diagnostic_result)
    # actions = liste des filtres activés avec durée et raison
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
        "desc":   "Élimine métaux lourds, forte minéralisation",
        "color":  "#1C7293",
    },
    3: {
        "pin":    25,
        "name":   "Filtre à charbon actif",
        "desc":   "Adsorption chlore, pesticides, composés organiques",
        "color":  "#02C39A",
    },
}

# Durée par défaut d'activation d'une pompe (secondes)
DEFAULT_PUMP_DURATION: float = 5.0

# Seuils de décision (NM 03.7.001)
THRESHOLDS = {
    "turbidity_high":    5.0,    # NTU — seuil filtre sédiments
    "tds_moderate_low":  500.0,  # mg/L
    "tds_moderate_high": 700.0,  # mg/L — zone charbon compressé
}


# ── Structure de résultat d'activation ───────────────────────────────────────
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


# ════════════════════════════════════════════════════════════════════════════
# CONTRÔLEUR
# ════════════════════════════════════════════════════════════════════════════

class FilterController:
    """
    Contrôleur des pompes de filtration.

    Gère l'initialisation GPIO, l'activation/désactivation des pompes et
    la logique de sélection du filtre adapté aux résultats de diagnostic.

    Parameters
    ----------
    pump_duration : float
        Durée d'activation de chaque pompe en secondes (défaut : 5s).
    mock : bool | None
        Forcer le mode mock. Si None, détecté automatiquement.
    """

    def __init__(
        self,
        pump_duration: float = DEFAULT_PUMP_DURATION,
        mock: bool | None = None,
    ):
        self.pump_duration = pump_duration
        self.mock          = (not _GPIO_AVAILABLE) if mock is None else mock
        self._lock         = threading.Lock()   # évite deux pompes simultanées
        self._active_pins: set[int] = set()

        if not self.mock:
            self._setup_gpio()
            atexit.register(self.cleanup)   # nettoyage propre à la fin du processus
        else:
            logger.info("FilterController initialisé en mode MOCK.")

    # ── GPIO ─────────────────────────────────────────────────────────────────

    def _setup_gpio(self) -> None:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for fid, cfg in FILTER_CONFIG.items():
            GPIO.setup(cfg["pin"], GPIO.OUT)
            GPIO.output(cfg["pin"], GPIO.LOW)   # s'assurer que tout est éteint
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

    # ── Activation d'une pompe ────────────────────────────────────────────────

    def activate_pump(
        self,
        filter_id: int,
        reason:    str = "",
        duration:  float | None = None,
    ) -> FilterAction:
        """
        Active une pompe pendant ``duration`` secondes puis l'éteint.

        L'activation est bloquante (attend la fin de la durée) mais
        thread-safe via un verrou pour éviter les activations simultanées.

        Parameters
        ----------
        filter_id : int
            Identifiant du filtre (1, 2 ou 3).
        reason : str
            Raison de l'activation (pour les logs et l'interface web).
        duration : float | None
            Durée d'activation en secondes. Défaut : self.pump_duration.

        Returns
        -------
        FilterAction
        """
        if filter_id not in FILTER_CONFIG:
            raise ValueError(f"filter_id doit être 1, 2 ou 3 — reçu : {filter_id}")

        cfg      = FILTER_CONFIG[filter_id]
        pin      = cfg["pin"]
        dur      = duration if duration is not None else self.pump_duration
        name     = cfg["name"]
        activated = False

        with self._lock:
            try:
                logger.info("Activation pompe %d — %s (pin=%d, durée=%.1fs) [%s]",
                            filter_id, name, pin, dur, reason)

                if not self.mock:
                    GPIO.output(pin, GPIO.HIGH)
                    self._active_pins.add(pin)

                time.sleep(dur)
                activated = True

            except Exception as e:
                logger.error("Erreur activation pompe %d : %s", filter_id, e)
            finally:
                if not self.mock and pin in self._active_pins:
                    try:
                        GPIO.output(pin, GPIO.LOW)
                    except Exception:
                        pass
                    self._active_pins.discard(pin)

                status = "✓ (MOCK)" if self.mock else ("✓ OK" if activated else "✗ ERREUR")
                logger.info("Pompe %d — %s → arrêtée. %s", filter_id, name, status)

        return FilterAction(
            filter_id   = filter_id,
            filter_name = name,
            pin         = pin,
            activated   = activated if not self.mock else True,
            duration_s  = dur,
            reason      = reason,
            mock        = self.mock,
        )

    # ── Logique de sélection des filtres ─────────────────────────────────────

    def decide_filters(
        self,
        raw_values:     dict[str, float],
        source:         str   = "inconnue",
        zone_agricole:  bool  = False,
        historique_odeur: bool = False,
        eau_chloree:    bool  = False,
    ) -> list[tuple[int, str]]:
        """
        Détermine quels filtres activer à partir des valeurs capteurs.

        Returns
        -------
        list[tuple[int, str]]
            Liste de (filter_id, raison) à activer, dans l'ordre.
        """
        ph           = raw_values.get("ph",           7.0)
        turbidity    = raw_values.get("Turbidity",    0.0)
        tds          = raw_values.get("Solids",       0.0)
        conductivity = raw_values.get("Conductivity", 0.0)

        to_activate: list[tuple[int, str]] = []

        # ── Filtre 1 : Sédiments ─────────────────────────────────────────
        if turbidity > THRESHOLDS["turbidity_high"]:
            to_activate.append((
                1,
                f"Turbidité élevée ({turbidity:.1f} NTU > seuil NM 5 NTU)"
            ))

        # ── Filtre 2 : Charbon compressé ─────────────────────────────────
        if THRESHOLDS["tds_moderate_low"] < tds <= THRESHOLDS["tds_moderate_high"]:
            to_activate.append((
                2,
                f"Minéralisation modérée à élevée (TDS {tds:.0f} mg/L)"
            ))
        elif tds > THRESHOLDS["tds_moderate_high"]:
            to_activate.append((
                2,
                f"Forte minéralisation (TDS {tds:.0f} mg/L > 700 mg/L)"
            ))

        # ── Filtre 3 : Charbon actif ──────────────────────────────────────
        risque_chimique = (
            eau_chloree
            or zone_agricole
            or historique_odeur
            or source.lower() in ["réseau", "reseau", "rivière", "riviere", "fleuve"]
            or not (6.5 <= ph <= 8.5)
        )
        if risque_chimique:
            raisons = []
            if not (6.5 <= ph <= 8.5):
                raisons.append(f"pH hors norme ({ph:.2f})")
            if eau_chloree:
                raisons.append("eau chlorée")
            if zone_agricole:
                raisons.append("zone agricole (risque nitrates)")
            if historique_odeur:
                raisons.append("historique d'odeurs")
            if source.lower() in ["réseau", "reseau", "rivière", "riviere", "fleuve"]:
                raisons.append(f"source '{source}'")
            to_activate.append((
                3,
                "Risque chimique/organique : " + ", ".join(raisons)
            ))

        # ── Cas par défaut : eau acceptable, filtration de base ───────────
        if not to_activate:
            to_activate.append((2, "Filtration standard (minéralisation normale)"))
            to_activate.append((3, "Filtration standard (précaution organique)"))

        return to_activate

    # ── Pipeline complet : décision + activation ──────────────────────────────

    def decide_and_activate(
        self,
        diagnostic_result:  dict[str, Any],
        source:             str  = "inconnue",
        zone_agricole:      bool = False,
        historique_odeur:   bool = False,
        eau_chloree:        bool = False,
        duration:           float | None = None,
    ) -> list[FilterAction]:
        """
        Point d'entrée principal.

        Reçoit le résultat de ``sensor_inference.SensorPipeline.run_once()``
        et active les pompes appropriées séquentiellement.

        Parameters
        ----------
        diagnostic_result : dict
            Sortie de ``SensorPipeline.run_once()`` (clé ``raw_values``).
        source, zone_agricole, historique_odeur, eau_chloree :
            Contexte supplémentaire (peut être fourni via l'interface web).
        duration : float | None
            Durée d'activation (override de pump_duration).

        Returns
        -------
        list[FilterAction]
            Actions effectuées (une par filtre activé).
        """
        raw_values = diagnostic_result.get("raw_values", {})
        if not raw_values:
            logger.warning("decide_and_activate : raw_values vide — aucune action.")
            return []

        filters_to_run = self.decide_filters(
            raw_values      = raw_values,
            source          = source,
            zone_agricole   = zone_agricole,
            historique_odeur= historique_odeur,
            eau_chloree     = eau_chloree,
        )

        actions = []
        for fid, reason in filters_to_run:
            action = self.activate_pump(fid, reason=reason, duration=duration)
            actions.append(action)

        logger.info(
            "Filtration terminée : %d filtre(s) activé(s) → %s",
            len(actions),
            [a.filter_name for a in actions],
        )
        return actions