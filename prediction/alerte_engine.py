"""
alert_engine.py
================
Moteur d'alertes préventives pour les opérateurs de la source d'eau.

Fonctionnement
--------------
Les alertes sont déclenchées à partir des **valeurs prévues** par le LSTM
(horizon 24h), pas des valeurs actuelles. L'objectif est de donner aux
opérateurs le temps d'intervenir **avant** qu'un problème survienne.

Deux niveaux d'analyse
-----------------------
1. **Seuils directs** — comparaison des valeurs prévues à la norme
   marocaine NM 03.7.001 (eau potable). Une valeur qui dépasse le seuil
   dans les 24h déclenche une alerte.

2. **Règles combinées** — détection de signatures multi-paramètres
   corrélées à des risques non mesurés directement (contamination
   microbiologique, intrusion agricole, acidification...). Ces alertes
   sont explicitement qualifiées d'"indicateur de risque", pas de certitude.

Norme de référence
-------------------
NM 03.7.001 — Norme Marocaine pour l'eau destinée à la consommation humaine
(IMANOR, dernière révision). Valeurs plus adaptées au contexte marocain
que les seuils OMS génériques (notamment conductivité et TDS plus élevés).

Niveaux d'alerte
-----------------
- INFO    : tendance à surveiller, pas encore critique
- WARNING : seuil approché (> 80% de la limite)
- CRITICAL: seuil dépassé dans l'horizon de prévision

Usage
------
    from alert_engine import AlertEngine
    engine = AlertEngine()
    alerts = engine.analyze(predicted_values, timestamps)
    for alert in alerts:
        print(alert)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import TDS_EC_FACTOR

logger = logging.getLogger(__name__)


# ===========================================================================
# NORME MAROCAINE NM 03.7.001
# ===========================================================================

MOROCCAN_STANDARDS: dict[str, dict] = {
    "ph": {
        "min":   6.5,
        "max":   8.5,
        "unit":  "",
        "label": "pH",
        "norm":  "NM 03.7.001 §5.1",
    },
    "Turbidity": {
        "min":   0.0,
        "max":   5.0,
        "unit":  "NTU",
        "label": "Turbidité",
        "norm":  "NM 03.7.001 §5.2",
    },
    "Conductivity": {
        "min":   0.0,
        "max":   2700.0,
        "unit":  "µS/cm",
        "label": "Conductivité",
        "norm":  "NM 03.7.001 §5.3",
    },
    "Solids": {
        "min":   0.0,
        "max":   1500.0,
        "unit":  "mg/L",
        "label": "TDS (Solides dissous)",
        "norm":  "NM 03.7.001 §5.3",
    },
    "Temperature": {
        "min":   0.0,
        "max":   25.0,
        "unit":  "°C",
        "label": "Température",
        "norm":  "NM 03.7.001 §5.4",
    },
}

# Seuil d'approche : alerte WARNING si valeur > 80% de la limite max
WARNING_RATIO: float = 0.80


# ===========================================================================
# STRUCTURES DE DONNÉES
# ===========================================================================

class AlertLevel(str, Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


class AlertCategory(str, Enum):
    THRESHOLD   = "SEUIL_DIRECT"       # dépassement d'un seuil NM 03.7.001
    COMBINED    = "RÈGLE_COMBINÉE"     # signature multi-paramètres
    TREND       = "TENDANCE"           # évolution temporelle préoccupante


@dataclass
class Alert:
    """Représente une alerte préventive destinée aux opérateurs."""

    level:       AlertLevel
    category:    AlertCategory
    feature:     str                     # feature(s) impliquée(s)
    message:     str                     # message lisible opérateur
    detail:      str                     # explication technique
    hour_onset:  int                     # heure d'horizon à laquelle l'alerte est déclenchée (1-24)
    value:       float | None = None     # valeur prévue impliquée
    threshold:   float | None = None     # seuil de référence
    risk:        str | None   = None     # risque inféré (règles combinées)
    norm_ref:    str | None   = None     # référence normative
    timestamp:   str          = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def __str__(self) -> str:
        icon = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}[self.level]
        val_str = f" (valeur prévue : {self.value:.3g} {self._unit()})" if self.value is not None else ""
        thr_str = f" — seuil : {self.threshold:.3g}" if self.threshold is not None else ""
        return (
            f"{icon} [{self.level}] h+{self.hour_onset:02d} | "
            f"{self.message}{val_str}{thr_str}"
        )

    def _unit(self) -> str:
        std = MOROCCAN_STANDARDS.get(self.feature, {})
        return std.get("unit", "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "level":      self.level,
            "category":   self.category,
            "feature":    self.feature,
            "message":    self.message,
            "detail":     self.detail,
            "hour_onset": self.hour_onset,
            "value":      self.value,
            "threshold":  self.threshold,
            "risk":       self.risk,
            "norm_ref":   self.norm_ref,
            "timestamp":  self.timestamp,
        }


# ===========================================================================
# MOTEUR D'ALERTES
# ===========================================================================

class AlertEngine:
    """
    Analyse les prévisions LSTM et produit des alertes préventives.

    Parameters
    ----------
    warning_ratio : float
        Fraction de la limite max au-delà de laquelle une alerte WARNING
        est émise avant le dépassement effectif. Défaut : 0.80 (80%).
    standards : dict | None
        Seuils normatifs. Si None, utilise MOROCCAN_STANDARDS.
    """

    def __init__(
        self,
        warning_ratio: float = WARNING_RATIO,
        standards: dict | None = None,
    ):
        self.warning_ratio = warning_ratio
        self.standards     = standards or MOROCCAN_STANDARDS

    def analyze(
        self,
        predicted: np.ndarray,
        feature_names: list[str],
        horizon_h: int = 24,
    ) -> list[Alert]:
        """
        Analyse un tableau de prévisions et retourne toutes les alertes.

        Parameters
        ----------
        predicted : np.ndarray, shape (horizon_h, n_features)
            Valeurs prévues par le LSTM (dans les unités physiques d'origine,
            après dénormalisation).
        feature_names : list[str]
            Noms des colonnes de ``predicted`` (même ordre).
        horizon_h : int
            Nombre d'heures de l'horizon (défaut : 24).

        Returns
        -------
        list[Alert]
            Alertes triées par niveau décroissant (CRITICAL > WARNING > INFO),
            puis par heure d'onset.
        """
        if predicted.shape != (horizon_h, len(feature_names)):
            raise ValueError(
                f"predicted doit avoir shape ({horizon_h}, {len(feature_names)}), "
                f"reçu {predicted.shape}"
            )

        alerts: list[Alert] = []

        # 1. Seuils directs NM 03.7.001
        alerts.extend(self._check_thresholds(predicted, feature_names))

        # 2. Règles combinées (risques inférés)
        alerts.extend(self._check_combined_rules(predicted, feature_names))

        # 3. Tendances préoccupantes
        alerts.extend(self._check_trends(predicted, feature_names))

        # Tri : CRITICAL d'abord, puis WARNING, puis INFO ; à heure égale
        level_order = {AlertLevel.CRITICAL: 0, AlertLevel.WARNING: 1, AlertLevel.INFO: 2}
        alerts.sort(key=lambda a: (level_order[a.level], a.hour_onset))

        logger.info(
            "Analyse terminée : %d alertes (%d CRITICAL, %d WARNING, %d INFO)",
            len(alerts),
            sum(1 for a in alerts if a.level == AlertLevel.CRITICAL),
            sum(1 for a in alerts if a.level == AlertLevel.WARNING),
            sum(1 for a in alerts if a.level == AlertLevel.INFO),
        )
        return alerts

    # ------------------------------------------------------------------
    # 1. SEUILS DIRECTS
    # ------------------------------------------------------------------

    def _check_thresholds(
        self,
        predicted: np.ndarray,
        feature_names: list[str],
    ) -> list[Alert]:
        alerts = []

        for j, feat in enumerate(feature_names):
            std = self.standards.get(feat)
            if std is None:
                continue

            col  = predicted[:, j]
            lo   = std.get("min", -np.inf)
            hi   = std.get("max",  np.inf)
            lbl  = std["label"]
            unit = std.get("unit", "")
            ref  = std.get("norm", "NM 03.7.001")

            for h, val in enumerate(col, start=1):

                # — Dépassement critique —
                if val > hi:
                    alerts.append(Alert(
                        level      = AlertLevel.CRITICAL,
                        category   = AlertCategory.THRESHOLD,
                        feature    = feat,
                        message    = f"{lbl} prévu au-dessus du seuil réglementaire",
                        detail     = (
                            f"Valeur prévue à h+{h:02d} : {val:.3g} {unit} "
                            f"(limite NM : {hi} {unit}). "
                            f"Dépassement de {val - hi:.3g} {unit}."
                        ),
                        hour_onset = h,
                        value      = round(float(val), 4),
                        threshold  = hi,
                        norm_ref   = ref,
                    ))
                    break   # on ne signale que le premier dépassement par feature

                if val < lo:
                    alerts.append(Alert(
                        level      = AlertLevel.CRITICAL,
                        category   = AlertCategory.THRESHOLD,
                        feature    = feat,
                        message    = f"{lbl} prévu en dessous du seuil réglementaire",
                        detail     = (
                            f"Valeur prévue à h+{h:02d} : {val:.3g} {unit} "
                            f"(limite NM : {lo} {unit}). "
                            f"Écart : {lo - val:.3g} {unit}."
                        ),
                        hour_onset = h,
                        value      = round(float(val), 4),
                        threshold  = lo,
                        norm_ref   = ref,
                    ))
                    break

                # — Approche du seuil max (WARNING) —
                if (hi < np.inf
                        and val > self.warning_ratio * hi
                        and val <= hi):
                    alerts.append(Alert(
                        level      = AlertLevel.WARNING,
                        category   = AlertCategory.THRESHOLD,
                        feature    = feat,
                        message    = f"{lbl} approche du seuil réglementaire",
                        detail     = (
                            f"Valeur prévue à h+{h:02d} : {val:.3g} {unit} "
                            f"({val / hi * 100:.0f}% de la limite NM {hi} {unit})."
                        ),
                        hour_onset = h,
                        value      = round(float(val), 4),
                        threshold  = hi,
                        norm_ref   = ref,
                    ))
                    break   # un seul WARNING par feature

                # — Approche du seuil min (WARNING) —
                if (lo > 0
                        and val < lo / self.warning_ratio
                        and val >= lo):
                    alerts.append(Alert(
                        level      = AlertLevel.WARNING,
                        category   = AlertCategory.THRESHOLD,
                        feature    = feat,
                        message    = f"{lbl} approche du seuil minimum réglementaire",
                        detail     = (
                            f"Valeur prévue à h+{h:02d} : {val:.3g} {unit} "
                            f"(seuil min NM : {lo} {unit})."
                        ),
                        hour_onset = h,
                        value      = round(float(val), 4),
                        threshold  = lo,
                        norm_ref   = ref,
                    ))
                    break

        return alerts

    # ------------------------------------------------------------------
    # 2. RÈGLES COMBINÉES (risques inférés)
    # ------------------------------------------------------------------

    def _check_combined_rules(
        self,
        predicted: np.ndarray,
        feature_names: list[str],
    ) -> list[Alert]:
        """
        Règles multi-paramètres pour inférer des risques non mesurés.

        Ces alertes sont des **indicateurs de risque**, pas des certitudes
        analytiques. Elles sont basées sur des corrélations physico-chimiques
        connues et doivent conduire à un prélèvement de confirmation.
        """
        alerts = []
        idx    = {f: i for i, f in enumerate(feature_names)}

        def col(feat):
            return predicted[:, idx[feat]] if feat in idx else None

        ph_arr   = col("ph")
        turb_arr = col("Turbidity")
        cond_arr = col("Conductivity")
        tds_arr  = col("Solids")
        temp_arr = col("Temperature")

        # ── Règle 1 : Contamination microbiologique probable ──────────────
        # Turbidité élevée + chute de pH => ruissellement de surface
        # => risque E. coli, coliformes
        if ph_arr is not None and turb_arr is not None:
            ph_drop   = ph_arr[0] - ph_arr.min()     # chute max de pH
            turb_peak = turb_arr.max()

            if ph_drop > 0.5 and turb_peak > 4.0:
                h_onset = int(np.argmax(turb_arr)) + 1
                alerts.append(Alert(
                    level      = AlertLevel.CRITICAL,
                    category   = AlertCategory.COMBINED,
                    feature    = "ph + Turbidity",
                    message    = "Risque de contamination microbiologique",
                    detail     = (
                        f"Signature détectée : turbidité prévue à {turb_peak:.2f} NTU "
                        f"avec chute de pH de {ph_drop:.2f} unités. "
                        "Corrélé à un ruissellement de surface ou à un événement pluvieux. "
                        "Risque de présence de coliformes / E. coli."
                    ),
                    hour_onset = h_onset,
                    risk       = "Contamination microbiologique (E. coli, coliformes)",
                    norm_ref   = "Règle combinée — prélèvement de confirmation recommandé",
                ))

        # ── Règle 2 : Intrusion agricole (nitrates / pesticides) ──────────
        # Conductivité qui monte brutalement sans hausse de turbidité
        # => possible lessivage de terres agricoles (engrais ioniques)
        if cond_arr is not None and turb_arr is not None:
            cond_rise = cond_arr.max() - cond_arr[0]
            turb_rise = turb_arr.max() - turb_arr[0]

            if cond_rise > 200 and turb_rise < 1.0:
                h_onset = int(np.argmax(cond_arr)) + 1
                alerts.append(Alert(
                    level      = AlertLevel.WARNING,
                    category   = AlertCategory.COMBINED,
                    feature    = "Conductivity",
                    message    = "Possible intrusion agricole (nitrates / engrais)",
                    detail     = (
                        f"Hausse de conductivité prévue : +{cond_rise:.0f} µS/cm "
                        f"sans augmentation notable de turbidité ({turb_rise:.2f} NTU). "
                        "Signature typique d'un lessivage d'engrais ioniques "
                        "(nitrates, sulfates). Capteurs non disponibles pour confirmation."
                    ),
                    hour_onset = h_onset,
                    risk       = "Nitrates / pesticides (contamination agricole)",
                    norm_ref   = "Règle combinée — analyse nitrates recommandée",
                ))

        # ── Règle 3 : Acidification progressive ───────────────────────────
        # pH qui descend régulièrement sur > 12h sous 6.8
        if ph_arr is not None:
            ph_trend = ph_arr[-1] - ph_arr[0]    # variation sur l'horizon
            ph_min   = ph_arr.min()

            if ph_trend < -0.3 and ph_min < 6.8:
                h_onset = int(np.argmin(ph_arr)) + 1
                alerts.append(Alert(
                    level      = AlertLevel.WARNING,
                    category   = AlertCategory.COMBINED,
                    feature    = "ph",
                    message    = "Acidification progressive de la source",
                    detail     = (
                        f"pH prévu en baisse de {abs(ph_trend):.2f} unités sur {len(ph_arr)}h "
                        f"(min prévu : {ph_min:.2f}). "
                        "Peut indiquer une pollution atmosphérique acide, "
                        "une décomposition organique ou une activité industrielle en amont."
                    ),
                    hour_onset = h_onset,
                    risk       = "Acidification — possible pollution industrielle ou organique",
                    norm_ref   = "Règle combinée",
                ))

        # ── Règle 4 : Anomalie TDS / Conductivité ─────────────────────────
        # TDS monte mais conductivité reste stable
        # => matières organiques dissoutes non ioniques (hydrocarbures, solvants)
        if tds_arr is not None and cond_arr is not None:
            tds_rise  = tds_arr.max()  - tds_arr[0]
            cond_rise = cond_arr.max() - cond_arr[0]
            # Ratio attendu : TDS ≈ Conductivité × TDS_EC_FACTOR
            # Si TDS augmente bien plus vite que prévu par la conductivité => anomalie
            expected_tds_rise = cond_rise * TDS_EC_FACTOR
            if tds_rise > expected_tds_rise + 50:
                h_onset = int(np.argmax(tds_arr)) + 1
                alerts.append(Alert(
                    level      = AlertLevel.WARNING,
                    category   = AlertCategory.COMBINED,
                    feature    = "Solids + Conductivity",
                    message    = "Anomalie TDS / Conductivité — possible contamination organique",
                    detail     = (
                        f"TDS prévu en hausse de {tds_rise:.0f} mg/L, "
                        f"alors que la conductivité n'explique que {expected_tds_rise:.0f} mg/L "
                        f"(ratio TDS/EC anormal). "
                        "Peut indiquer des composés organiques non ioniques dissous "
                        "(hydrocarbures, solvants)."
                    ),
                    hour_onset = h_onset,
                    risk       = "Contamination organique non ionique (hydrocarbures, solvants)",
                    norm_ref   = "Règle combinée — analyse organique recommandée",
                ))

        # ── Règle 5 : Prolifération bactérienne thermique ─────────────────
        # Température > 22°C sur plus de 6h consécutives
        if temp_arr is not None:
            hot_hours = np.sum(temp_arr > 22.0)
            if hot_hours >= 6:
                h_onset = int(np.argmax(temp_arr > 22.0)) + 1
                alerts.append(Alert(
                    level      = AlertLevel.WARNING,
                    category   = AlertCategory.COMBINED,
                    feature    = "Temperature",
                    message    = "Température favorable à la prolifération microbienne",
                    detail     = (
                        f"Température prévue > 22°C pendant {hot_hours}h consécutives. "
                        "Au-dessus de 20°C, la vitesse de multiplication bactérienne "
                        "double toutes les ~20 min (E. coli). "
                        "Surveillance renforcée recommandée, surtout en période estivale."
                    ),
                    hour_onset = h_onset,
                    risk       = "Prolifération bactérienne (E. coli, légionelles)",
                    norm_ref   = "Règle combinée — référence OMS température eau",
                ))

        return alerts

    # ------------------------------------------------------------------
    # 3. TENDANCES
    # ------------------------------------------------------------------

    def _check_trends(
        self,
        predicted: np.ndarray,
        feature_names: list[str],
    ) -> list[Alert]:
        """Détecte les tendances monotones préoccupantes sur l'horizon."""
        alerts = []

        for j, feat in enumerate(feature_names):
            std = self.standards.get(feat)
            if std is None:
                continue

            col  = predicted[:, j]
            hi   = std.get("max", np.inf)
            lo   = std.get("min", -np.inf)
            lbl  = std["label"]
            unit = std.get("unit", "")

            # Pente sur l'horizon (régression linéaire simple)
            x     = np.arange(len(col), dtype=float)
            slope = float(np.polyfit(x, col, 1)[0])

            # Ignorer les pentes insignifiantes (< 0.1% de la valeur moyenne)
            mean_val      = float(np.abs(col).mean()) or 1.0
            min_slope_abs = 0.001 * mean_val
            if abs(slope) < min_slope_abs:
                continue

            current = col[0]
            final   = col[-1]

            # Tendance à la hausse qui rapproche du seuil max
            if hi < np.inf and slope > 0 and current < hi and final > 0.70 * hi:
                alerts.append(Alert(
                    level      = AlertLevel.INFO,
                    category   = AlertCategory.TREND,
                    feature    = feat,
                    message    = f"{lbl} en hausse continue vers le seuil réglementaire",
                    detail     = (
                        f"Pente prévue : +{slope:.4g} {unit}/h. "
                        f"Valeur actuelle : {current:.3g} {unit} → "
                        f"prévue à {final:.3g} {unit} dans {len(col)}h "
                        f"(limite NM : {hi} {unit})."
                    ),
                    hour_onset = len(col),
                    value      = round(float(final), 4),
                    threshold  = hi,
                    norm_ref   = std.get("norm", "NM 03.7.001"),
                ))

            # Tendance à la baisse qui rapproche du seuil min (pH surtout)
            if lo > 0 and slope < 0 and current > lo and final < 1.30 * lo:
                alerts.append(Alert(
                    level      = AlertLevel.INFO,
                    category   = AlertCategory.TREND,
                    feature    = feat,
                    message    = f"{lbl} en baisse continue vers le seuil réglementaire",
                    detail     = (
                        f"Pente prévue : {slope:.4g} {unit}/h. "
                        f"Valeur actuelle : {current:.3g} {unit} → "
                        f"prévue à {final:.3g} {unit} dans {len(col)}h "
                        f"(limite NM min : {lo} {unit})."
                    ),
                    hour_onset = len(col),
                    value      = round(float(final), 4),
                    threshold  = lo,
                    norm_ref   = std.get("norm", "NM 03.7.001"),
                ))

        return alerts

    # ------------------------------------------------------------------
    # RÉSUMÉ
    # ------------------------------------------------------------------

    def summary(self, alerts: list[Alert]) -> str:
        """Retourne un résumé textuel lisible des alertes pour l'interface."""
        if not alerts:
            return "✅ Aucune alerte préventive — qualité de source conforme sur 24h"

        n_crit = sum(1 for a in alerts if a.level == AlertLevel.CRITICAL)
        n_warn = sum(1 for a in alerts if a.level == AlertLevel.WARNING)
        n_info = sum(1 for a in alerts if a.level == AlertLevel.INFO)

        lines = [
            f"━━━ RAPPORT D'ALERTES PRÉVENTIVES (horizon 24h) ━━━",
            f"🚨 CRITICAL : {n_crit}  ⚠️ WARNING : {n_warn}  ℹ️ INFO : {n_info}",
            "",
        ]
        for a in alerts:
            lines.append(str(a))
            lines.append(f"   → {a.detail}")
            if a.risk:
                lines.append(f"   💡 Risque inféré : {a.risk}")
            lines.append("")

        lines.append("━━━ FIN DU RAPPORT ━━━")
        return "\n".join(lines)