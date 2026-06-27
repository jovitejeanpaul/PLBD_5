"""
5_dashboard.py
===============
Interface Tkinter de prévision à t+24h avec :
    - Onglet Automatique : utilise l'historique des WINDOW_SIZE dernières
      mesures stockées (simulées au démarrage, remplaçables par de vraies
      lectures capteurs)
    - Onglet Manuel      : saisie libre des WINDOW_SIZE mesures horaires
    - Intégration        : après chaque prévision LSTM, les 4 features
      diagnostiques sont passées au modèle CatBoost pour obtenir la
      classe de potabilité à t+24h
    - Zones OMS          : affichées sur chaque graphique
    - Analyse des tendances : résumé textuel automatique

Usage
-----
    python prediction/5_dashboard.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk, messagebox

import joblib
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from model import FEATURES, HORIZON, LSTMModel, N_FEATURES, WINDOW_SIZE

# ── Chemins ─────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
OUTPUTS_DIR = BASE.parent / "outputs" / "models"
sys.path.insert(0, str(BASE.parent / "src"))

# ── Seuils OMS par feature ────────────────────────────────────────────────────
OMS_LIMITS = {
    "ph":           (6.5,  8.5),
    "Solids":       (0,    1800),    # TDS ≤ 600 mg/L (OMS)
    "Conductivity": (0,    2700),
    "Turbidity":    (0,    5),      # NTU ≤ 5 (OMS)
    "temperature":  (0,    50),
}
OMS_LABELS = {
    "ph":           "pH : 6.5–8.5",
    "Solids":       "TDS ≤ 1800 mg/L",
    "Conductivity": "≤ 2700 µS/cm",
    "Turbidity":    "≤ 5 NTU",
    "temperature":  "≤ 50 °C",
}
UNITS = {
    "ph":           "",
    "Solids":       "mg/L",
    "Conductivity": "µS/cm",
    "Turbidity":    "NTU",
    "temperature":  "°C",
}

DIAG_FEATURES = ["ph", "Solids", "Conductivity", "Turbidity"]
DIAG_IDX      = [FEATURES.index(f) for f in DIAG_FEATURES]

# ── Chargement des modèles ────────────────────────────────────────────────────
def _load_lstm():
    m = LSTMModel()
    m.load_state_dict(torch.load(BASE / "models/lstm_model.pth", map_location="cpu"))
    m.eval()
    return m

def _load_scaler():
    return joblib.load(BASE / "data/scaler.joblib")

def _load_diagnostic():
    """Charge le modèle CatBoost + scaler du module de diagnostic (optionnel)."""
    path = next(OUTPUTS_DIR.glob("model_1_*.joblib"), None) if OUTPUTS_DIR.exists() else None
    scaler_path = OUTPUTS_DIR / "scaler.joblib" if OUTPUTS_DIR.exists() else None
    if path and scaler_path and scaler_path.exists():
        try:
            from src.threshold_classifier import ThresholdClassifier  # noqa: F401
            return joblib.load(path), joblib.load(scaler_path)
        except Exception:
            pass
    return None, None

try:
    LSTM_MODEL   = _load_lstm()
    SCALER       = _load_scaler()
    DIAG_MODEL, DIAG_SCALER = _load_diagnostic()
    DIAG_AVAIL   = DIAG_MODEL is not None
except FileNotFoundError as e:
    tk.Tk().withdraw()
    messagebox.showerror("Erreur de chargement", str(e))
    sys.exit(1)

# ── Historique simulé (WINDOW_SIZE mesures) ───────────────────────────────────
# Remplacé par de vraies lectures capteurs lors du déploiement Pi.
def _make_default_history() -> np.ndarray:
    """Génère WINDOW_SIZE mesures réalistes pour initialiser l'historique."""
    rng   = np.random.default_rng(0)
    means = dict(zip(FEATURES, [7.2,1000.0,1500.0,3.8,22.0]))
    stds  = dict(zip(FEATURES, [5.0,200,500.0,2.0, 8.0]))
    rows  = []
    for _ in range(WINDOW_SIZE):
        row = [np.clip(rng.normal(means[f], stds[f]), *_phys_bounds(f)) for f in FEATURES]
        rows.append(row)
    return np.array(rows, dtype=np.float32)   # (WINDOW_SIZE, N_FEATURES)

def _phys_bounds(feat: str):
    bounds = {"ph":(0,14), "Solids":(0,3000), "Conductivity":(0,3500),
              "Turbidity":(0,100), "temperature":(0,50)}
    return bounds.get(feat, (0, 1e6))

# ── Moteur de prévision ───────────────────────────────────────────────────────
def predict(history: np.ndarray) -> dict:
    """
    Lance la prévision LSTM + diagnostic CatBoost optionnel.

    Parameters
    ----------
    history : np.ndarray, shape (WINDOW_SIZE, N_FEATURES)

    Returns
    -------
    dict avec :
        predicted       : np.ndarray (HORIZON, N_FEATURES) — valeurs dénormalisées
        potability_24h  : str | None — "Potable" / "Non potable"
        proba_24h       : float | None
    """
    if history.shape != (WINDOW_SIZE, N_FEATURES):
        raise ValueError(f"history doit avoir shape ({WINDOW_SIZE}, {N_FEATURES}), "
                         f"reçu {history.shape}")

    # Normalisation + inférence LSTM
    seq_scaled = SCALER.transform(history)   # (WINDOW_SIZE, N_FEATURES)
    x_tensor   = torch.FloatTensor(seq_scaled).unsqueeze(0)   # (1, WINDOW_SIZE, N_FEATURES)
    with torch.no_grad():
        y_scaled = LSTM_MODEL(x_tensor).squeeze(0).numpy()    # (HORIZON, N_FEATURES)

    # Dénormalisation
    predicted = SCALER.inverse_transform(y_scaled)   # (HORIZON, N_FEATURES)

    # Diagnostic CatBoost à t+24h (dernier pas de l'horizon)
    potability, proba = None, None
    if DIAG_AVAIL:
        row_24h  = predicted[-1, DIAG_IDX].reshape(1, -1)        # (1, 4)
        row_s    = DIAG_SCALER.transform(row_24h)
        pot_pred = int(DIAG_MODEL.predict(row_s)[0])
        proba    = float(DIAG_MODEL.predict_proba(row_s)[0][1])
        potability = "Non potable" if pot_pred == 1 else "Potable"

    return {"predicted": predicted, "potability_24h": potability, "proba_24h": proba}


def analyser_tendance(predicted: np.ndarray) -> str:
    """Résumé textuel des tendances sur l'horizon de prévision."""
    lignes = []
    for i, feat in enumerate(FEATURES):
        vals   = predicted[:, i]
        lo, hi = OMS_LIMITS[feat]
        pct_hors = np.mean((vals < lo) | (vals > hi)) * 100
        direction = "↗" if vals[-1] > vals[0] else ("↘" if vals[-1] < vals[0] else "→")
        statut    = "⚠️" if pct_hors > 0 else "✅"
        oms_lbl   = OMS_LABELS[feat]
        lignes.append(
            f"{statut} {feat} {direction}  "
            f"moy={vals.mean():.2f} {UNITS[feat]}  "
            f"({pct_hors:.0f}% hors {oms_lbl})"
        )
    return "\n".join(lignes)


# ════════════════════════════════════════════════════════════════════════════
# INTERFACE GRAPHIQUE
# ════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"AQUA-SENS — Prévision {HORIZON}h")
        self.geometry("1200x780")
        self.configure(bg="#1e1e2e")
        self.resizable(True, True)

        # Données partagées
        self._history = _make_default_history()
        self._result  = None

        # Notebook (onglets)
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",       background="#1e1e2e", borderwidth=0)
        style.configure("TNotebook.Tab",   background="#2a2a3e", foreground="white",
                        padding=[14, 6])
        style.map("TNotebook.Tab",         background=[("selected", "#065A82")])

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        self._tab_auto   = ttk.Frame(nb, style="TFrame")
        self._tab_manual = ttk.Frame(nb, style="TFrame")
        nb.add(self._tab_auto,   text=f"▶  Automatique  ({WINDOW_SIZE} mesures)")
        nb.add(self._tab_manual, text="✏️  Manuel")

        self._build_tab_auto()
        self._build_tab_manual()

    # ── Onglet Automatique ────────────────────────────────────────────────────
    def _build_tab_auto(self):
        frame = self._tab_auto
        frame.configure(style="TFrame")

        # Bandeau info
        info = tk.Label(
            frame,
            text=(f"Historique simulé de {WINDOW_SIZE} mesures horaires — "
                  "Remplacez _make_default_history() par des lectures capteurs réelles"),
            bg="#065A82", fg="white", font=("Calibri", 9, "italic"), pady=4,
        )
        info.pack(fill="x")

        btn = tk.Button(
            frame, text="🔮  Lancer la prévision",
            bg="#02C39A", fg="#1e1e2e", font=("Calibri", 12, "bold"),
            relief="flat", padx=20, pady=8,
            command=self._run_auto,
        )
        btn.pack(pady=10)

        self._canvas_auto = tk.Frame(frame, bg="#1e1e2e")
        self._canvas_auto.pack(fill="both", expand=True)

        self._result_auto = tk.StringVar(value="")
        tk.Label(
            frame, textvariable=self._result_auto,
            bg="#1e1e2e", fg="#02C39A", font=("Calibri", 11, "bold"), justify="left",
        ).pack(pady=4)

    def _run_auto(self):
        try:
            result = predict(self._history)
            self._draw_charts(result["predicted"], self._canvas_auto)
            self._show_result_banner(result, self._result_auto)
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    # ── Onglet Manuel ─────────────────────────────────────────────────────────
    def _build_tab_manual(self):
        frame = self._tab_manual

        tk.Label(
            frame,
            text=(f"Entrez les {WINDOW_SIZE} dernières mesures horaires "
                  f"({', '.join(FEATURES)})"),
            bg="#1e1e2e", fg="#CADCFC", font=("Calibri", 10),
        ).pack(pady=6)

        # Cadre saisie scrollable
        outer = tk.Frame(frame, bg="#1e1e2e")
        outer.pack(fill="both", expand=False, padx=10)

        canvas = tk.Canvas(outer, bg="#1e1e2e", height=220, highlightthickness=0)
        sb     = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg="#1e1e2e")
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # En-têtes
        for j, feat in enumerate(FEATURES):
            tk.Label(
                inner, text=feat, bg="#065A82", fg="white",
                font=("Calibri", 9, "bold"), width=13, anchor="center",
            ).grid(row=0, column=j + 1, padx=2, pady=2)

        # Valeurs par défaut (dernière rangée de l'historique par défaut)
        means = dict(zip(FEATURES, [7.2, 18500, 420, 3.8, 22.0]))
        self._entries = []
        for i in range(WINDOW_SIZE):
            tk.Label(
                inner, text=f"h-{WINDOW_SIZE-i}", bg="#2a2a3e", fg="#CADCFC",
                font=("Calibri", 8), width=5,
            ).grid(row=i + 1, column=0, padx=2, pady=1)
            row_entries = []
            for j, feat in enumerate(FEATURES):
                e = tk.Entry(inner, width=10, bg="#2a2a3e", fg="white",
                             insertbackground="white", font=("Calibri", 9))
                e.insert(0, str(round(means[feat] + np.random.normal(0, 0.01 * means[feat]), 2)))
                e.grid(row=i + 1, column=j + 1, padx=2, pady=1)
                row_entries.append(e)
            self._entries.append(row_entries)

        # Bouton + résultats
        tk.Button(
            frame, text="🔮  Calculer la prévision",
            bg="#02C39A", fg="#1e1e2e", font=("Calibri", 12, "bold"),
            relief="flat", padx=20, pady=8,
            command=self._run_manual,
        ).pack(pady=10)

        self._canvas_manual = tk.Frame(frame, bg="#1e1e2e")
        self._canvas_manual.pack(fill="both", expand=True)

        self._result_manual = tk.StringVar(value="")
        tk.Label(
            frame, textvariable=self._result_manual,
            bg="#1e1e2e", fg="#02C39A", font=("Calibri", 11, "bold"), justify="left",
        ).pack(pady=4)

    def _run_manual(self):
        try:
            history = np.zeros((WINDOW_SIZE, N_FEATURES), dtype=np.float32)
            for i in range(WINDOW_SIZE):
                for j in range(N_FEATURES):
                    val = self._entries[i][j].get()
                    lo, hi = _phys_bounds(FEATURES[j])
                    v = float(val)
                    if not (lo <= v <= hi):
                        raise ValueError(
                            f"Valeur hors bornes pour {FEATURES[j]} à h-{WINDOW_SIZE-i} : "
                            f"{v} (attendu [{lo}, {hi}])"
                        )
                    history[i, j] = v
            result = predict(history)
            self._draw_charts(result["predicted"], self._canvas_manual)
            self._show_result_banner(result, self._result_manual)
        except ValueError as e:
            messagebox.showerror("Valeur invalide", str(e))
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    # ── Rendu graphique ───────────────────────────────────────────────────────
    def _draw_charts(self, predicted: np.ndarray, container: tk.Frame):
        """Dessine les N_FEATURES sous-graphes avec zones OMS."""
        for w in container.winfo_children():
            w.destroy()

        ncols = 3
        nrows = (N_FEATURES + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.5 * nrows))
        fig.patch.set_facecolor("#1e1e2e")
        axes_flat = axes.flatten()

        hours = list(range(1, HORIZON + 1))
        colors = ["#2E86AB", "#E84855", "#3BB273", "#F18F01", "#7B2D8B"]

        for i, feat in enumerate(FEATURES):
            ax    = axes_flat[i]
            vals  = predicted[:, i]
            lo, hi = OMS_LIMITS[feat]

            ax.set_facecolor("#16213e")
            ax.plot(hours, vals, color=colors[i % len(colors)], lw=2, marker="o",
                    markersize=3, label=feat)

            # Zone OMS en vert transparent
            ax.axhspan(lo, hi, alpha=0.12, color="#02C39A", label="Zone OMS")
            ax.axhline(hi, color="#02C39A", lw=0.8, linestyle="--", alpha=0.6)
            if lo > 0:
                ax.axhline(lo, color="#02C39A", lw=0.8, linestyle="--", alpha=0.6)

            unit = UNITS[feat]
            ax.set_title(f"{feat}  [{unit}]" if unit else feat,
                         color="white", fontsize=9)
            ax.set_xlabel("Heure +", color="#AAAAAA", fontsize=8)
            ax.tick_params(colors="#AAAAAA", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")
            ax.legend(fontsize=7, facecolor="#2a2a3e", labelcolor="white")

        # Cacher les axes inutilisés
        for j in range(N_FEATURES, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.tight_layout(pad=1.5)
        canvas_widget = FigureCanvasTkAgg(fig, master=container)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    # ── Bandeau résultat ──────────────────────────────────────────────────────
    def _show_result_banner(self, result: dict, var: tk.StringVar):
        tendance = analyser_tendance(result["predicted"])
        lines = [f"── Tendances prévues sur {HORIZON}h ──", tendance]

        if result["potability_24h"]:
            emoji = "🚨" if result["potability_24h"] == "Non potable" else "✅"
            lines.append(
                f"\n{emoji}  Potabilité à t+24h : {result['potability_24h']}"
                f"  (confiance : {result['proba_24h']:.1%})"
            )
        else:
            lines.append("\nℹ️  Modèle de diagnostic non disponible")

        var.set("\n".join(lines))


if __name__ == "__main__":
    app = App()
    app.mainloop()
