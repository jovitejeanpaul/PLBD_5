"""
1_generate_data.py
==================
Génération d'une série temporelle synthétique de 120 jours (1 mesure/heure)
pour 5 capteurs alignés avec le module de diagnostic immédiat :

    ph · Solids · Conductivity · Turbidity · temperature

Alignement avec le module de diagnostic
-----------------------------------------
Le module de diagnostic (src/diagnostic) utilise les features :
    ph, Solids, Conductivity, Turbidity
avec les conventions Kaggle Water Potability.

Ce fichier adopte exactement ces noms afin que les prédictions du LSTM
puissent être directement passées au modèle de diagnostic (CatBoost).

    Conductivity [µS/cm]  →  Solids [mg/L] = Conductivity × TDS_EC_FACTOR (0.67)

Usage
-----
    python prediction/1_generate_data.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ── Chemins ─────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
(BASE / "data").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE.parent / "src"))

from config import TDS_EC_FACTOR

# ── Reproductibilité ────────────────────────────────────────────────────────
np.random.seed(42)

# ── Paramètres ──────────────────────────────────────────────────────────────
N_JOURS       = 120
FREQ_MINUTES  = 60

n_points = N_JOURS * 24 * (60 // FREQ_MINUTES)
debut     = datetime(2024, 1, 1)
timestamps = [debut + timedelta(minutes=i * FREQ_MINUTES) for i in range(n_points)]
heures     = np.array([t.hour + t.minute / 60 for t in timestamps])
jours      = np.array([t.timetuple().tm_yday  for t in timestamps])


def evenement_progressif(n, idx, duree, intensite):
    """Signal trapézoïdal : montée douce → plateau → descente."""
    signal  = np.zeros(n)
    montee  = duree // 3
    plateau = duree // 3
    descente = duree - montee - plateau
    for k in range(montee):
        if idx + k < n:
            signal[idx + k] = intensite * (k / montee)
    for k in range(plateau):
        if idx + montee + k < n:
            signal[idx + montee + k] = intensite
    for k in range(descente):
        if idx + montee + plateau + k < n:
            signal[idx + montee + plateau + k] = intensite * (1 - k / descente)
    return signal


# ── Température ─────────────────────────────────────────────────────────────
temperature = (
    20
    + 4 * np.sin(2 * np.pi * (heures - 6) / 24)   # cycle diurne (pic 14h)
    + 3 * np.sin(2 * np.pi * jours / 365)           # cycle saisonnier
    + np.random.normal(0, 0.3, n_points)
)
for _ in range(3):
    idx = np.random.randint(0, n_points - 72)
    temperature += evenement_progressif(n_points, idx, 48, np.random.uniform(4, 7))

# ── Pluie (0–1, influence pH, turbidité, conductivité) ──────────────────────
pluie = np.zeros(n_points)
for _ in range(15):
    idx      = np.random.randint(0, n_points - 20)
    duree    = np.random.randint(4, 16)
    intensite = np.random.uniform(0.3, 0.7)
    pluie   += evenement_progressif(n_points, idx, duree, intensite)
pluie = np.clip(pluie, 0, 1)

# ── pH ───────────────────────────────────────────────────────────────────────
ph = (
    7.2
    - 0.05 * (temperature - 20)   # hausse temp → légère acidification
    - 0.2  * pluie                # pluie → pH plus bas
    + np.random.normal(0, 0.1, n_points)
)
for _ in range(3):
    idx = np.random.randint(0, n_points - 24)
    ph -= evenement_progressif(n_points, idx, 20, np.random.uniform(1.0, 2.0))
idx = np.random.randint(0, n_points - 24)
ph += evenement_progressif(n_points, idx, 16, np.random.uniform(1.0, 1.8))

# ── Turbidité [NTU] ──────────────────────────────────────────────────────────
turbidity = (
    2
    + 5 * pluie
    + np.random.exponential(0.3, n_points)
)
for _ in range(3):
    idx = np.random.randint(0, n_points - 24)
    turbidity += evenement_progressif(n_points, idx, 18, np.random.uniform(4, 8))

# ── Conductivité [µS/cm] ─────────────────────────────────────────────────────
conductivity = (
    600
    + 2  * (temperature - 20)
    + 30 * pluie
    + np.random.normal(0, 10, n_points)
)
for _ in range(3):
    idx = np.random.randint(0, n_points - 24)
    conductivity += evenement_progressif(n_points, idx, 20, np.random.uniform(200, 400))

# ── Solids [mg/L] = TDS, dérivé de la conductivité ──────────────────────────
# Aligné avec config.py : Solids ≈ Conductivity × TDS_EC_FACTOR
solids = conductivity * TDS_EC_FACTOR + np.random.normal(0, 5, n_points)
for _ in range(3):
    idx = np.random.randint(0, n_points - 24)
    solids += evenement_progressif(n_points, idx, 20, np.random.uniform(100, 250))

# ── Clipping aux bornes physiques (alignées avec data_processing.py) ─────────
ph           = np.clip(ph,           0.0,     14.0)
solids       = np.clip(solids,       0.0,  3000.0)
turbidity    = np.clip(turbidity,    0.0,    100.0)
conductivity = np.clip(conductivity, 0.0,   3500.0)
temperature  = np.clip(temperature,  0.0,     50.0)

# ── Sauvegarde ───────────────────────────────────────────────────────────────
df = pd.DataFrame({
    "timestamp":   timestamps,
    "ph":          np.round(ph,           3),
    "Solids":      np.round(solids,       2),
    "Conductivity":np.round(conductivity, 2),
    "Turbidity":   np.round(turbidity,    3),
    "Temperature": np.round(temperature,  2),
})
df.to_csv(BASE / "data/dataset.csv", index=False)

print(f"✅ Dataset : {len(df)} points sur {N_JOURS} jours (intervalle {FREQ_MINUTES} min)")
print(f"\nStatistiques :")
print(df[["ph", "Solids", "Conductivity", "Turbidity", "Temperature"]].describe().round(2))
