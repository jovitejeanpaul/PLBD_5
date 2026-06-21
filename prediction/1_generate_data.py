import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
n_jours = 120
freq_minutes = 60
n_points = n_jours * 24 * (60 // freq_minutes)

debut = datetime(2024, 1, 1)
timestamps = [debut + timedelta(minutes=i * freq_minutes) for i in range(n_points)]
heures = np.array([t.hour + t.minute / 60 for t in timestamps])
jours  = np.array([t.timetuple().tm_yday for t in timestamps])

def evenement_progressif(n_points, idx, duree, intensite):
    signal = np.zeros(n_points)
    montee  = duree // 3
    plateau = duree // 3
    descente = duree - montee - plateau
    for k in range(montee):
        if idx + k < n_points:
            signal[idx + k] = intensite * (k / montee)
    for k in range(plateau):
        if idx + montee + k < n_points:
            signal[idx + montee + k] = intensite
    for k in range(descente):
        if idx + montee + plateau + k < n_points:
            signal[idx + montee + plateau + k] = intensite * (1 - k / descente)
    return signal

# ── Température ───────────────────────────────────────────────────────────────
temperature = (
    20
    + 4 * np.sin(2 * np.pi * (heures - 6) / 24)
    + 3 * np.sin(2 * np.pi * jours / 365)
    + np.random.normal(0, 0.3, n_points)
)
for _ in range(3):
    idx = np.random.randint(0, n_points - 72)
    temperature += evenement_progressif(n_points, idx, 48, np.random.uniform(4, 7))

# ── Pluie ─────────────────────────────────────────────────────────────────────
pluie = np.zeros(n_points)
for _ in range(15):
    idx = np.random.randint(0, n_points - 20)
    duree = np.random.randint(4, 16)
    intensite = np.random.uniform(0.3, 0.7)
    pluie += evenement_progressif(n_points, idx, duree, intensite)
pluie = np.clip(pluie, 0, 1)

# ── pH ────────────────────────────────────────────────────────────────────────
ph = (
    7.2
    - 0.05 * (temperature - 20)
    - 0.2  * pluie
    + np.random.normal(0, 0.1, n_points)
)
for _ in range(3):
    idx = np.random.randint(0, n_points - 24)
    ph -= evenement_progressif(n_points, idx, 20, np.random.uniform(1.0, 2.0))
idx = np.random.randint(0, n_points - 24)
ph += evenement_progressif(n_points, idx, 16, np.random.uniform(1.0, 1.8))

# ── Turbidité ─────────────────────────────────────────────────────────────────
turbidite = (
    2
    + 5 * pluie
    + np.random.exponential(0.3, n_points)
)
for _ in range(3):
    idx = np.random.randint(0, n_points - 24)
    turbidite += evenement_progressif(n_points, idx, 18, np.random.uniform(4, 8))

# ── Conductivité ──────────────────────────────────────────────────────────────
conductivite = (
    600
    + 2  * (temperature - 20)
    + 30 * pluie
    + np.random.normal(0, 10, n_points)
)
for _ in range(3):
    idx = np.random.randint(0, n_points - 24)
    conductivite += evenement_progressif(n_points, idx, 20, np.random.uniform(200, 400))

# ── TDS ───────────────────────────────────────────────────────────────────────
tds = conductivite * 0.5 + np.random.normal(0, 5, n_points)
for _ in range(3):
    idx = np.random.randint(0, n_points - 24)
    tds += evenement_progressif(n_points, idx, 20, np.random.uniform(200, 400))

# ── Clipping ──────────────────────────────────────────────────────────────────
ph           = np.clip(ph,           4.0,  11.0)
tds          = np.clip(tds,          50,   1000)
turbidite    = np.clip(turbidite,    0,    15)
conductivite = np.clip(conductivite, 200,  1200)
temperature  = np.clip(temperature,  10,   35)

# ── Sauvegarde ────────────────────────────────────────────────────────────────
df = pd.DataFrame({
    'timestamp':    timestamps,
    'pH':           np.round(ph,           2),
    'TDS':          np.round(tds,          2),
    'turbidite':    np.round(turbidite,    2),
    'conductivite': np.round(conductivite, 2),
    'temperature':  np.round(temperature,  2),
})

df.to_csv('data/dataset.csv', index=False)
print(f"✅ Dataset : {len(df)} points sur {n_jours} jours (intervalle {freq_minutes} min)")
print(f"\nStatistiques :")
print(df[['pH','TDS','turbidite','conductivite','temperature']].describe().round(2))