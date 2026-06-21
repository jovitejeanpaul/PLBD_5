import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

# Chargement
df = pd.read_csv('data/dataset.csv', parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

features = ['pH', 'TDS', 'turbidite', 'conductivite', 'temperature']

# Normalisation entre 0 et 1
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])

# Création des séquences pour le LSTM
# Fenêtre : 10 mesures passées → prédire les 48 suivantes (24h)
WINDOW_SIZE = 10
HORIZON = 24

X, y = [], []
for i in range(len(df_scaled) - WINDOW_SIZE - HORIZON):
    X.append(df_scaled[features].iloc[i:i+WINDOW_SIZE].values)
    y.append(df_scaled[features].iloc[i+WINDOW_SIZE:i+WINDOW_SIZE+HORIZON].values)

X = np.array(X)
y = np.array(y)

# Sauvegarde
np.save('data/X.npy', X)
np.save('data/y.npy', y)
with open('data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"✅ Prétraitement terminé")
print(f"X shape : {X.shape}  →  (nb_séquences, fenêtre, nb_capteurs)")
print(f"y shape : {y.shape}  →  (nb_séquences, horizon, nb_capteurs)")