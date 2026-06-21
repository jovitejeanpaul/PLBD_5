import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Définition du modèle (identique à 3_train_model.py)
class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=5, horizon=24):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size * horizon)
        self.horizon = horizon
        self.output_size = output_size

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.view(-1, self.horizon, self.output_size)

# Chargement
model = LSTMModel()
model.load_state_dict(torch.load('models/lstm_model.pth'))
model.eval()

X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

with open('data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prédiction
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_test)
    y_pred = model(X_tensor).numpy()

# Dénormalisation
def denormalize(data, scaler):
    shape = data.shape
    return scaler.inverse_transform(data.reshape(-1, 5)).reshape(shape)

y_test_real = denormalize(y_test, scaler)
y_pred_real = denormalize(y_pred, scaler)

# Métriques
features = ['pH', 'TDS', 'turbidite', 'conductivite', 'temperature']
print("📊 Métriques de performance :\n")
for i, f in enumerate(features):
    mae = mean_absolute_error(y_test_real[:,:,i].flatten(), y_pred_real[:,:,i].flatten())
    rmse = np.sqrt(mean_squared_error(y_test_real[:,:,i].flatten(), y_pred_real[:,:,i].flatten()))
    print(f"{f:15s} → MAE: {mae:.4f} | RMSE: {rmse:.4f}")

# Graphiques
fig, axes = plt.subplots(5, 1, figsize=(12, 14))
fig.suptitle('Prévision vs Réalité — 24h', fontsize=14)

for i, f in enumerate(features):
    axes[i].plot(y_test_real[0,:,i], label='Réel', color='blue')
    axes[i].plot(y_pred_real[0,:,i], label='Prédit', color='red', linestyle='--')
    axes[i].set_title(f)
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.savefig('data/evaluation.png')
plt.show()
print("\n✅ Graphiques sauvegardés dans data/evaluation.png")