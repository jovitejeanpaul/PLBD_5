"""
model.py
=========
Définition unique du modèle LSTM de prévision.

Importé par 3_train_model.py, 4_evaluate.py et 5_dashboard.py
pour éviter toute désynchronisation de l'architecture.

Features (5 capteurs, noms alignés avec le module de diagnostic) :
    ph · Solids · Conductivity · Turbidity · temperature
"""

from __future__ import annotations

import torch.nn as nn

# ── Constantes partagées ────────────────────────────────────────────────────
FEATURES     : list[str] = ["ph", "Solids", "Conductivity", "Turbidity", "Temperature"]
N_FEATURES   : int       = len(FEATURES)   # 5
WINDOW_SIZE  : int       = 24              # heures passées utilisées comme entrée
HORIZON      : int       = 24              # heures à prédire en avance


# ── Modèle ──────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    """
    LSTM bi-couche pour la prévision multi-capteurs à horizon 24h.

    Architecture
    ------------
    Entrée  : (batch, WINDOW_SIZE, N_FEATURES)
    LSTM    : hidden_size=64, num_layers=2, dropout=0.2
    FC      : hidden_size → N_FEATURES × HORIZON
    Sortie  : (batch, HORIZON, N_FEATURES)

    Parameters
    ----------
    input_size  : int  — nombre de features en entrée (défaut : N_FEATURES)
    hidden_size : int  — dimension de l'état caché (défaut : 64)
    num_layers  : int  — nombre de couches LSTM empilées (défaut : 2)
    output_size : int  — nombre de features en sortie (défaut : N_FEATURES)
    horizon     : int  — nombre de pas de temps à prédire (défaut : HORIZON)
    """

    def __init__(
        self,
        input_size:  int = N_FEATURES,
        hidden_size: int = 64,
        num_layers:  int = 2,
        output_size: int = N_FEATURES,
        horizon:     int = HORIZON,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.2,
        )
        self.fc      = nn.Linear(hidden_size, output_size * horizon)
        self.horizon = horizon
        self.output_size = output_size

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.view(-1, self.horizon, self.output_size)
