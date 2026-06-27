"""
4_evaluate.py
==============
Évaluation du LSTM de prévision sur le jeu de test :
    - MAE / RMSE / R² par feature (valeurs dénormalisées)
    - Graphiques prévision vs réel
    - Accuracy de la classe de potabilité à t+24h
      (intégration avec le modèle de diagnostic CatBoost de src/)

Usage
-----
    python prediction/4_evaluate.py
"""

from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import FEATURES, HORIZON, LSTMModel, N_FEATURES, WINDOW_SIZE

# ── Chemins ─────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
OUTPUTS_DIR = BASE.parent / "outputs" / "models"   # src/ pipeline

# ── Chargement du modèle LSTM ────────────────────────────────────────────────
model = LSTMModel()
model.load_state_dict(
    torch.load(BASE / "models/lstm_model.pth", map_location="cpu")
)
model.eval()

# ── Chargement du scaler et des données de test ──────────────────────────────
scaler = joblib.load(BASE / "data/scaler.joblib")
X_test = np.load(BASE / "data/X_test.npy")
y_test = np.load(BASE / "data/y_test.npy")

# ── Prédictions LSTM ─────────────────────────────────────────────────────────
with torch.no_grad():
    y_pred = model(torch.FloatTensor(X_test)).numpy()   # (n, HORIZON, N_FEATURES)

# ── Dénormalisation ──────────────────────────────────────────────────────────
def denormalize(data: np.ndarray) -> np.ndarray:
    """data : (n, HORIZON, N_FEATURES) → retourne dans les unités physiques."""
    n, h, f = data.shape
    return scaler.inverse_transform(data.reshape(-1, f)).reshape(n, h, f)

y_test_real = denormalize(y_test)
y_pred_real = denormalize(y_pred)

# ── Métriques par feature ────────────────────────────────────────────────────
print("=" * 60)
print("  MÉTRIQUES DE PERFORMANCE DU LSTM (jeu de test)")
print(f"  Fenêtre : {WINDOW_SIZE}h → Horizon : {HORIZON}h")
print("=" * 60)

for i, feat in enumerate(FEATURES):
    true_flat = y_test_real[:, :, i].flatten()
    pred_flat = y_pred_real[:, :, i].flatten()
    mae  = mean_absolute_error(true_flat, pred_flat)
    rmse = np.sqrt(mean_squared_error(true_flat, pred_flat))
    r2   = r2_score(true_flat, pred_flat)
    print(f"  {feat:<14}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

print("=" * 60)

# ── Intégration avec le modèle de diagnostic (optionnel) ─────────────────────
# Les 4 features diagnostiques sont communes aux deux pipelines :
DIAG_FEATURES = ["ph", "Solids", "Conductivity", "Turbidity"]
DIAG_IDX      = [FEATURES.index(f) for f in DIAG_FEATURES]

diag_model_path = next(OUTPUTS_DIR.glob("model_1_*.joblib"), None) if OUTPUTS_DIR.exists() else None
diag_scaler_path = OUTPUTS_DIR / "scaler.joblib" if OUTPUTS_DIR.exists() else None

if diag_model_path and diag_scaler_path and diag_scaler_path.exists():
    try:
        import sys
        sys.path.insert(0, str(BASE.parent / "src"))
        from threshold_classifier import ThresholdClassifier  # noqa: F401

        diag_model  = joblib.load(diag_model_path)
        diag_scaler = joblib.load(diag_scaler_path)

        # Extraire les 4 features diagnostiques depuis les prédictions à t+24h
        # On prend le dernier pas de l'horizon (t+24h)
        X_diag_pred = y_pred_real[:, -1, :][:, DIAG_IDX]   # (n, 4)
        X_diag_true = y_test_real[:, -1, :][:, DIAG_IDX]

        X_pred_scaled = diag_scaler.transform(X_diag_pred)
        X_true_scaled = diag_scaler.transform(X_diag_true)

        y_class_pred  = diag_model.predict(X_pred_scaled)
        y_class_true  = diag_model.predict(X_true_scaled)

        accuracy = (y_class_pred == y_class_true).mean()
        print(f"\n  Potabilité t+24h :")
        print(f"    Accuracy (prédit vs réel)  : {accuracy:.3f}")
        print(f"    Non potable prédit (%)     : {y_class_pred.mean()*100:.1f}%")
        print(f"    Non potable réel (%)       : {y_class_true.mean()*100:.1f}%")
        print("=" * 60)
    except Exception as e:
        print(f"\n  ⚠️  Modèle diagnostic non disponible ({e})")
        print("     Lancez d'abord train_model.py dans src/ pour activer cette section.")
else:
    print(f"\n  ℹ️  Modèle diagnostic non trouvé dans {OUTPUTS_DIR}")
    print("     Lancez d'abord train_model.py dans src/ pour activer cette section.")

# ── Graphiques prévision vs réel ─────────────────────────────────────────────
fig, axes = plt.subplots(N_FEATURES, 1, figsize=(12, 3 * N_FEATURES))
fig.suptitle(f"Prévision LSTM vs Réalité — horizon {HORIZON}h\n(première séquence du test)", fontsize=13)

for i, (ax, feat) in enumerate(zip(axes, FEATURES)):
    ax.plot(y_test_real[0, :, i], label="Réel",  color="steelblue", lw=1.8)
    ax.plot(y_pred_real[0, :, i], label="Prédit", color="tomato",    lw=1.5, linestyle="--")
    ax.set_title(feat, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlabel("Heure")

plt.tight_layout()
out_fig = BASE / "data/evaluation.png"
fig.savefig(out_fig, dpi=120, bbox_inches="tight")
print(f"\n✅ Graphique sauvegardé → {out_fig}")
