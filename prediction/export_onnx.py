"""
export_onnx.py
===============
Exporte le modèle LSTM PyTorch en format ONNX pour le déploiement
sur Raspberry Pi (onnxruntime, sans PyTorch).

À exécuter sur PC après l'entraînement (3_train_model.py).

Usage
-----
    python prediction/export_onnx.py
"""

from pathlib import Path

import torch

from model import LSTMModel, N_FEATURES, WINDOW_SIZE

BASE = Path(__file__).parent

model = LSTMModel()
model.load_state_dict(
    torch.load(BASE / "models/lstm_model.pth", map_location="cpu")
)
model.eval()

dummy_input = torch.randn(1, WINDOW_SIZE, N_FEATURES)

onnx_path = BASE / "models/lstm_model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    str(onnx_path),
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input":  {0: "batch"},
        "output": {0: "batch"},
    },
    opset_version=14,
)

print(f"Modèle ONNX exporté → {onnx_path}")
print(f"  Taille : {onnx_path.stat().st_size / 1024:.1f} Ko")
