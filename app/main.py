"""
main.py
========
Backend FastAPI du système AQUA-SENS.

Endpoints
---------
GET  /              → Page HTML principale (interface web)
WS   /ws            → WebSocket : push diagnostic toutes les REFRESH_S secondes
GET  /api/forecast  → Prévision LSTM 24h + alertes NM 03.7.001
GET  /api/shap      → Résumé SHAP pré-calculé (importance features)
GET  /api/status    → État du système (modèles chargés, mode mock, etc.)

Lancement
---------
    # Depuis la racine du projet
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

    # Production Pi (sans reload)
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Résolution des chemins ───────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
SRC_DIR    = ROOT / "src"
PRED_DIR   = ROOT / "prediction"
STATIC_DIR = Path(__file__).parent / "static"

for p in (str(ROOT), str(SRC_DIR), str(PRED_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aquasens")

# ── Configuration ────────────────────────────────────────────────────────────
REFRESH_S       : int   = 5      # intervalle entre deux lectures capteurs (secondes)
HISTORY_MAXLEN  : int   = 24     # taille du buffer pour la prévision LSTM
NM_STANDARDS    : dict  = {      # NM 03.7.001 — seuils marocains
    "ph":           {"min": 6.5,  "max": 8.5,   "unit": ""},
    "Solids":       {"min": 0,    "max": 1500,   "unit": "mg/L"},
    "Conductivity": {"min": 0,    "max": 2700,   "unit": "µS/cm"},
    "Turbidity":    {"min": 0,    "max": 5,      "unit": "NTU"},
    "temperature":  {"min": 0,    "max": 25,     "unit": "°C"},
}
DIAG_FEATURES   : list  = ["ph", "Solids", "Conductivity", "Turbidity"]

# ── Chargement des modules IA (avec fallback gracieux) ────────────────────────
_STATUS: dict[str, Any] = {
    "diagnostic_model": False,
    "forecast_model":   False,
    "shap_available":   False,
    "alert_engine":     False,
    "mock_mode":        True,
    "errors":           [],
}

pipeline     = None
alert_engine = None
lstm_model   = None
lstm_scaler  = None

try:
    from src.threshold_classifier import ThresholdClassifier  # noqa: F401 — nécessaire pour joblib
    from src.sensor_inference import SensorPipeline
    pipeline = SensorPipeline.from_saved_models(
        models_dir=ROOT / "outputs" / "models",
        mock=True,  # passe en False quand les capteurs sont branchés
    )
    _STATUS["diagnostic_model"] = True
    _STATUS["mock_mode"] = not pipeline.reader.__class__.__name__ == "ADS1115Reader"
    logger.info("Modèle de diagnostic chargé (mock=%s)", _STATUS["mock_mode"])
except Exception as e:
    _STATUS["errors"].append(f"Diagnostic : {e}")
    logger.warning("Modèle de diagnostic non disponible : %s", e)

try:
    from prediction.alerte_engine import AlertEngine
    alert_engine = AlertEngine()
    _STATUS["alert_engine"] = True
    logger.info("Moteur d'alertes chargé")
except Exception as e:
    _STATUS["errors"].append(f"AlertEngine : {e}")

try:
    import joblib, torch
    from prediction.model import LSTMModel, FEATURES as FORECAST_FEATURES, HORIZON, WINDOW_SIZE
    lstm_model = LSTMModel()
    lstm_model.load_state_dict(
        torch.load(PRED_DIR / "models/lstm_model.pth", map_location="cpu")
    )
    lstm_model.eval()
    lstm_scaler  = joblib.load(PRED_DIR / "data/scaler.joblib")
    _STATUS["forecast_model"] = True
    logger.info("Modèle LSTM chargé (fenêtre=%dh, horizon=%dh)", WINDOW_SIZE, HORIZON)
except Exception as e:
    _STATUS["errors"].append(f"LSTM : {e}")
    logger.warning("Modèle LSTM non disponible : %s", e)

shap_summary = None
try:
    shap_path = ROOT / "outputs" / "reports" / "shap_summary.json"
    if shap_path.exists():
        shap_summary = json.loads(shap_path.read_text(encoding="utf-8"))
        _STATUS["shap_available"] = True
        logger.info("Résumé SHAP chargé")
except Exception as e:
    _STATUS["errors"].append(f"SHAP : {e}")

# ── Buffer historique (alimenté par chaque lecture capteur) ───────────────────
_history: deque = deque(maxlen=HISTORY_MAXLEN)

# ── Connexions WebSocket actives ─────────────────────────────────────────────
_ws_clients: list[WebSocket] = []

# ── Application FastAPI ───────────────────────────────────────────────────────
app = FastAPI(
    title="AQUA-SENS",
    description="Système intelligent de diagnostic et de prévision de la qualité de l'eau",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ════════════════════════════════════════════════════════════════════════════
# PAGE PRINCIPALE
# ════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ════════════════════════════════════════════════════════════════════════════
# WEBSOCKET — Diagnostic temps réel
# ════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info("Client WebSocket connecté (total: %d)", len(_ws_clients))
    try:
        while True:
            data = _run_diagnostic()
            await websocket.send_json(data)
            await asyncio.sleep(REFRESH_S)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("Erreur WebSocket : %s", e)
    finally:
        _ws_clients.remove(websocket)
        logger.info("Client WebSocket déconnecté (total: %d)", len(_ws_clients))


def _run_diagnostic() -> dict:
    """Exécute une lecture capteurs + inférence et retourne le résultat JSON."""
    if pipeline is None:
        # Mode dégradé : retourner des données simulées sans modèle
        return {
            "type":              "diagnostic",
            "error":             "Modèle de diagnostic non chargé",
            "raw_values":        {f: 0.0 for f in DIAG_FEATURES},
            "potability_now":    None,
            "potability_label":  "Indisponible",
            "confidence_proba":  None,
            "out_of_bounds":     [],
            "inference_time_ms": 0,
            "timestamp":         _now(),
        }

    result = pipeline.run_once()

    # Stocker dans le buffer pour alimenter le LSTM
    _history.append({f: result["raw_values"].get(f, 0.0) for f in DIAG_FEATURES})

    return {**result, "type": "diagnostic"}


# ════════════════════════════════════════════════════════════════════════════
# REST — Prévision 24h + Alertes
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/forecast")
async def get_forecast():
    if lstm_model is None or lstm_scaler is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Modèle de prévision non disponible. "
                               "Lancez prediction/3_train_model.py d'abord."},
        )

    try:
        import torch
        from prediction.model import FEATURES as FC_FEATURES, HORIZON, WINDOW_SIZE

        # ── Construire la séquence d'entrée ──────────────────────────────
        if len(_history) >= WINDOW_SIZE:
            history_arr = np.array(
                [[h.get(f, 0.0) for f in FC_FEATURES if f in h or True]
                 for h in list(_history)[-WINDOW_SIZE:]],
                dtype=np.float32,
            )
            # S'assurer qu'on a bien N_FEATURES colonnes (ajouter temp=0 si absente)
            if history_arr.shape[1] < len(FC_FEATURES):
                pad = np.zeros((WINDOW_SIZE, len(FC_FEATURES) - history_arr.shape[1]),
                                dtype=np.float32)
                history_arr = np.hstack([history_arr, pad])
        else:
            # Buffer pas encore plein : générer un historique par défaut
            rng = np.random.default_rng(42)
            means = [7.2, 18500, 420, 3.8, 22.0]
            history_arr = np.array(
                [means for _ in range(WINDOW_SIZE)], dtype=np.float32
            ) + rng.normal(0, 0.01, (WINDOW_SIZE, len(FC_FEATURES))).astype(np.float32)

        # ── Inférence LSTM ───────────────────────────────────────────────
        t0     = time.perf_counter()
        scaled = lstm_scaler.transform(history_arr)
        x      = torch.FloatTensor(scaled).unsqueeze(0)
        with torch.no_grad():
            y_scaled = lstm_model(x).squeeze(0).numpy()   # (HORIZON, N_FEATURES)
        predicted = lstm_scaler.inverse_transform(y_scaled)  # (HORIZON, N_FEATURES)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # ── Alertes ──────────────────────────────────────────────────────
        alerts_data = []
        if alert_engine is not None:
            # On passe uniquement les features communes avec le moteur d'alertes
            alerts = alert_engine.analyze(predicted, FC_FEATURES, horizon_h=HORIZON)
            alerts_data = [a.to_dict() for a in alerts]

        # ── Formater la réponse ──────────────────────────────────────────
        predictions = {
            feat: [round(float(predicted[h, j]), 4) for h in range(HORIZON)]
            for j, feat in enumerate(FC_FEATURES)
        }
        return {
            "features":       FC_FEATURES,
            "hours":          list(range(1, HORIZON + 1)),
            "predictions":    predictions,
            "alerts":         alerts_data,
            "n_alerts":       {
                "CRITICAL": sum(1 for a in alerts_data if a["level"] == "CRITICAL"),
                "WARNING":  sum(1 for a in alerts_data if a["level"] == "WARNING"),
                "INFO":     sum(1 for a in alerts_data if a["level"] == "INFO"),
            },
            "inference_ms":   round(elapsed_ms, 2),
            "timestamp":      _now(),
            "buffer_size":    len(_history),
        }

    except Exception as e:
        logger.error("Erreur prévision : %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


# ════════════════════════════════════════════════════════════════════════════
# REST — SHAP
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/shap")
async def get_shap():
    if shap_summary is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Résumé SHAP non disponible. "
                               "Lancez src/explainability.py d'abord."},
        )
    return shap_summary


# ════════════════════════════════════════════════════════════════════════════
# REST — Statut système
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/status")
async def get_status():
    return {
        **_STATUS,
        "ws_clients":     len(_ws_clients),
        "history_size":   len(_history),
        "refresh_s":      REFRESH_S,
        "timestamp":      _now(),
    }


# ════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ════════════════════════════════════════════════════════════════════════════

def _now() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)