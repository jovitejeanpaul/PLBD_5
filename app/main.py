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
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
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
    "Temperature":  {"min": 0,    "max": 25,     "unit": "°C"},
}
DIAG_FEATURES   : list  = ["ph", "Solids", "Conductivity", "Turbidity"]
FORECAST_FEATURES_FULL : list = ["ph", "Solids", "Conductivity", "Turbidity", "Temperature"]

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
        mock=False,  # passe en False quand les capteurs sont branchés
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

filter_controller = None
try:
    from src.filter_controller import FilterController
    filter_controller = FilterController(mock=_STATUS.get("mock_mode", True))
    _STATUS["filter_controller"] = True
    logger.info("FilterController chargé (mock=%s)", filter_controller.mock)
except Exception as e:
    _STATUS["filter_controller"] = False
    _STATUS["errors"].append(f"FilterController : {e}")
    logger.warning("FilterController non disponible : %s", e)

_lstm_backend = None   # "torch" ou "onnx"

try:
    import joblib
    from prediction.model import FEATURES as FORECAST_FEATURES, HORIZON, WINDOW_SIZE

    lstm_scaler = joblib.load(PRED_DIR / "data/scaler.joblib")

    # Priorité 1 : ONNX Runtime (léger, compatible ARM64 Pi)
    onnx_path = PRED_DIR / "models/lstm_model.onnx"
    if onnx_path.exists():
        import onnxruntime as ort
        lstm_model = ort.InferenceSession(str(onnx_path))
        _lstm_backend = "onnx"
        _STATUS["forecast_model"] = True
        logger.info("LSTM chargé via ONNX Runtime (fenêtre=%dh, horizon=%dh)", WINDOW_SIZE, HORIZON)
    else:
        # Priorité 2 : PyTorch (PC de développement)
        import torch
        from prediction.model import LSTMModel
        lstm_model = LSTMModel()
        lstm_model.load_state_dict(
            torch.load(PRED_DIR / "models/lstm_model.pth", map_location="cpu")
        )
        lstm_model.eval()
        _lstm_backend = "torch"
        _STATUS["forecast_model"] = True
        logger.info("LSTM chargé via PyTorch (fenêtre=%dh, horizon=%dh)", WINDOW_SIZE, HORIZON)

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

# ── Rapport d'évaluation des modèles ─────────────────────────────────────────
_models_report: list[dict] = []
_models_dir = ROOT / "outputs" / "models"
try:
    import pandas as _pd
    _eval_path = ROOT / "outputs" / "reports" / "evaluation_report.csv"
    if _eval_path.exists():
        _df = _pd.read_csv(_eval_path)
        _cols = ["model_key", "rank", "composite_score", "roc_auc_test",
                 "gmean_test", "f1_test", "recall_test", "precision_test",
                 "accuracy_test", "mcc_test", "pr_auc_test", "threshold"]
        _cols = [c for c in _cols if c in _df.columns]
        _models_report = _df[_cols].to_dict(orient="records")
        logger.info("Rapport d'évaluation chargé : %d modèles", len(_models_report))
except Exception as e:
    logger.warning("Rapport d'évaluation non disponible : %s", e)

_active_model_key: str = _models_report[0]["model_key"] if _models_report else "unknown"

# ── Buffers ──────────────────────────────────────────────────────────────────
# Buffer brut : accumule les mesures entre deux échantillonnages horaires.
# Toutes les mesures du diagnostic temps réel (5s) y entrent.
_raw_buffer: list[dict] = []

# Buffer horaire : 24 médianes horaires pour alimenter le LSTM.
# Une entrée = la médiane des mesures des 5 dernières minutes, stockée 1×/heure.
_hourly_buffer: deque = deque(maxlen=HISTORY_MAXLEN)

# Timestamp du dernier échantillonnage horaire
_last_hourly_sample: float = 0.0

# Intervalle entre deux échantillonnages horaires (secondes)
HOURLY_INTERVAL: float = 3600.0

# Fenêtre de mesures pour calculer la médiane (secondes)
MEDIAN_WINDOW: float = 300.0

# ── Connexions WebSocket actives ─────────────────────────────────────────────
_ws_clients: list[WebSocket] = []

# ── Application FastAPI ───────────────────────────────────────────────────────
from app.auth import router as auth_router, get_current_user, require_admin
from app.database import (
    initialize_database,
    save_diagnostic,
    save_filtration,
    save_forecast,
    get_history_diagnostic,
    get_history_filtration,
    get_history_forecast,
)

app = FastAPI(
    title="AQUA-SENS",
    description="Système intelligent de diagnostic et de prévision de la qualité de l'eau",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(auth_router)


@app.on_event("startup")
def startup():
    initialize_database()


# ════════════════════════════════════════════════════════════════════════════
# PAGE PRINCIPALE
# ════════════════════════════════════════════════════════════════════════════

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return HTMLResponse(content=(STATIC_DIR / "login.html").read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    try:
        get_current_user(request)
    except Exception:
        return RedirectResponse("/login", status_code=302)
    return HTMLResponse(content=(STATIC_DIR / "index.html").read_text(encoding="utf-8"))


# ════════════════════════════════════════════════════════════════════════════
# WEBSOCKET — Diagnostic temps réel
# ════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    token = websocket.cookies.get("access_token")
    if not token:
        await websocket.close(code=4001, reason="Non authentifié")
        return
    from app.security import decode_access_token
    if decode_access_token(token) is None:
        await websocket.close(code=4001, reason="Session expirée")
        return

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


def _update_buffers(raw_values: dict) -> None:
    """
    Gère les deux niveaux de buffer :
    1. Accumule chaque mesure dans _raw_buffer (5s)
    2. Toutes les HOURLY_INTERVAL secondes, calcule la médiane des
       MEDIAN_WINDOW dernières secondes et l'ajoute à _hourly_buffer
    """
    global _last_hourly_sample

    now = time.time()
    entry = {f: raw_values.get(f, 0.0) for f in FORECAST_FEATURES_FULL}
    entry["_ts"] = now
    _raw_buffer.append(entry)

    elapsed = now - _last_hourly_sample
    if elapsed >= HOURLY_INTERVAL:
        # Filtrer les mesures des MEDIAN_WINDOW dernières secondes
        cutoff = now - MEDIAN_WINDOW
        recent = [e for e in _raw_buffer if e["_ts"] >= cutoff]

        if recent:
            median_entry = {}
            for f in FORECAST_FEATURES_FULL:
                values = [e[f] for e in recent]
                values.sort()
                n = len(values)
                median_entry[f] = values[n // 2] if n % 2 == 1 else (values[n // 2 - 1] + values[n // 2]) / 2
            _hourly_buffer.append(median_entry)
            logger.info(
                "Buffer horaire : médiane de %d mesures stockée (%d/24 échantillons)",
                len(recent), len(_hourly_buffer),
            )
        else:
            _hourly_buffer.append(entry)

        _last_hourly_sample = now
        # Purger le buffer brut (garder seulement la fenêtre de médiane)
        _raw_buffer[:] = [e for e in _raw_buffer if e["_ts"] >= cutoff]


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

    # Accumuler les mesures dans le buffer brut pour la médiane horaire
    _update_buffers(result["raw_values"])

    # Décision de filtration basée sur les valeurs capteurs
    filter_decision = None
    if filter_controller is not None:
        try:
            decision = filter_controller.decide(result, shap_summary=shap_summary)
            filter_decision = decision.to_dict()
        except Exception as e:
            logger.error("Erreur décision filtration : %s", e)

    # Sauvegarder le diagnostic en base
    try:
        save_diagnostic(result)
    except Exception as e:
        logger.error("Erreur sauvegarde diagnostic : %s", e)

    return {**result, "type": "diagnostic", "filter_decision": filter_decision}


# ════════════════════════════════════════════════════════════════════════════
# REST — Prévision 24h + Alertes
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/forecast")
async def get_forecast(request: Request):
    try:
        get_current_user(request)
    except Exception:
        return JSONResponse(status_code=401, content={"error": "Non authentifié"})
    if lstm_model is None or lstm_scaler is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Modèle de prévision non disponible. "
                               "Lancez prediction/3_train_model.py d'abord."},
        )

    try:
        from prediction.model import FEATURES as FC_FEATURES, HORIZON, WINDOW_SIZE

        # ── Construire la séquence d'entrée (24 médianes horaires) ────
        if len(_hourly_buffer) >= WINDOW_SIZE:
            history_arr = np.array(
                [[h.get(f, 0.0) for f in FC_FEATURES]
                 for h in list(_hourly_buffer)[-WINDOW_SIZE:]],
                dtype=np.float32,
            )
        else:
            hours_remaining = WINDOW_SIZE - len(_hourly_buffer)
            return JSONResponse(
                status_code=503,
                content={
                    "error": f"Buffer horaire insuffisant : {len(_hourly_buffer)}/{WINDOW_SIZE} échantillons. "
                             f"Encore ~{hours_remaining}h de mesures nécessaires.",
                    "buffer_size": len(_hourly_buffer),
                    "required": WINDOW_SIZE,
                },
            )

        # ── Inférence LSTM (ONNX ou PyTorch) ────────────────────────────
        t0     = time.perf_counter()
        scaled = lstm_scaler.transform(history_arr)

        if _lstm_backend == "onnx":
            x = scaled.reshape(1, WINDOW_SIZE, len(FC_FEATURES)).astype(np.float32)
            y_scaled = lstm_model.run(None, {"input": x})[0].squeeze(0)
        else:
            import torch
            x = torch.FloatTensor(scaled).unsqueeze(0)
            with torch.no_grad():
                y_scaled = lstm_model(x).squeeze(0).numpy()

        predicted = lstm_scaler.inverse_transform(y_scaled)
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
        response_data = {
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
            "buffer_size":    len(_hourly_buffer),
        }

        try:
            save_forecast(response_data)
        except Exception as e:
            logger.error("Erreur sauvegarde prévision : %s", e)

        return response_data

    except Exception as e:
        logger.error("Erreur prévision : %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


# ════════════════════════════════════════════════════════════════════════════
# REST — SHAP
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/shap")
async def get_shap(request: Request):
    try:
        get_current_user(request)
    except Exception:
        return JSONResponse(status_code=401, content={"error": "Non authentifié"})
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
async def get_status(request: Request):
    try:
        get_current_user(request)
    except Exception:
        return JSONResponse(status_code=401, content={"error": "Non authentifié"})
    return {
        **_STATUS,
        "ws_clients":     len(_ws_clients),
        "hourly_buffer_size": len(_hourly_buffer),
        "raw_buffer_size":   len(_raw_buffer),
        "refresh_s":      REFRESH_S,
        "active_model":   _active_model_key,
        "timestamp":      _now(),
    }


# ════════════════════════════════════════════════════════════════════════════
# REST — Modèles IA
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/models")
async def get_models(request: Request):
    try:
        get_current_user(request)
    except Exception:
        return JSONResponse(status_code=401, content={"error": "Non authentifié"})
    return {
        "active_model": _active_model_key,
        "models": _models_report,
    }


@app.post("/api/models/select/{model_key}")
async def select_model(model_key: str, request: Request):
    global pipeline, _active_model_key
    try:
        require_admin(request)
    except Exception:
        return JSONResponse(status_code=403, content={"error": "Accès réservé aux administrateurs"})

    valid_keys = [m["model_key"] for m in _models_report]
    if model_key not in valid_keys:
        return JSONResponse(
            status_code=404,
            content={"error": f"Modèle '{model_key}' introuvable. Disponibles : {valid_keys}"},
        )

    rank = next(m["rank"] for m in _models_report if m["model_key"] == model_key)
    model_file = _models_dir / f"model_{rank}_{model_key}.joblib"
    if not model_file.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"Fichier {model_file.name} introuvable sur disque."},
        )

    try:
        import joblib
        from src.sensor_inference import get_sensor_reader
        from config import DEFAULT_THRESHOLD

        model = joblib.load(model_file)
        scaler = joblib.load(_models_dir / "scaler.joblib")

        if hasattr(model, "threshold"):
            model.threshold = DEFAULT_THRESHOLD

        from src.sensor_inference import SensorPipeline
        pipeline = SensorPipeline(model, scaler,
                                  sensor_reader=get_sensor_reader(mock=_STATUS.get("mock_mode", True)))
        _active_model_key = model_key

        logger.info("Modèle actif changé → %s (rank=%d, seuil=%.2f)",
                    model_key, rank, getattr(model, "threshold", 0.5))

        return {
            "message": f"Modèle actif : {model_key} (rank #{rank})",
            "active_model": model_key,
            "threshold": getattr(model, "threshold", 0.5),
        }
    except Exception as e:
        logger.error("Erreur changement de modèle : %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


# ════════════════════════════════════════════════════════════════════════════
# REST — Historiques
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/history/diagnostic")
async def api_history_diagnostic(request: Request, limit: int = 100):
    try:
        get_current_user(request)
    except Exception:
        return JSONResponse(status_code=401, content={"error": "Non authentifié"})
    return get_history_diagnostic(limit)


@app.get("/api/history/filtration")
async def api_history_filtration(request: Request, limit: int = 100):
    try:
        get_current_user(request)
    except Exception:
        return JSONResponse(status_code=401, content={"error": "Non authentifié"})
    return get_history_filtration(limit)


@app.get("/api/history/forecast")
async def api_history_forecast(request: Request, limit: int = 50):
    try:
        get_current_user(request)
    except Exception:
        return JSONResponse(status_code=401, content={"error": "Non authentifié"})
    return get_history_forecast(limit)


# ════════════════════════════════════════════════════════════════════════════
# REST — Activation manuelle des filtres
# ════════════════════════════════════════════════════════════════════════════

@app.post("/api/filter/activate")
async def activate_filters(request: Request):
    try:
        require_admin(request)
    except Exception:
        return JSONResponse(status_code=403, content={"error": "Accès réservé aux administrateurs"})

    if filter_controller is None:
        return JSONResponse(
            status_code=503,
            content={"error": "FilterController non disponible."},
        )

    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Pipeline de diagnostic non disponible."},
        )

    result = pipeline.run_once()

    decision, action = filter_controller.decide_and_activate(
        result, shap_summary=shap_summary,
    )

    if action is not None:
        try:
            save_filtration([action.to_dict()])
        except Exception as e:
            logger.error("Erreur sauvegarde filtration : %s", e)

    return {
        "decision":       decision.to_dict(),
        "action":         action.to_dict() if action else None,
        "message":        decision.reason,
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