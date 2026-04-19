# main.py
# Run: uvicorn main:app --reload --port 8000

import io
import json
import logging
import time
import asyncio
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from schemas import ActiveContextRequest, RULRequest
from services import (
    ContextStore,
    DrillPipeLifeEstimator,
    SpikeDetector,
    DEFAULT_COATING_SPECS,
    DEFAULT_JOB_STEPS,
    clamp,
    ds1_current_grade,
    ds1_next_drop,
    estimate_thickness_from_waveform,
)
from rules import classify_thickness, compute_integrity

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Pipe Integrity Monitoring API",
    description="Real-time ultrasonic waveform processing, integrity scoring, and RUL estimation for coated drill pipe.",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
hub_clients: set[WebSocket] = set()
spike_detector = SpikeDetector(window=21, k=5.0)
context_store = ContextStore()


# ---------------------------------------------------------------------------
# WebSocket helpers
# ---------------------------------------------------------------------------

async def _broadcast(event: Dict[str, Any]) -> None:
    dead: List[WebSocket] = []
    for ws in list(hub_clients):
        try:
            await ws.send_text(json.dumps(event, ensure_ascii=False))
        except Exception:
            dead.append(ws)
    for ws in dead:
        hub_clients.discard(ws)


async def _ping_loop(ws: WebSocket, interval: float = 15.0) -> None:
    try:
        while True:
            await asyncio.sleep(interval)
            await ws.send_text('{"type":"ping"}')
    except Exception:
        hub_clients.discard(ws)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
def health():
    return {"ok": True, "version": "2.0.0"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    hub_clients.add(ws)
    logger.info("WS client connected (%d total)", len(hub_clients))
    ping_task = asyncio.create_task(_ping_loop(ws))
    try:
        while True:
            await ws.receive_text()   # keep alive; detect disconnect
    except Exception:
        pass
    finally:
        ping_task.cancel()
        hub_clients.discard(ws)
        logger.info("WS client disconnected (%d remaining)", len(hub_clients))


@app.post("/master/upload", tags=["context"])
async def upload_master(file: UploadFile = File(...)):
    """
    Upload a master Excel file containing job/stage records.
    Required columns: Job_ID, Stage.
    Optional columns: Fluid_System, Coating_Type, Coating_Thickness_mils.
    """
    try:
        raw = await file.read()
        df = pd.read_excel(io.BytesIO(raw))
        n = context_store.load_master_excel(df)
        return {"ok": True, "rows_loaded": n}
    except Exception as exc:
        logger.warning("Master upload failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/context/active", tags=["context"])
async def set_active_context(body: ActiveContextRequest):
    """Set the active job/stage used to annotate waveform readings."""
    context_store.set_active(body.job_id, body.stage)
    return {"ok": True, "active": {"job_id": body.job_id, "stage": body.stage}}


@app.get("/context/active", tags=["context"])
async def get_active_context():
    return {"active": context_store.get_active_context()}


@app.post("/ingest/waveform", tags=["monitoring"])
async def ingest_waveform(file: UploadFile = File(...)):
    """
    Upload an ultrasonic A-scan waveform (Excel or CSV).

    Required columns:
      - Distance_inch
      - Amplitude_mV

    Computes wall thickness, runs spike detection, scores integrity,
    and broadcasts a `thickness.update` event over WebSocket.

    Example response payload:
    ```json
    {
      "ok": true,
      "reading": {
        "type": "thickness.update",
        "ts": 1718000000000,
        "thickness_mm": 6.35,
        "thickness_in": 0.250,
        "frontwall_in": 0.112,
        "backwall_in": 0.362,
        "is_spike": false,
        "spike_score": 0.8,
        "baseline_mm": 6.40,
        "status": "normal",
        "integrity_score": 84.2,
        "integrity_status": "NORMAL",
        "recommendation": "Continue operation",
        "job": { "job_id": "RA", "stage": "II", ... }
      }
    }
    ```
    """
    try:
        raw = await file.read()
        filename = (file.filename or "").lower()
        df = pd.read_csv(io.BytesIO(raw)) if filename.endswith(".csv") else pd.read_excel(io.BytesIO(raw))

        if "Distance_inch" not in df.columns or "Amplitude_mV" not in df.columns:
            raise ValueError("File must contain 'Distance_inch' and 'Amplitude_mV' columns.")

        dist = df["Distance_inch"].astype(float).tolist()
        amp = df["Amplitude_mV"].astype(float).tolist()

        fw_in, bw_in, thickness_in = estimate_thickness_from_waveform(dist, amp)
        thickness_mm = thickness_in * 25.4

        is_spike, spike_score, baseline_mm = spike_detector.update(thickness_mm)
        status = classify_thickness(thickness_mm)

        integrity = compute_integrity(
            thickness_mm=thickness_mm,
            is_spike=is_spike,
            spike_score=spike_score,
        )

        event: Dict[str, Any] = {
            "type": "thickness.update",
            "ts": int(time.time() * 1000),
            "thickness_in": round(thickness_in, 5),
            "thickness_mm": round(thickness_mm, 3),
            "frontwall_in": round(fw_in, 5),
            "backwall_in": round(bw_in, 5),
            "is_spike": is_spike,
            "spike_score": round(spike_score, 2),
            "baseline_mm": round(baseline_mm, 3),
            "status": status,
            "integrity_score": integrity.score,
            "integrity_status": integrity.status,
            "recommendation": integrity.recommendation,
            "job": context_store.get_active_context(),
        }

        await _broadcast(event)
        logger.info(
            "Waveform ingested — thickness=%.3f mm  status=%s  integrity=%.1f",
            thickness_mm, integrity.status, integrity.score,
        )
        return {"ok": True, "reading": event}

    except Exception as exc:
        logger.warning("Waveform ingest failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/rul/estimate", tags=["rul"])
async def estimate_rul(body: RULRequest):
    """
    Estimate corrosion rate and remaining useful life (in jobs) for a
    coated drill pipe given current wall thickness inspection data.

    The confidence penalty scales up the predicted loss-per-job when
    ML coating classification confidence is low, conservatively shortening
    the RUL estimate.

    Example request:
    ```json
    {
      "wt_before_in": 0.368,
      "wt_current_in": 0.356,
      "wt_min_in": 0.310,
      "coating": "TK34P",
      "confidence": 0.87,
      "alpha": 0.5
    }
    ```
    """
    try:
        estimator = DrillPipeLifeEstimator(
            od_in=body.od_in,
            wt_before_in=body.wt_before_in,
            wt_measured_in=body.wt_current_in,
            wt_min_in=body.wt_min_in,
            grade=body.grade,
            job_steps=DEFAULT_JOB_STEPS,
            coating_specs=DEFAULT_COATING_SPECS,
            jobs_per_year=body.jobs_per_year,
            k0_in_per_hour=body.k0_in_per_hour,
            safety_factor=body.safety_factor,
            thickness_ref_mils=body.thickness_ref_mils,
            thickness_exponent=body.thickness_exponent,
        )

        base_loss = estimator.loss_per_job_in(body.coating)
        risk_mult = 1.0 + body.alpha * (1.0 - clamp(body.confidence, 0.0, 1.0))
        adj_loss = base_loss * risk_mult

        jobs_remaining = (
            max((body.wt_current_in - body.wt_min_in) / adj_loss, 0.0)
            if adj_loss > 0 else float("inf")
        )
        corrosion_mm_per_year = adj_loss * body.jobs_per_year * 25.4

        grade_now = ds1_current_grade(body.wt_current_in, body.wt_before_in, body.wt_min_in)
        _, next_event, next_threshold_wt = ds1_next_drop(
            body.wt_current_in, body.wt_before_in, body.wt_min_in
        )

        # Include integrity score factoring in RUL
        integrity = compute_integrity(
            thickness_mm=body.wt_current_in * 25.4,
            is_spike=False,
            spike_score=0.0,
            confidence=body.confidence,
            jobs_remaining=jobs_remaining,
        )

        return {
            "ok": True,
            "coating": body.coating,
            "base_loss_per_job_in": round(base_loss, 8),
            "risk_multiplier": round(risk_mult, 4),
            "adjusted_loss_per_job_in": round(adj_loss, 8),
            "jobs_remaining": round(jobs_remaining, 1),
            "corrosion_rate_mm_per_year": round(corrosion_mm_per_year, 4),
            "ds1_grade": grade_now,
            "ds1_next_event": next_event,
            "ds1_next_threshold_wt_in": next_threshold_wt,
            "integrity_score": integrity.score,
            "integrity_status": integrity.status,
            "recommendation": integrity.recommendation,
        }

    except Exception as exc:
        logger.warning("RUL estimate failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
