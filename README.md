# Pipe Integrity Monitoring System

A predictive maintenance tool for coated drill pipe used in acid stimulation service.
Combines physics-based corrosion modelling, ML coating classification, and real-time ultrasonic waveform processing to produce an actionable integrity score and maintenance recommendation.

---

## What this does

| Function | Description |
|---|---|
| **Coating classification** | Random Forest + SVM ensemble predicts the in-service coating type (TK34 / TK34P / TC2000P) from 17 operational and material features |
| **Corrosion rate estimation** | Physics-based model computes wall-thickness loss per stimulation job from fluid chemistry, inhibitor loading, and coating protection factor |
| **RUL estimation** | Remaining useful life expressed in jobs, adjusted by ML confidence (conservative penalty when model certainty is low) |
| **Integrity scoring** | Single 0–100 index combining thickness condition, spike anomaly, ML confidence deficit, and RUL risk |
| **Real-time waveform processing** | FastAPI backend parses ultrasonic A-scan data, extracts frontwall/backwall peaks, and broadcasts `thickness.update` events over WebSocket |
| **DS-1 grade tracking** | Tracks API/ISO drill pipe inspection grade (Premium → Class 3 → Scrap) and predicts next grade-drop threshold |

---

## Project structure

```
pipe_integrity/
├── backend/
│   ├── main.py          # FastAPI app and API endpoints
│   ├── services.py      # Waveform processing, spike detection, RUL estimator, context store
│   ├── rules.py         # Integrity scoring and maintenance decision logic
│   └── schemas.py       # Pydantic request/response models
├── frontend/
│   └── app.py           # Streamlit dashboard (coating prediction + RUL UI)
├── models/              # Trained model artefacts (not committed — see below)
│   ├── model_rf_coating.pkl
│   ├── model_svm_coating.pkl
│   ├── scaler_coating.pkl
│   └── label_encoder_coating.pkl
├── data/                # Sample waveform and master Excel files
├── requirements.txt
└── README.md
```

---

## Backend API

Start the API server:
```bash
uvicorn main:app --reload --port 8000
```

### Key endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `WS` | `/ws` | Real-time thickness updates |
| `POST` | `/master/upload` | Upload job/stage master Excel |
| `POST` | `/context/active` | Set active job/stage context |
| `GET` | `/context/active` | Get active context |
| `POST` | `/ingest/waveform` | Process ultrasonic waveform file |
| `POST` | `/rul/estimate` | Estimate corrosion rate and RUL |

### Example: Ingest waveform (curl)

```bash
curl -X POST http://localhost:8000/ingest/waveform \
  -F "file=@sample_ascan.xlsx"
```

Example response:
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
    "job": { "job_id": "RA", "stage": "II", "coating_type": "TK34P" }
  }
}
```

### Example: RUL estimate

```bash
curl -X POST http://localhost:8000/rul/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "wt_before_in": 0.368,
    "wt_current_in": 0.356,
    "wt_min_in": 0.310,
    "coating": "TK34P",
    "confidence": 0.87,
    "alpha": 0.5
  }'
```

---

## Frontend dashboard

```bash
cd frontend
streamlit run app.py
```

Requires trained model artefacts in the working directory.

---

## Integrity score logic

The integrity score is a weighted composite:

| Component | Max contribution |
|---|---|
| Wall thickness condition | 100 (base) |
| Spike / anomaly penalty | −20 |
| ML confidence deficit (α × deficit) | −15 |
| RUL risk (< 20 jobs remaining) | −15 |

Score bands map to maintenance recommendations:

| Score | Status | Recommendation |
|---|---|---|
| ≥ 70 | NORMAL | Continue operation |
| 40 – 70 | WARNING | Schedule inspection |
| < 40 | CRITICAL | Immediate shutdown |

---

## Requirements

```
fastapi
uvicorn[standard]
pandas
openpyxl
pydantic>=2
streamlit
plotly
scikit-learn
joblib
numpy
```

---

## Model artefacts

Trained models are excluded from the repository. To reproduce:

1. Prepare labelled coating dataset with the 17 input features listed in `app.py`.
2. Train RF and SVM classifiers with `scikit-learn`.
3. Fit a `StandardScaler` on training features (used for SVM only).
4. Serialize with `joblib.dump`.

---

## Design notes

- **No enterprise architecture**: four compact files cover the full backend. No repository pattern, no service locator, no unnecessary abstraction.
- **Physics first**: corrosion model is derived from chemical exposure factors, not purely statistical. The ML model informs coating selection; the physics model drives the RUL number.
- **Conservative by default**: confidence penalty ensures the system degrades RUL estimates gracefully when model certainty is low, which is the correct behaviour for a safety-relevant system.
- **Industrial style**: the UI avoids decorative elements in favour of clean typography, restrained status colours, and data density that matches internal engineering tooling.
