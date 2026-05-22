"""
Edge Defense API — FastAPI backend for edge-deployed intrusion detection.

Serves real-time XGBoost predictions on network flow features with
SHAP explanations, calibrated thresholding, and sample-flow simulation.

Environment variables (all optional, sensible defaults provided):
    ARTIFACT_DIR          — path to the artifacts folder   (default: ./artifacts)
    PORT                  — server port                    (default: 8000)
    HOST                  — bind address                   (default: 0.0.0.0)
    CORS_ORIGINS          — comma-separated allowed origins (default: * )
    SUPABASE_URL          — Supabase project URL            (optional)
    SUPABASE_KEY          — Supabase anon/service key       (optional)
    CLERK_PUBLISHABLE_KEY — Clerk publishable key           (optional)
"""
#main

import os
import time
import pickle
from pathlib import Path
from typing import List, Dict, Any

import jwt
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from dotenv import load_dotenv
from supabase import create_client
from fastapi import FastAPI, HTTPException, Path as PathParam, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import FlowInput, HealthResponse, AnalysisResponse, _EXAMPLE_FEATURES

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration (all overridable via env vars)
# ---------------------------------------------------------------------------
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", Path(__file__).resolve().parent.parent / "artifacts"))
CORS_ORIGINS: List[str] = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "*").split(",")
    if o.strip()
]

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Edge Defense API",
    description=(
        "Real-time intrusion detection with SHAP explanations.\n\n"
        "Uses a lightweight XGBoost model trained on CIC-IDS flow features "
        "to classify network flows as **ATTACK** or **BENIGN**, with a "
        "calibrated decision threshold and per-prediction SHAP explanations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — configurable via CORS_ORIGINS env var
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state (populated on startup)
# ---------------------------------------------------------------------------
model: xgb.XGBClassifier = None  # type: ignore[assignment]
features: List[str] = []
feature_stats: pd.DataFrame = pd.DataFrame()
threshold: float = 0.5
sample_flows: pd.DataFrame = pd.DataFrame()
explainer: shap.TreeExplainer = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
def load_artifacts() -> None:
    """Load all ML artifacts into global state."""
    global model, features, feature_stats, threshold, sample_flows, explainer

    print(f"📂 Loading artifacts from: {ARTIFACT_DIR}")

    # Model
    model = xgb.XGBClassifier()
    model.load_model(str(ARTIFACT_DIR / "edge_model.json"))

    # Feature name list (order matters)
    with open(ARTIFACT_DIR / "features.pkl", "rb") as f:
        features = pickle.load(f)

    # Feature statistics (min / max / mean / std)
    with open(ARTIFACT_DIR / "feature_stats.pkl", "rb") as f:
        feature_stats = pickle.load(f)

    # Calibrated threshold
    with open(ARTIFACT_DIR / "optimal_threshold.pkl", "rb") as f:
        threshold = pickle.load(f)

    # Sample flows
    sample_flows = pd.read_csv(ARTIFACT_DIR / "sample_flows.csv")

    # SHAP explainer
    explainer = shap.TreeExplainer(model)

    print("✅ Artifacts loaded")
    print(f"   Model objective : {model.objective}")
    print(f"   Features        : {len(features)}")
    print(f"   Threshold       : {threshold}")
    print(f"   Sample flows    : {len(sample_flows)}")

    # Supabase setup
    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = os.getenv("SUPABASE_KEY", "")
    if supabase_url and supabase_key:
        app.state.supabase = create_client(supabase_url, supabase_key)
        print("✅ Supabase connected")
    else:
        app.state.supabase = None
        print("⚠️  Supabase not configured — logging disabled")

    # Clerk setup
    app.state.clerk_publishable_key = os.getenv("CLERK_PUBLISHABLE_KEY", "")
    if app.state.clerk_publishable_key:
        print("✅ Clerk auth configured")
    else:
        print("⚠️  Clerk not configured — auth disabled, all requests allowed")


# ---------------------------------------------------------------------------
# Auth + logging helpers
# ---------------------------------------------------------------------------
def get_user_id_from_request(request: Request) -> str | None:
    """Extract Clerk user ID from JWT token. Returns None if no valid token."""
    try:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return None
        token = auth_header.split(" ")[1]
        # Decode without verification to extract sub claim
        # Clerk tokens are verified by Clerk's infrastructure
        decoded = jwt.decode(token, options={"verify_signature": False})
        return decoded.get("sub")
    except Exception:
        return None


async def log_analysis(request: Request, result: dict, source: str = "api"):
    """Non-blocking Supabase logger — never crashes the API if DB is down."""
    try:
        if not hasattr(request.app.state, 'supabase') or request.app.state.supabase is None:
            return
        user_id = get_user_id_from_request(request)
        request.app.state.supabase.table("analyses").insert({
            "user_id": user_id,
            "prediction": result["prediction"],
            "label": result["label"],
            "probability": round(result["probability"], 6),
            "inference_ms": round(result["inference_ms"], 3),
            "feature_values": dict(zip(result["feature_names"], result["feature_values"])),
            "shap_values": result["shap_values"],
            "shap_base_value": round(result["shap_base_value"], 6),
            "source": source
        }).execute()
    except Exception as e:
        print(f"⚠️  DB logging failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Core analysis logic (shared by /analyze and /sample-analyze)
# ---------------------------------------------------------------------------
def _run_analysis(feature_dict: Dict[str, float]) -> Dict[str, Any]:
    """Run prediction + SHAP on a single flow and return the result dict."""

    # 1. Validate all required features are present
    missing = [f for f in features if f not in feature_dict]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Missing required features",
                "missing_features": missing,
                "expected_count": len(features),
                "received_count": len(features) - len(missing),
            },
        )

    # 2. Build ordered DataFrame
    df = pd.DataFrame(
        [[feature_dict[f] for f in features]],
        columns=features,
    )

    # 3. Range validation (warn only — never reject)
    for feat in features:
        val = feature_dict[feat]
        f_min = float(feature_stats.loc["min", feat])
        f_max = float(feature_stats.loc["max", feat])
        if val < f_min or val > f_max:
            print(
                f"⚠️  Feature '{feat}' value {val} is outside training range "
                f"[{f_min}, {f_max}]"
            )

    # 4. Inference with latency measurement
    start = time.perf_counter()
    prob = float(model.predict_proba(df)[0][1])
    end = time.perf_counter()

    prediction = 1 if prob >= threshold else 0
    label = "ATTACK" if prediction == 1 else "BENIGN"

    # 5. SHAP explanation (handle both list and ndarray safely)
    shap_vals = explainer.shap_values(df)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    # Ensure we have a plain Python list of floats
    shap_row = shap_vals[0]
    if hasattr(shap_row, "tolist"):
        shap_vals_list = shap_row.tolist()
    else:
        shap_vals_list = [float(v) for v in shap_row]

    # expected_value may be a scalar, ndarray, or list
    base_val_raw = explainer.expected_value
    if isinstance(base_val_raw, np.ndarray):
        base_val = float(base_val_raw.flat[0])
    elif isinstance(base_val_raw, (list, tuple)):
        base_val = float(base_val_raw[0])
    else:
        base_val = float(base_val_raw)

    return {
        "prediction": prediction,
        "label": label,
        "probability": round(prob, 6),
        "threshold_used": float(threshold),
        "inference_ms": round((end - start) * 1000, 4),
        "shap_values": shap_vals_list,
        "shap_base_value": base_val,
        "feature_names": list(features),
        "feature_values": [float(v) for v in df.iloc[0]],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
def health_check() -> Dict[str, Any]:
    """Basic health / readiness probe.  Returns model name and feature count."""
    return {
        "status": "ok",
        "model": "edge_xgboost",
        "features": len(features),
    }


@app.get(
    "/sample-flows",
    summary="List sample flows",
    tags=["Samples"],
)
def get_sample_flows() -> List[Dict[str, Any]]:
    """Return all 200 sample flows from sample_flows.csv as a JSON list.

    Each element is a dict with the 20 CIC-IDS feature names as keys.
    Use the row index (0-199) with `/sample-analyze/{row_index}` to
    run inference on any of these flows.
    """
    return sample_flows.to_dict(orient="records")


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Analyze a network flow",
    tags=["Inference"],
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": {"features": _EXAMPLE_FEATURES},
                }
            }
        }
    },
)
async def analyze_flow(request: Request, payload: FlowInput) -> Dict[str, Any]:
    """Analyze a single network flow and return prediction + SHAP.

    Submit all 20 CIC-IDS features as a **`features`** dict.  
    The Swagger editor is pre-filled with a real sample flow — 
    click **Try it out** then **Execute** to see a live prediction.
    """
    try:
        result = _run_analysis(payload.features)
        await log_analysis(request, result, source="api")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get(
    "/sample-analyze/{row_index}",
    response_model=AnalysisResponse,
    summary="Analyze a sample flow by index",
    tags=["Inference"],
)
async def sample_analyze(
    request: Request,
    row_index: int = PathParam(
        ...,
        ge=0,
        le=199,
        description="Row index in sample_flows.csv (0-199)",
        example=0,
    ),
) -> Dict[str, Any]:
    """Pull a row from sample_flows.csv by index and run the same analysis
    pipeline as `/analyze`.  Returns an identical response structure.
    """
    try:
        if row_index < 0 or row_index >= len(sample_flows):
            raise HTTPException(
                status_code=422,
                detail=f"row_index must be between 0 and {len(sample_flows) - 1}",
            )
        row = sample_flows.iloc[row_index]
        feature_dict = {f: float(row[f]) for f in features}
        result = _run_analysis(feature_dict)
        await log_analysis(request, result, source="sample")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# History endpoint
# ---------------------------------------------------------------------------
@app.get("/history", tags=["History"], summary="Fetch recent analyses")
async def get_history(request: Request, limit: int = 50):
    """Return the last N analyses from Supabase, filtered by user if authenticated.

    Pass a valid Clerk Bearer token in the `Authorization` header to see only
    your own analyses.  Without a token, returns all rows (up to `limit`).
    """
    try:
        if not hasattr(request.app.state, 'supabase') or request.app.state.supabase is None:
            return []
        user_id = get_user_id_from_request(request)
        query = request.app.state.supabase.table("analyses") \
            .select("id,created_at,user_id,prediction,label,probability,inference_ms,source") \
            .order("created_at", desc=True) \
            .limit(limit)
        # If user is authenticated, show only their analyses
        if user_id:
            query = query.eq("user_id", user_id)
        result = query.execute()
        return result.data
    except Exception as e:
        print(f"⚠️  History fetch failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Standalone entry-point (for Railway / Docker / direct python run)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("app.main:app", host=host, port=port, reload=False)
