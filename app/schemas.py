"""Pydantic schemas for the Edge Defense API."""

from pydantic import BaseModel, Field
from typing import Dict, List

# ---------------------------------------------------------------------------
# Example payload — a real sample flow so Swagger pre-fills usable values
# ---------------------------------------------------------------------------
_EXAMPLE_FEATURES: Dict[str, float] = {
    "Packet Length Variance": 2617702.352,
    "Bwd Packet Length Max": 5792.0,
    "Max Packet Length": 5792.0,
    "Total Length of Fwd Packets": 430.0,
    "Packet Length Mean": 802.0666667,
    "Fwd Packet Length Mean": 71.66666667,
    "Fwd IAT Std": 37000000.0,
    "Fwd Packet Length Max": 412.0,
    "Bwd Header Length": 264.0,
    "Fwd Header Length": 164.0,
    "PSH Flag Count": 0.0,
    "Flow IAT Std": 23000000.0,
    "Init_Win_bytes_backward": 235.0,
    "Flow IAT Mean": 6378959.538,
    "Bwd Packet Length Min": 0.0,
    "Flow IAT Max": 82800000.0,
    "min_seg_size_forward": 20.0,
    "Min Packet Length": 0.0,
    "Init_Win_bytes_forward": 0.0,
    "act_data_pkt_fwd": 3.0,
}


class FlowInput(BaseModel):
    """Input schema for network flow analysis.

    Accepts a dictionary mapping the 20 CIC-IDS feature names to their
    float values.  All 20 features listed below must be present:

    Packet Length Variance, Bwd Packet Length Max, Max Packet Length,
    Total Length of Fwd Packets, Packet Length Mean, Fwd Packet Length Mean,
    Fwd IAT Std, Fwd Packet Length Max, Bwd Header Length, Fwd Header Length,
    PSH Flag Count, Flow IAT Std, Init_Win_bytes_backward, Flow IAT Mean,
    Bwd Packet Length Min, Flow IAT Max, min_seg_size_forward,
    Min Packet Length, Init_Win_bytes_forward, act_data_pkt_fwd
    """

    features: Dict[str, float] = Field(
        ...,
        description=(
            "Dictionary mapping each of the 20 CIC-IDS feature names to a "
            "float value.  All 20 keys must be present."
        ),
        json_schema_extra={
            "example": _EXAMPLE_FEATURES,
        },
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": _EXAMPLE_FEATURES,
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Response models (optional but improves Swagger docs)
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    model: str = Field(..., example="edge_xgboost")
    features: int = Field(..., example=20)


class AnalysisResponse(BaseModel):
    prediction: int = Field(..., description="1 = ATTACK, 0 = BENIGN", example=1)
    label: str = Field(..., example="ATTACK")
    probability: float = Field(..., description="Model confidence (0-1)", example=0.999984)
    threshold_used: float = Field(..., example=0.6587)
    inference_ms: float = Field(..., description="Inference latency in milliseconds", example=0.85)
    shap_values: List[float] = Field(..., description="Per-feature SHAP contributions")
    shap_base_value: float = Field(..., description="SHAP base (expected) value")
    feature_names: List[str] = Field(..., description="Ordered feature names used by the model")
    feature_values: List[float] = Field(..., description="Input feature values in model order")
