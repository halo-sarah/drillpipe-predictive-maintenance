# schemas.py
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ActiveContextRequest(BaseModel):
    job_id: str = Field(..., min_length=1)
    stage: str = Field(..., min_length=1)


class RULRequest(BaseModel):
    od_in: float = Field(5.0, gt=0)
    wt_before_in: float = Field(..., gt=0)
    wt_current_in: float = Field(..., gt=0)
    wt_min_in: float = Field(..., gt=0)
    coating: str = "TK34P"
    grade: str = "G105"
    jobs_per_year: int = Field(12, ge=1, le=100)
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    alpha: float = Field(0.5, ge=0.0, le=2.0)
    k0_in_per_hour: float = Field(2.5e-5, gt=0)
    safety_factor: float = Field(1.25, gt=0)
    thickness_ref_mils: float = Field(9.0, gt=0)
    thickness_exponent: float = Field(0.6, gt=0)

    @field_validator("wt_current_in")
    @classmethod
    def current_below_before(cls, v, info):
        before = info.data.get("wt_before_in")
        if before is not None and v > before:
            raise ValueError("wt_current_in must be <= wt_before_in")
        return v

    @field_validator("wt_min_in")
    @classmethod
    def min_below_current(cls, v, info):
        current = info.data.get("wt_current_in")
        if current is not None and v >= current:
            raise ValueError("wt_min_in must be < wt_current_in")
        return v


class ThicknessEvent(BaseModel):
    """Outbound WebSocket event for a processed waveform reading."""
    type: str
    ts: int
    thickness_in: float
    thickness_mm: float
    frontwall_in: float
    backwall_in: float
    is_spike: bool
    spike_score: float
    baseline_mm: float
    status: str                    # normal | warning | critical
    integrity_score: float         # 0–100
    integrity_status: str          # NORMAL | WARNING | CRITICAL
    recommendation: str
    job: Optional[dict]
