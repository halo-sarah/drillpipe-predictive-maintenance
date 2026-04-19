# services.py
"""
Core domain services: waveform processing, spike detection,
context management, and RUL estimation.

No external state — each class is self-contained and independently testable.
"""

import math
import logging
from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Inhibitor model
# ---------------------------------------------------------------------------

def inhibitor_efficiency(gpt: float, e_max: float = 0.92, k: float = 10.0) -> float:
    """Saturating Langmuir-type inhibitor efficiency: eff = e_max * gpt / (k + gpt)."""
    if gpt <= 0:
        return 0.0
    return clamp(e_max * gpt / (k + gpt), 0.0, 0.95)


# ---------------------------------------------------------------------------
# Job / Coating domain objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JobStep:
    name: str
    minutes: float
    hcl_wt_pct: float
    hf_wt_pct: float
    nh4cl_wt_pct: float
    inhibitor_gpt: float
    chelating_agent: bool
    penetrating_agent: bool


@dataclass(frozen=True)
class CoatingSpec:
    name: str
    thickness_min_mils: float
    thickness_max_mils: float
    base_factor: float  # Lower = better corrosion protection

    def thickness_nominal_mils(self) -> float:
        return 0.5 * (self.thickness_min_mils + self.thickness_max_mils)


# Default job programme and coating library
_stim_recipe = dict(
    hcl_wt_pct=10.0, hf_wt_pct=3.0, nh4cl_wt_pct=2.0,
    inhibitor_gpt=20.0, chelating_agent=True, penetrating_agent=True,
)
_flush_recipe = dict(
    hcl_wt_pct=2.0, hf_wt_pct=0.0, nh4cl_wt_pct=2.0,
    inhibitor_gpt=20.0, chelating_agent=False, penetrating_agent=False,
)

DEFAULT_JOB_STEPS: List[JobStep] = [
    JobStep("stim_pre",   54.0, **_stim_recipe),
    JobStep("stim_main",  90.0, **_stim_recipe),
    JobStep("flush",      97.5, **_flush_recipe),
]

DEFAULT_COATING_SPECS: Dict[str, CoatingSpec] = {
    "TK34":    CoatingSpec("TK34",    5.0,  9.0,  base_factor=0.60),
    "TK34P":   CoatingSpec("TK34P",   6.0, 12.0,  base_factor=0.55),
    "TC2000P": CoatingSpec("TC2000P", 6.0, 10.0,  base_factor=0.55),
}

DS1_GRADE_THRESHOLDS = [
    ("Premium", 0.90),
    ("Class 1", 0.80),
    ("Class 2", 0.70),
    ("Class 3", 0.60),
]


# ---------------------------------------------------------------------------
# DS-1 grade helpers
# ---------------------------------------------------------------------------

def ds1_current_grade(wt_current: float, wt_reference: float, wt_min: float) -> str:
    if wt_current <= wt_min:
        return "Scrap"
    if wt_reference <= 0:
        return "Unknown"
    ratio = wt_current / wt_reference
    for grade, threshold in DS1_GRADE_THRESHOLDS:
        if ratio >= threshold:
            return grade
    return "Below Class 3"


def ds1_next_drop(
    wt_current: float,
    wt_reference: float,
    wt_min: float,
) -> Tuple[str, str, Optional[float]]:
    """Return (current_grade, next_event_label, threshold_wt_in)."""
    current = ds1_current_grade(wt_current, wt_reference, wt_min)
    if current == "Scrap":
        return current, "N/A", None
    if wt_reference <= 0:
        return current, "Retire/Scrap", wt_min

    idx = next((i for i, (g, _) in enumerate(DS1_GRADE_THRESHOLDS) if g == current), None)
    if idx is None or idx == len(DS1_GRADE_THRESHOLDS) - 1:
        return current, "Retire/Scrap", wt_min

    next_grade, next_ratio = DS1_GRADE_THRESHOLDS[idx + 1]
    next_wt = next_ratio * wt_reference
    if wt_min >= next_wt:
        return current, "Retire/Scrap", wt_min

    return current, f"Drop to {next_grade}", next_wt


# ---------------------------------------------------------------------------
# RUL Estimator
# ---------------------------------------------------------------------------

class DrillPipeLifeEstimator:
    """
    Physics-based remaining-useful-life estimator for coated drill pipe
    under acid stimulation service.

    Loss per job is computed step-by-step from chemical exposure factors,
    inhibitor efficiency, and coating protection factor.
    """

    def __init__(
        self,
        od_in: float,
        wt_before_in: float,
        wt_measured_in: float,
        wt_min_in: float,
        grade: str,
        job_steps: List[JobStep],
        coating_specs: Dict[str, CoatingSpec],
        jobs_per_year: int = 12,
        k0_in_per_hour: float = 2.5e-5,
        safety_factor: float = 1.25,
        thickness_ref_mils: float = 9.0,
        thickness_exponent: float = 0.6,
    ) -> None:
        self.od_in = float(od_in)
        self.wt_before_in = float(wt_before_in)
        self.wt_measured_in = float(wt_measured_in)
        self.wt_min_in = float(wt_min_in)
        self.grade = str(grade)
        self.job_steps = list(job_steps)
        self.coating_specs = dict(coating_specs)
        self.jobs_per_year = int(jobs_per_year)
        self.k0_in_per_hour = float(k0_in_per_hour)
        self.safety_factor = float(safety_factor)
        self.thickness_ref_mils = float(thickness_ref_mils)
        self.thickness_exponent = float(thickness_exponent)

    # -- Chemical severity factors --

    @staticmethod
    def _f_hcl(hcl_wt_pct: float) -> float:
        return 1.0 + 0.25 * (max(hcl_wt_pct, 0.0) ** 1.2)

    @staticmethod
    def _f_hf(hf_wt_pct: float) -> float:
        return 1.0 + 0.45 * (max(hf_wt_pct, 0.0) ** 1.1)

    @staticmethod
    def _f_nh4cl(nh4cl_wt_pct: float) -> float:
        return 1.0 + 0.20 * math.log10(1.0 + max(nh4cl_wt_pct, 0.0))

    @staticmethod
    def _f_additives(chelating_agent: bool, penetrating_agent: bool) -> float:
        f = 1.0
        if chelating_agent:
            f *= 0.95
        if penetrating_agent:
            f *= 1.05
        return f

    def _coating_factor(self, coating: str) -> float:
        if coating not in self.coating_specs:
            raise ValueError(
                f"Unknown coating '{coating}'. Available: {list(self.coating_specs)}"
            )
        spec = self.coating_specs[coating]
        t_nom = max(spec.thickness_nominal_mils(), 0.1)
        t_ref = max(self.thickness_ref_mils, 0.1)
        return spec.base_factor * (t_ref / t_nom) ** self.thickness_exponent

    def _step_loss_in(self, step: JobStep, coating: str) -> float:
        hours = max(step.minutes, 0.0) / 60.0
        f_inhib = 1.0 - inhibitor_efficiency(step.inhibitor_gpt)
        return max(
            0.0,
            self.k0_in_per_hour
            * self._f_hcl(step.hcl_wt_pct)
            * self._f_hf(step.hf_wt_pct)
            * self._f_nh4cl(step.nh4cl_wt_pct)
            * f_inhib
            * self._f_additives(step.chelating_agent, step.penetrating_agent)
            * self._coating_factor(coating)
            * self.safety_factor
            * hours,
        )

    def loss_per_job_in(self, coating: str) -> float:
        return sum(self._step_loss_in(s, coating) for s in self.job_steps)

    def calibrate_k0(self, coating_ref: str, measured_loss_per_job_in: float) -> float:
        """Back-calculate k0 from a known measured loss for a reference coating/job."""
        if measured_loss_per_job_in <= 0.0:
            return float("nan")
        old_k0 = self.k0_in_per_hour
        self.k0_in_per_hour = 1.0
        unit_loss = self.loss_per_job_in(coating_ref)
        self.k0_in_per_hour = old_k0
        if unit_loss <= 0.0:
            return float("inf")
        self.k0_in_per_hour = measured_loss_per_job_in / unit_loss
        return self.k0_in_per_hour


# ---------------------------------------------------------------------------
# Spike Detector
# ---------------------------------------------------------------------------

class SpikeDetector:
    """
    Rolling-window median + MAD anomaly detector.

    A reading is flagged as a spike when its deviation from the rolling
    median exceeds k × MAD (median absolute deviation).
    """

    def __init__(self, window: int = 21, k: float = 5.0) -> None:
        self.window = window
        self.k = k
        self._buf: deque[float] = deque(maxlen=window)
        self._min_samples = max(7, window // 3)

    def update(self, value_mm: float) -> Tuple[bool, float, float]:
        """
        Push a new reading.

        Returns:
            is_spike: True if anomalous
            score:    deviation in units of MAD
            baseline: rolling median (best estimate of true thickness)
        """
        self._buf.append(float(value_mm))
        if len(self._buf) < self._min_samples:
            return False, 0.0, float(value_mm)

        med = float(median(self._buf))
        mad = float(median([abs(v - med) for v in self._buf])) or 1e-9
        score = abs(value_mm - med) / mad
        return bool(score > self.k), float(score), med


# ---------------------------------------------------------------------------
# Waveform → Thickness
# ---------------------------------------------------------------------------

def _local_peaks(x: List[float], y: List[float], min_sep: int = 5) -> List[int]:
    peaks, last = [], -(10 ** 9)
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1] and i - last >= min_sep:
            peaks.append(i)
            last = i
    return peaks


def estimate_thickness_from_waveform(
    distance_in: List[float],
    amplitude_mv: List[float],
    ignore_near_in: float = 0.05,
    min_backwall_gap_in: float = 0.15,
) -> Tuple[float, float, float]:
    """
    Heuristic frontwall / backwall peak picker on an ultrasonic A-scan.

    Returns:
        (frontwall_in, backwall_in, thickness_in)

    The algorithm:
        1. Strip near-field noise below ignore_near_in.
        2. Find dominant local peaks (amplitude-sorted).
        3. Identify frontwall as highest peak; backwall as highest peak
           at least min_backwall_gap_in beyond the frontwall.
    """
    if len(distance_in) < 10:
        raise ValueError("Waveform too short — need at least 10 samples.")

    pts = sorted(zip(distance_in, amplitude_mv), key=lambda t: t[0])
    x_all = [float(p[0]) for p in pts]
    y_all = [abs(float(p[1])) for p in pts]

    start = next((i for i, v in enumerate(x_all) if v >= ignore_near_in), None)
    if start is None or start >= len(x_all) - 5:
        raise ValueError("No usable waveform region after near-field cutoff.")

    x = x_all[start:]
    y = y_all[start:]
    peaks = _local_peaks(x, y)

    if len(peaks) < 2:
        # Fall back: use two highest-amplitude indices
        top2 = sorted(range(len(y)), key=lambda i: y[i], reverse=True)[:2]
        top2.sort()
        p1, p2 = top2[0], top2[1]
    else:
        p1 = max(peaks, key=lambda i: y[i])
        fw_x = x[p1]
        bw_candidates = [i for i in peaks if x[i] >= fw_x + min_backwall_gap_in]
        if bw_candidates:
            p2 = max(bw_candidates, key=lambda i: y[i])
        else:
            gap_start = next((i for i, v in enumerate(x) if v >= fw_x + min_backwall_gap_in), None)
            if gap_start is None or gap_start >= len(y) - 1:
                raise ValueError("Cannot locate backwall — increase scan range.")
            p2 = max(range(gap_start, len(y)), key=lambda i: y[i])

    fw = float(x[p1])
    bw = float(x[p2])
    return fw, bw, max(0.0, bw - fw)


# ---------------------------------------------------------------------------
# Context Store
# ---------------------------------------------------------------------------

class ContextStore:
    """
    In-memory store for job/stage master data.
    Keeps track of the active job context used to annotate waveform readings.
    """

    def __init__(self) -> None:
        self._master: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._active_job_id: Optional[str] = None
        self._active_stage: Optional[str] = None

    def set_active(self, job_id: Optional[str], stage: Optional[str]) -> None:
        self._active_job_id = job_id
        self._active_stage = stage
        logger.info("Active context set: job=%s stage=%s", job_id, stage)

    def get_active_context(self) -> Optional[Dict[str, Any]]:
        if not self._active_job_id or not self._active_stage:
            return None
        return self._master.get((self._active_job_id, self._active_stage))

    def load_master_excel(self, df: pd.DataFrame) -> int:
        """
        Parse a master DataFrame (must have Job_ID and Stage columns).
        Returns the number of valid rows loaded.
        """
        col_index = {str(c).strip(): c for c in df.columns}

        def pick(*candidates: str) -> Optional[str]:
            return next((col_index[c] for c in candidates if c in col_index), None)

        job_col = pick("Job_ID", "job_id", "JOB_ID")
        stage_col = pick("Stage", "stage", "STAGE")
        if not job_col or not stage_col:
            raise ValueError("Master file must include 'Job_ID' and 'Stage' columns.")

        loaded = 0
        for _, row in df.iterrows():
            job_id = str(row[job_col]).strip()
            stage = str(row[stage_col]).strip()
            if not job_id or job_id.lower() == "nan":
                continue
            if not stage or stage.lower() == "nan":
                continue

            def safe(col: str) -> Optional[Any]:
                v = row.get(col)
                return None if pd.isna(v) else v

            record: Dict[str, Any] = {
                "job_id": job_id,
                "stage": stage,
                "fluid_system":            str(safe("Fluid_System")).strip() if safe("Fluid_System") else None,
                "coating_type":            str(safe("Coating_Type")).strip() if safe("Coating_Type") else None,
                "coating_thickness_mils":  float(safe("Coating_Thickness_mils")) if safe("Coating_Thickness_mils") is not None else None,
                "meta": {
                    str(c): row[c]
                    for c in df.columns
                    if c not in (job_col, stage_col) and not pd.isna(row.get(c))
                },
            }
            self._master[(job_id, stage)] = record
            loaded += 1

        logger.info("Loaded %d job/stage records from master.", loaded)
        return loaded
