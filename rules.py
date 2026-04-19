# rules.py
"""
Integrity scoring and maintenance decision rules.

Combines:
  - wall thickness condition
  - anomaly / spike detection
  - ML confidence
  - RUL risk

Outputs a single 0–100 integrity score and a maintenance recommendation tier.
"""

from dataclasses import dataclass


WARN_MM = 5.0   # mm — below this: warning
CRIT_MM = 3.0   # mm — below this: critical


@dataclass(frozen=True)
class IntegrityResult:
    score: float          # 0 = worst, 100 = best
    status: str           # NORMAL | WARNING | CRITICAL
    recommendation: str   # Continue operation | Schedule inspection | Immediate shutdown


def thickness_score(thickness_mm: float, warn: float = WARN_MM, crit: float = CRIT_MM) -> float:
    """
    Linear score from 0 to 100.
      >= warn  → 70–100
      crit..warn → 30–70
      <= crit  → 0–30
    """
    if thickness_mm >= warn:
        # map [warn, warn*2] → [70, 100], clamp at 100
        return min(100.0, 70.0 + 30.0 * (thickness_mm - warn) / warn)
    elif thickness_mm > crit:
        ratio = (thickness_mm - crit) / (warn - crit)
        return 30.0 + 40.0 * ratio
    else:
        # map [0, crit] → [0, 30]
        return max(0.0, 30.0 * thickness_mm / crit)


def spike_penalty(is_spike: bool, spike_score: float) -> float:
    """Return a 0–20 deduction based on anomaly severity."""
    if not is_spike:
        return 0.0
    # spike_score is MAD-normalised; cap contribution at 20 pts
    return min(20.0, 5.0 * min(spike_score, 4.0))


def confidence_penalty(confidence: float, alpha: float = 0.4) -> float:
    """Return a 0–15 deduction when ML confidence is low."""
    deficit = 1.0 - max(0.0, min(1.0, confidence))
    return min(15.0, alpha * deficit * 15.0 / 0.4)


def rul_penalty(jobs_remaining: float) -> float:
    """
    Return a 0–15 deduction based on remaining useful life.
      >= 20 jobs → no penalty
      5–20 jobs  → linear ramp
      < 5 jobs   → full 15 pts
    """
    if jobs_remaining >= 20:
        return 0.0
    elif jobs_remaining >= 5:
        return 15.0 * (1.0 - (jobs_remaining - 5.0) / 15.0)
    else:
        return 15.0


def compute_integrity(
    thickness_mm: float,
    is_spike: bool,
    spike_score: float,
    confidence: float = 1.0,
    jobs_remaining: float = 99.0,
) -> IntegrityResult:
    """
    Combine sub-scores into a single integrity index and map to
    a maintenance recommendation tier.

    Score bands:
      >= 70  → NORMAL   — Continue operation
      40–70  → WARNING  — Schedule inspection
      < 40   → CRITICAL — Immediate shutdown
    """
    base = thickness_score(thickness_mm)
    deductions = (
        spike_penalty(is_spike, spike_score)
        + confidence_penalty(confidence)
        + rul_penalty(jobs_remaining)
    )
    score = max(0.0, min(100.0, base - deductions))

    if score >= 70:
        status = "NORMAL"
        recommendation = "Continue operation"
    elif score >= 40:
        status = "WARNING"
        recommendation = "Schedule inspection"
    else:
        status = "CRITICAL"
        recommendation = "Immediate shutdown"

    return IntegrityResult(score=round(score, 1), status=status, recommendation=recommendation)


def classify_thickness(thickness_mm: float) -> str:
    """Simple three-band status label for individual readings."""
    if thickness_mm < CRIT_MM:
        return "critical"
    if thickness_mm < WARN_MM:
        return "warning"
    return "normal"
