"""
AdaptGuard AI — Performance Drift Monitor (Label-Required Channel)
Monitors model error rates using ADWIN and Page-Hinkley.

IMPORTANT: This monitor requires CONFIRMED labels.
In this system, labels arrive after a delay (1–7 days).
Therefore this monitor CANNOT fire in real-time — it fires only after
the delayed label buffer releases a confirmed label.

Architecture role: Second drift signal — confirms genuine concept drift
once labels provide evidence of model error-rate changes.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

from src.utils.logger import get_logger
from src.utils.config import load_config

log = get_logger("drift.perf_monitor")


@dataclass
class PerfDriftResult:
    """Result from one performance monitoring update."""
    n_samples_processed: int
    adwin_detected:      bool
    ph_detected:         bool
    current_error_rate:  float
    error_trend:         float          # positive = error rate increasing
    detection_event:     bool           # True if any detector fired
    detector_fired:      str            = ""  # "ADWIN" | "PH" | "ADWIN+PH" | ""


class ADWINMonitor:
    """
    ADWIN (Adaptive Windowing) drift detector.

    Monitors a stream of binary prediction errors (0=correct, 1=wrong).
    Requires confirmed labels to compute errors.

    ADWIN automatically shrinks its window when it detects a statistically
    significant change in the mean of sub-windows (Hoeffding bound).

    Reference: river.drift.ADWIN (riverml.xyz)
    """

    def __init__(self, delta: float = 0.002):
        self.delta  = delta
        self._adwin = None
        self._init_adwin()
        self.detected        = False
        self.n_samples       = 0
        self.current_mean    = 0.0

    def _init_adwin(self):
        try:
            from river.drift import ADWIN
            self._adwin = ADWIN(delta=self.delta)
            log.debug(f"ADWIN initialized with delta={self.delta}")
        except ImportError:
            log.error("River not installed. ADWIN unavailable.")
            self._adwin = None

    def update(self, error: int) -> bool:
        """
        Update ADWIN with a binary error signal.

        Args:
            error: 1 if model prediction was wrong, 0 if correct.
                   REQUIRES a confirmed label to compute.

        Returns:
            True if drift detected.
        """
        if self._adwin is None:
            return False

        self._adwin.update(error)
        self.n_samples    += 1
        self.current_mean  = self._adwin.estimation
        self.detected      = self._adwin.drift_detected

        if self.detected:
            log.warning(
                f"[ADWIN] DRIFT DETECTED at sample #{self.n_samples} | "
                f"error_rate={self.current_mean:.4f}"
            )

        return self.detected

    def reset(self) -> None:
        self._init_adwin()
        self.detected = False


class PageHinkleyMonitor:
    """
    Page-Hinkley change detection test.

    Detects abrupt increases in a stream's mean.
    Used as a secondary performance drift detector.

    Requires confirmed labels to compute prediction errors.
    """

    def __init__(self, threshold: float = 50, alpha: float = 0.9999):
        self.threshold  = threshold
        self.alpha      = alpha
        self._sum       = 0.0
        self._min_sum   = 0.0
        self._mean      = 0.0
        self.n_samples  = 0
        self.detected   = False

    def update(self, value: float) -> bool:
        """
        Update PH with a new error value.

        Args:
            value: Prediction error (0 or 1) after confirmed label.

        Returns:
            True if drift detected.
        """
        self.n_samples += 1
        self._mean = self._mean + (value - self._mean) / self.n_samples
        self._sum  = self._sum * self.alpha + (value - self._mean - 0.01)

        if self._sum < self._min_sum:
            self._min_sum = self._sum

        ph_statistic  = self._sum - self._min_sum
        self.detected = ph_statistic > self.threshold

        if self.detected:
            log.warning(
                f"[PageHinkley] DRIFT DETECTED at sample #{self.n_samples} | "
                f"PH_stat={ph_statistic:.2f}"
            )

        return self.detected

    def reset(self) -> None:
        self._sum     = 0.0
        self._min_sum = 0.0
        self._mean    = 0.0
        self.n_samples = 0
        self.detected  = False


# ---------------------------------------------------------------------------
# Unified Performance Monitor
# ---------------------------------------------------------------------------

class PerformanceDriftMonitor:
    """
    Orchestrates ADWIN and Page-Hinkley for performance-based drift detection.

    Called ONLY when a confirmed label becomes available from the delayed-label buffer.
    Feeds on prediction errors — therefore it is a PERFORMANCE drift detector,
    not a real-time data drift detector.
    """

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or load_config()
        dc  = cfg["drift"]

        self.adwin = ADWINMonitor(delta=dc["adwin"]["delta"])
        self.ph    = PageHinkleyMonitor(
            threshold = dc["page_hinkley"]["threshold"],
            alpha     = dc["page_hinkley"]["alpha"],
        )

        self._error_buffer: deque = deque(maxlen=500)  # Rolling error rate buffer
        self.n_processed = 0
        self.history: list[PerfDriftResult] = []

    def update(self, y_true: int, y_pred: int) -> PerfDriftResult:
        """
        Update monitors with a confirmed prediction-label pair.

        Args:
            y_true: Confirmed ground-truth fraud label (requires delay wait).
            y_pred: The prediction made at transaction time.

        Returns:
            PerfDriftResult with current state.
        """
        error = int(y_true != y_pred)

        adwin_detected = self.adwin.update(error)
        ph_detected    = self.ph.update(float(error))

        self._error_buffer.append(error)
        self.n_processed += 1

        current_error_rate = np.mean(self._error_buffer) if self._error_buffer else 0.0

        # Error trend: compare recent 50 vs previous 50
        if len(self._error_buffer) >= 100:
            recent   = list(self._error_buffer)[-50:]
            previous = list(self._error_buffer)[-100:-50]
            error_trend = float(np.mean(recent) - np.mean(previous))
        else:
            error_trend = 0.0

        any_detected = adwin_detected or ph_detected
        fired = (
            "ADWIN+PH" if (adwin_detected and ph_detected)
            else "ADWIN"  if adwin_detected
            else "PH"     if ph_detected
            else ""
        )

        result = PerfDriftResult(
            n_samples_processed = self.n_processed,
            adwin_detected      = adwin_detected,
            ph_detected         = ph_detected,
            current_error_rate  = current_error_rate,
            error_trend         = error_trend,
            detection_event     = any_detected,
            detector_fired      = fired,
        )

        self.history.append(result)
        return result

    def reset_detectors(self) -> None:
        """Reset drift detectors after adaptation (prevent re-triggering)."""
        self.adwin.reset()
        self.ph.reset()
        log.info("[PerfMonitor] Detectors reset after adaptation.")

    @property
    def current_error_rate(self) -> float:
        return float(np.mean(self._error_buffer)) if self._error_buffer else 0.0
