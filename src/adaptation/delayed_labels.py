"""
AdaptGuard AI — Delayed Label Buffer
Simulates the real-world delay between transaction time and label availability.

In real fraud detection, fraud labels arrive later:
- A chargeback might take 3–7 days
- Investigation might take 1–30 days

This module holds predictions in a buffer and releases them only after
the configured delay has elapsed.

Experiment conditions: 0 / 1 / 3 / 7 day delays.
"""

import pandas as pd
from collections import deque
from dataclasses import dataclass
from typing import Optional, Iterator
from datetime import timedelta

from src.utils.logger import get_logger

log = get_logger("adaptation.delayed_labels")


@dataclass
class PendingLabel:
    """
    A stored prediction awaiting label confirmation.
    Released after delay_days from tx_datetime.
    """
    transaction_id:  int
    tx_datetime:     pd.Timestamp
    release_time:    pd.Timestamp     # tx_datetime + delay
    y_true:          int              # Ground-truth (held until release)
    y_pred:          int              # Prediction made at transaction time
    y_prob:          float            # Fraud probability
    features:        pd.Series        # Feature vector at prediction time


class DelayedLabelBuffer:
    """
    Buffers confirmed labels and releases them after a configurable delay.

    Usage:
        buffer = DelayedLabelBuffer(delay_days=3)

        # At transaction time:
        buffer.store(tx_id, tx_datetime, y_true, y_pred, y_prob, features)

        # During stream processing (simulate clock advancing):
        confirmed = buffer.release(current_datetime)
        for label in confirmed:
            # label.y_true is now available for drift detection and model update
    """

    def __init__(self, delay_days: int = 3):
        """
        Args:
            delay_days: Number of days to hold each label before release.
                        Experiment values: 0 (oracle), 1, 3, 7.
                        0 = labels available immediately (oracle condition).
        """
        self.delay_days    = delay_days
        self._buffer: deque = deque()
        self._total_stored  = 0
        self._total_released = 0

        log.info(f"DelayedLabelBuffer initialized: delay={delay_days} days")

    def store(
        self,
        transaction_id: int,
        tx_datetime:    pd.Timestamp,
        y_true:         int,
        y_pred:         int,
        y_prob:         float,
        features:       pd.Series,
    ) -> None:
        """
        Store a new prediction-label pair in the buffer.

        The true label (y_true) is stored but NOT released until release_time.
        This simulates label unavailability in real fraud detection.
        """
        release_time = tx_datetime + timedelta(days=self.delay_days)

        entry = PendingLabel(
            transaction_id = transaction_id,
            tx_datetime    = tx_datetime,
            release_time   = release_time,
            y_true         = y_true,
            y_pred         = y_pred,
            y_prob         = y_prob,
            features       = features,
        )
        self._buffer.append(entry)
        self._total_stored += 1

    def release(self, current_time: pd.Timestamp) -> list[PendingLabel]:
        """
        Release all labels whose release_time <= current_time.

        Args:
            current_time: The current simulation time.

        Returns:
            List of PendingLabel objects now ready for:
            - Performance drift detection (ADWIN/PH updates)
            - Online learning updates
            - Metric calculation
        """
        released = []
        remaining = deque()

        for entry in self._buffer:
            if entry.release_time <= current_time:
                released.append(entry)
                self._total_released += 1
            else:
                remaining.append(entry)

        self._buffer = remaining

        if released:
            log.debug(
                f"[LabelBuffer] Released {len(released)} labels at {current_time} | "
                f"Buffer remaining: {len(self._buffer)}"
            )

        return released

    def peek_buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def total_stored(self) -> int:
        return self._total_stored

    @property
    def total_released(self) -> int:
        return self._total_released

    def get_stats(self) -> dict:
        return {
            "delay_days":      self.delay_days,
            "total_stored":    self._total_stored,
            "total_released":  self._total_released,
            "pending_count":   len(self._buffer),
        }
