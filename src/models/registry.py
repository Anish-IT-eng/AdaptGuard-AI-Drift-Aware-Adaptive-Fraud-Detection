"""
AdaptGuard AI — Model Registry
Tracks all model versions with metadata, supports promotion and rollback.

Every model in the registry has:
- version number
- timestamp
- training data range
- feature version
- hyperparameters
- evaluation metrics
- parent model version
- status (production | candidate | archived | rolled_back)
"""

import joblib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from src.utils.logger import get_logger

log = get_logger("models.registry")

MODEL_STATUS = {"production", "candidate", "archived", "rolled_back"}


@dataclass
class ModelRecord:
    """Metadata record for a single model version."""
    version:         int
    name:            str
    status:          str          # production | candidate | archived | rolled_back
    created_at:      str
    train_start:     str          # Start of training data window
    train_end:       str          # End of training data window
    feature_version: str
    metrics:         dict = field(default_factory=dict)
    hyperparams:     dict = field(default_factory=dict)
    parent_version:  Optional[int] = None
    promoted_at:     Optional[str] = None
    archived_at:     Optional[str] = None
    rollback_reason: Optional[str] = None
    model_path:      Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class ModelRegistry:
    """
    In-process model registry with file-system persistence.

    Responsibilities:
    - Track all model versions and their lifecycle states
    - Support promotion (candidate → production)
    - Support rejection (candidate → archived, never touched production)
    - Support rollback (production → rolled_back, previous version restored)
    - Persist registry index as JSON
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.models_dir / "registry_index.json"
        self._records: dict[int, ModelRecord] = {}
        self._production_version: Optional[int] = None
        self._version_counter = 0
        self._rejected_count  = 0   # Candidates rejected before reaching production
        self._rollback_count  = 0   # Promoted models that were later rolled back

        self._load_index()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        model: Any,
        name: str,
        train_start: str,
        train_end: str,
        metrics: dict,
        hyperparams: dict,
        feature_version: str = "v1",
        parent_version: Optional[int] = None,
        status: str = "candidate",
    ) -> int:
        """
        Register a new model version.

        Args:
            model:           Fitted sklearn-compatible model object.
            name:            Model name (e.g., 'xgboost', 'sgd_adaptive').
            train_start:     ISO string of training data start date.
            train_end:       ISO string of training data end date.
            metrics:         Dict of evaluation metrics at registration time.
            hyperparams:     Dict of model hyperparameters.
            feature_version: Version tag of the feature set used.
            parent_version:  Version number of the model this was derived from.
            status:          'candidate' or 'production'.

        Returns:
            Assigned version number.
        """
        self._version_counter += 1
        v = self._version_counter

        model_path = str(self.models_dir / f"model_v{v}_{name}.joblib")
        joblib.dump(model, model_path)

        record = ModelRecord(
            version         = v,
            name            = name,
            status          = status,
            created_at      = datetime.utcnow().isoformat(),
            train_start     = train_start,
            train_end       = train_end,
            feature_version = feature_version,
            metrics         = metrics,
            hyperparams     = hyperparams,
            parent_version  = parent_version,
            model_path      = model_path,
        )
        self._records[v] = record

        if status == "production":
            self._production_version = v

        self._save_index()
        log.info(
            f"Model registered: v{v} | name={name} | status={status} | "
            f"PR-AUC={metrics.get('pr_auc', 'N/A'):.4f}"
            if isinstance(metrics.get("pr_auc"), float)
            else f"Model registered: v{v} | name={name} | status={status}"
        )
        return v

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def promote(self, candidate_version: int) -> None:
        """
        Promote a candidate to production.
        Previous production model moves to 'archived'.

        Args:
            candidate_version: The version number to promote.
        """
        if candidate_version not in self._records:
            raise KeyError(f"Version {candidate_version} not found in registry.")

        record = self._records[candidate_version]
        if record.status not in ("candidate",):
            raise ValueError(f"Cannot promote a model with status '{record.status}'.")

        # Archive current production
        if self._production_version and self._production_version in self._records:
            old = self._records[self._production_version]
            old.status      = "archived"
            old.archived_at = datetime.utcnow().isoformat()
            log.info(f"Previous production v{self._production_version} → archived")

        record.status      = "production"
        record.promoted_at = datetime.utcnow().isoformat()
        self._production_version = candidate_version

        self._save_index()
        log.info(f"✅ Promoted v{candidate_version} to PRODUCTION")

    # ------------------------------------------------------------------
    # Rejection (candidate never reached production)
    # ------------------------------------------------------------------

    def reject(self, candidate_version: int, reason: str = "") -> None:
        """
        Reject a candidate model (it was never deployed).
        This is NOT a rollback.
        """
        if candidate_version not in self._records:
            raise KeyError(f"Version {candidate_version} not found.")

        record = self._records[candidate_version]
        record.status          = "archived"
        record.archived_at     = datetime.utcnow().isoformat()
        record.rollback_reason = f"REJECTED: {reason}"

        self._rejected_count += 1
        self._save_index()
        log.info(
            f"❌ Candidate v{candidate_version} REJECTED (never deployed). "
            f"Reason: {reason}. "
            f"Current production remains v{self._production_version}."
        )

    # ------------------------------------------------------------------
    # Rollback (production was already deployed, then degraded)
    # ------------------------------------------------------------------

    def rollback(self, reason: str = "post-deployment degradation") -> int:
        """
        Roll back from the current production model to the most recent
        archived (previously production) model.

        This is triggered ONLY when a promoted model degrades after deployment.

        Returns:
            Version number of the restored model.
        """
        if self._production_version is None:
            raise RuntimeError("No production model to roll back from.")

        # Mark current production as rolled_back
        current = self._records[self._production_version]
        current.status          = "rolled_back"
        current.rollback_reason = reason
        current.archived_at     = datetime.utcnow().isoformat()
        log.warning(
            f"⏮ ROLLBACK triggered for v{self._production_version}. "
            f"Reason: {reason}"
        )

        # Find previous production model (most recent 'archived' with production lineage)
        previous_production = None
        for v in sorted(self._records.keys(), reverse=True):
            r = self._records[v]
            if r.status == "archived" and r.promoted_at is not None and v < self._production_version:
                previous_production = v
                break

        if previous_production is None:
            raise RuntimeError("No previous production model found for rollback.")

        # Restore previous model to production
        restored = self._records[previous_production]
        restored.status      = "production"
        restored.promoted_at = datetime.utcnow().isoformat()
        self._production_version = previous_production

        self._rollback_count += 1
        self._save_index()
        log.info(f"⏮ Restored v{previous_production} to PRODUCTION after rollback.")
        return previous_production

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_production_model(self) -> tuple[Any, ModelRecord]:
        """Load and return the current production model + its record."""
        if self._production_version is None:
            raise RuntimeError("No production model registered.")
        record = self._records[self._production_version]
        model  = joblib.load(record.model_path)
        return model, record

    def get_record(self, version: int) -> ModelRecord:
        if version not in self._records:
            raise KeyError(f"Version {version} not found.")
        return self._records[version]

    def get_all_records(self) -> list[ModelRecord]:
        return list(self._records.values())

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def production_version(self) -> Optional[int]:
        return self._production_version

    @property
    def rejected_count(self) -> int:
        return self._rejected_count

    @property
    def rollback_count(self) -> int:
        return self._rollback_count

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_index(self) -> None:
        index = {
            "version_counter":      self._version_counter,
            "production_version":   self._production_version,
            "rejected_count":       self._rejected_count,
            "rollback_count":       self._rollback_count,
            "records": {str(k): v.to_dict() for k, v in self._records.items()},
        }
        with open(self.index_path, "w") as f:
            json.dump(index, f, indent=2)

    def _load_index(self) -> None:
        if not self.index_path.exists():
            return
        with open(self.index_path) as f:
            index = json.load(f)

        self._version_counter    = index.get("version_counter", 0)
        self._production_version = index.get("production_version")
        self._rejected_count     = index.get("rejected_count", 0)
        self._rollback_count     = index.get("rollback_count", 0)

        for k, v in index.get("records", {}).items():
            self._records[int(k)] = ModelRecord(**v)

        log.info(
            f"Registry loaded: {len(self._records)} versions | "
            f"production=v{self._production_version}"
        )

    def summary(self) -> str:
        lines = [
            f"{'='*60}",
            f"  MODEL REGISTRY SUMMARY",
            f"{'='*60}",
            f"  Total versions:    {len(self._records)}",
            f"  Production:        v{self._production_version}",
            f"  Rejected:          {self._rejected_count}",
            f"  Rollbacks:         {self._rollback_count}",
            f"{'='*60}",
        ]
        for r in sorted(self._records.values(), key=lambda x: x.version):
            pr_auc = r.metrics.get("pr_auc", "TBD")
            lines.append(
                f"  v{r.version:<4} [{r.status:<12}] {r.name:<20} PR-AUC={pr_auc}"
            )
        lines.append(f"{'='*60}")
        return "\n".join(lines)
