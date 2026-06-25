from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class PlatformError(Exception):
    pass


class NotFoundError(PlatformError):
    pass


class ValidationError(PlatformError):
    pass


class StageTransitionError(PlatformError):
    pass


class ModelStage(StrEnum):
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ArtifactType(StrEnum):
    PICKLE = "pickle"
    ONNX = "onnx"


@dataclass
class ModelVersion:
    model_name: str
    version: str
    artifact_type: ArtifactType
    artifact_uri: str
    stage: ModelStage = ModelStage.DEV
    tags: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, float] = field(default_factory=dict)
    is_warm: bool = False
    id: str = field(default_factory=lambda: new_id("model"))
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)


@dataclass
class PredictionRecord:
    model_name: str
    model_version: str
    request_id: str
    features: dict[str, float]
    prediction: float
    confidence: float
    experiment_id: str | None = None
    ground_truth: float | None = None
    id: str = field(default_factory=lambda: new_id("pred"))
    created_at: datetime = field(default_factory=utcnow)


@dataclass
class ExperimentVariant:
    model_version: str
    traffic_percentage: int


@dataclass
class Experiment:
    model_name: str
    variants: list[ExperimentVariant]
    metric_name: str = "absolute_error"
    active: bool = True
    id: str = field(default_factory=lambda: new_id("exp"))
    created_at: datetime = field(default_factory=utcnow)


@dataclass
class DriftReport:
    model_name: str
    model_version: str
    baseline_mean: float
    current_mean: float
    drift_score: float
    drift_detected: bool
    alert: str | None
    id: str = field(default_factory=lambda: new_id("drift"))
    created_at: datetime = field(default_factory=utcnow)
