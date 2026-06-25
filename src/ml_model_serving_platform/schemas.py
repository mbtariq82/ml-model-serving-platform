from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from ml_model_serving_platform.domain import ArtifactType, ModelStage


class HealthResponse(BaseModel):
    status: str


class RegisterModelRequest(BaseModel):
    model_name: str = Field(min_length=1, max_length=120)
    version: str = Field(min_length=1, max_length=80)
    artifact_type: ArtifactType
    artifact_uri: str = Field(min_length=1, max_length=500)
    tags: dict[str, str] = Field(default_factory=dict)
    parameters: dict[str, float] = Field(default_factory=dict)


class TransitionModelRequest(BaseModel):
    stage: ModelStage


class PredictRequest(BaseModel):
    features: dict[str, float]
    version: str | None = None
    request_id: str | None = None


class BatchPredictRequest(BaseModel):
    feature_rows: list[dict[str, float]]
    version: str | None = None


class ExperimentVariantRequest(BaseModel):
    model_version: str
    traffic_percentage: int = Field(ge=1, le=100)


class CreateExperimentRequest(BaseModel):
    model_name: str
    variants: list[ExperimentVariantRequest]


class ExperimentPredictRequest(BaseModel):
    entity_id: str
    features: dict[str, float]


class GroundTruthRequest(BaseModel):
    prediction_id: str
    ground_truth: float


class DriftRequest(BaseModel):
    baseline_mean: float
    threshold: float = Field(default=0.2, gt=0)


class ModelVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    model_name: str
    version: str
    artifact_type: ArtifactType
    artifact_uri: str
    stage: ModelStage
    tags: dict[str, str]
    parameters: dict[str, float]
    is_warm: bool
    created_at: datetime
    updated_at: datetime


class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    model_name: str
    model_version: str
    request_id: str
    features: dict[str, float]
    prediction: float
    confidence: float
    experiment_id: str | None
    ground_truth: float | None
    created_at: datetime


class ExperimentVariantResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_version: str
    traffic_percentage: int


class ExperimentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    model_name: str
    variants: list[ExperimentVariantResponse]
    metric_name: str
    active: bool
    created_at: datetime


class EvaluationResponse(BaseModel):
    scores: dict[str, float]
    winner: str | None
    significant: bool
    promoted: bool | None = None


class DistributionResponse(BaseModel):
    count: float
    mean_prediction: float
    mean_confidence: float
    min_prediction: float
    max_prediction: float


class DriftReportResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    model_name: str
    model_version: str
    baseline_mean: float
    current_mean: float
    drift_score: float
    drift_detected: bool
    alert: str | None
    created_at: datetime


class RetrainingResponse(BaseModel):
    status: str
    model_name: str | None = None
    model_version: str | None = None
    reason: str
