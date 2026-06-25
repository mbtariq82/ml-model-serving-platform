from __future__ import annotations

import pytest

from ml_model_serving_platform.domain import ArtifactType, ModelStage, StageTransitionError
from ml_model_serving_platform.registry import ModelRegistryService
from ml_model_serving_platform.repository import ModelRepository


def build_registry() -> ModelRegistryService:
    return ModelRegistryService(ModelRepository())


def test_register_model_version_with_metadata() -> None:
    registry = build_registry()

    model = registry.register(
        "risk-model",
        "1.0.0",
        ArtifactType.PICKLE,
        "s3://models/risk/1.0.0.pkl",
        tags={"team": "risk"},
        parameters={"multiplier": 2.0},
    )

    assert model.model_name == "risk-model"
    assert model.stage is ModelStage.DEV
    assert model.tags["team"] == "risk"


def test_valid_stage_transitions() -> None:
    repository = ModelRepository()
    registry = ModelRegistryService(repository)
    registry.register("risk-model", "1.0.0", ArtifactType.ONNX, "s3://models/risk.onnx")

    staging = registry.transition("risk-model", "1.0.0", ModelStage.STAGING)
    assert staging.stage is ModelStage.STAGING

    production = registry.transition("risk-model", "1.0.0", ModelStage.PRODUCTION)

    assert production.stage is ModelStage.PRODUCTION
    assert repository.production_model("risk-model").version == "1.0.0"


def test_invalid_stage_transition_is_rejected() -> None:
    registry = build_registry()
    registry.register("risk-model", "1.0.0", ArtifactType.ONNX, "s3://models/risk.onnx")

    with pytest.raises(StageTransitionError):
        registry.transition("risk-model", "1.0.0", ModelStage.PRODUCTION)
