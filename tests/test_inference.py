from __future__ import annotations

from ml_model_serving_platform.domain import ArtifactType, ModelStage
from ml_model_serving_platform.inference import InferenceService
from ml_model_serving_platform.registry import ModelRegistryService
from ml_model_serving_platform.repository import ModelRepository


def build_services() -> tuple[ModelRepository, ModelRegistryService, InferenceService]:
    repository = ModelRepository()
    return repository, ModelRegistryService(repository), InferenceService(repository)


def register_production_model(
    registry: ModelRegistryService,
    model_name: str = "risk-model",
    version: str = "1.0.0",
) -> None:
    registry.register(
        model_name,
        version,
        ArtifactType.PICKLE,
        f"s3://models/{model_name}/{version}.pkl",
        parameters={"multiplier": 2.0, "bias": 1.0},
    )
    registry.transition(model_name, version, ModelStage.STAGING)
    registry.transition(model_name, version, ModelStage.PRODUCTION)


def test_predict_loads_production_model_and_warms_it() -> None:
    repository, registry, inference = build_services()
    register_production_model(registry)

    prediction = inference.predict("risk-model", {"income": 10, "debt": 2}, request_id="req-1")

    assert prediction.model_version == "1.0.0"
    assert prediction.prediction == 25
    assert prediction.request_id == "req-1"
    assert repository.get_model("risk-model", "1.0.0").is_warm is True


def test_batch_predict_returns_one_prediction_per_row() -> None:
    _repository, registry, inference = build_services()
    register_production_model(registry)

    predictions = inference.batch_predict(
        "risk-model",
        [{"x": 1}, {"x": 2}, {"x": 3}],
    )

    assert [prediction.prediction for prediction in predictions] == [3, 5, 7]
