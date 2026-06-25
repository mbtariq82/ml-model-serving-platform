from __future__ import annotations

from ml_model_serving_platform.domain import ArtifactType, ExperimentVariant, ModelStage
from ml_model_serving_platform.experiments import ExperimentService
from ml_model_serving_platform.inference import InferenceService
from ml_model_serving_platform.registry import ModelRegistryService
from ml_model_serving_platform.repository import ModelRepository


def build_services() -> tuple[ModelRepository, ModelRegistryService, InferenceService, ExperimentService]:
    repository = ModelRepository()
    registry = ModelRegistryService(repository)
    inference = InferenceService(repository)
    experiments = ExperimentService(repository, inference, registry)
    return repository, registry, inference, experiments


def register_model(registry: ModelRegistryService, version: str, multiplier: float) -> None:
    registry.register(
        "risk-model",
        version,
        ArtifactType.PICKLE,
        f"s3://models/risk/{version}.pkl",
        parameters={"multiplier": multiplier},
    )


def test_experiment_routes_predictions_and_records_ground_truth() -> None:
    _repository, registry, _inference, experiments = build_services()
    register_model(registry, "1.0.0", multiplier=1)
    register_model(registry, "2.0.0", multiplier=2)
    experiment = experiments.create_experiment(
        "risk-model",
        [
            ExperimentVariant("1.0.0", 50),
            ExperimentVariant("2.0.0", 50),
        ],
    )

    prediction = experiments.predict(experiment.id, "user-1", {"x": 5})
    labelled = experiments.record_ground_truth(prediction.id, 10)

    assert prediction.experiment_id == experiment.id
    assert labelled.ground_truth == 10


def test_experiment_evaluation_auto_promotes_significant_winner() -> None:
    repository, registry, _inference, experiments = build_services()
    register_model(registry, "1.0.0", multiplier=1)
    register_model(registry, "2.0.0", multiplier=2)
    registry.transition("risk-model", "1.0.0", ModelStage.STAGING)
    registry.transition("risk-model", "1.0.0", ModelStage.PRODUCTION)
    experiment = experiments.create_experiment(
        "risk-model",
        [
            ExperimentVariant("1.0.0", 50),
            ExperimentVariant("2.0.0", 50),
        ],
    )

    seen_versions: set[str] = set()
    for index in range(40):
        prediction = experiments.predict(experiment.id, f"user-{index}", {"x": 5})
        seen_versions.add(prediction.model_version)
        ground_truth = 10 if prediction.model_version == "2.0.0" else 30
        experiments.record_ground_truth(prediction.id, ground_truth)
        if seen_versions == {"1.0.0", "2.0.0"} and index > 12:
            break

    evaluation = experiments.evaluate(experiment.id)
    promotion = experiments.auto_promote_winner(experiment.id)

    assert evaluation["winner"] == "2.0.0"
    assert evaluation["significant"] is True
    assert promotion["promoted"] is True
    assert repository.get_model("risk-model", "2.0.0").stage is ModelStage.PRODUCTION
