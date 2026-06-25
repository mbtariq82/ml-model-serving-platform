from __future__ import annotations

from ml_model_serving_platform.domain import ArtifactType, ModelStage
from ml_model_serving_platform.inference import InferenceService
from ml_model_serving_platform.monitoring import MonitoringService
from ml_model_serving_platform.registry import ModelRegistryService
from ml_model_serving_platform.repository import ModelRepository


def test_distribution_drift_and_retraining_trigger() -> None:
    repository = ModelRepository()
    registry = ModelRegistryService(repository)
    inference = InferenceService(repository)
    monitoring = MonitoringService(repository)
    registry.register(
        "risk-model",
        "1.0.0",
        ArtifactType.PICKLE,
        "s3://models/risk/1.0.0.pkl",
        parameters={"multiplier": 100},
    )
    registry.transition("risk-model", "1.0.0", ModelStage.STAGING)
    registry.transition("risk-model", "1.0.0", ModelStage.PRODUCTION)
    inference.batch_predict(
        "risk-model",
        [{"value": 4}, {"value": 5}, {"value": 6}],
    )

    distribution = monitoring.prediction_distribution("risk-model", "1.0.0")
    report = monitoring.detect_drift("risk-model", "1.0.0", baseline_mean=100, threshold=0.2)
    retraining = monitoring.trigger_retraining(report)

    assert distribution["count"] == 3
    assert report.drift_detected is True
    assert retraining["status"] == "queued"
