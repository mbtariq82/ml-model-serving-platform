from __future__ import annotations

from hashlib import sha256

from ml_model_serving_platform.domain import ModelVersion, PredictionRecord, ValidationError
from ml_model_serving_platform.repository import ModelRepository


class InferenceService:
    def __init__(self, repository: ModelRepository) -> None:
        self.repository = repository

    def warm_model(self, model_name: str, version: str) -> ModelVersion:
        model = self.repository.get_model(model_name, version)
        model.is_warm = True
        return self.repository.save_model(model)

    def predict(
        self,
        model_name: str,
        features: dict[str, float],
        version: str | None = None,
        request_id: str | None = None,
        experiment_id: str | None = None,
    ) -> PredictionRecord:
        model = (
            self.repository.get_model(model_name, version)
            if version
            else self.repository.production_model(model_name)
        )
        if not model.is_warm:
            self.warm_model(model.model_name, model.version)
        prediction_value = self._run_model(model, features)
        confidence = max(0.5, min(0.99, 1 - abs(prediction_value) / 1000))
        record = PredictionRecord(
            model_name=model.model_name,
            model_version=model.version,
            request_id=request_id or self._request_id(model.model_name, model.version, features),
            features=features,
            prediction=round(prediction_value, 4),
            confidence=round(confidence, 4),
            experiment_id=experiment_id,
        )
        return self.repository.save_prediction(record)

    def batch_predict(
        self,
        model_name: str,
        feature_rows: list[dict[str, float]],
        version: str | None = None,
    ) -> list[PredictionRecord]:
        return [
            self.predict(model_name, features, version=version)
            for features in feature_rows
        ]

    @staticmethod
    def _run_model(model: ModelVersion, features: dict[str, float]) -> float:
        if not features:
            raise ValidationError("features must not be empty.")
        multiplier = model.parameters.get("multiplier", 1.0)
        bias = model.parameters.get("bias", 0.0)
        return sum(float(value) for value in features.values()) * multiplier + bias

    @staticmethod
    def _request_id(model_name: str, version: str, features: dict[str, float]) -> str:
        raw = f"{model_name}:{version}:{sorted(features.items())}".encode()
        return sha256(raw).hexdigest()[:16]
