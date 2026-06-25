from __future__ import annotations

from threading import RLock

from ml_model_serving_platform.domain import (
    Experiment,
    ModelStage,
    ModelVersion,
    NotFoundError,
    PredictionRecord,
)


class ModelRepository:
    def __init__(self) -> None:
        self._models: dict[tuple[str, str], ModelVersion] = {}
        self._predictions: dict[str, PredictionRecord] = {}
        self._experiments: dict[str, Experiment] = {}
        self._lock = RLock()

    def save_model(self, model: ModelVersion) -> ModelVersion:
        with self._lock:
            self._models[(model.model_name, model.version)] = model
            return model

    def get_model(self, model_name: str, version: str) -> ModelVersion:
        with self._lock:
            try:
                return self._models[(model_name, version)]
            except KeyError as exc:
                raise NotFoundError(f"Model {model_name!r} version {version!r} was not found.") from exc

    def list_models(self, model_name: str | None = None) -> list[ModelVersion]:
        with self._lock:
            models = list(self._models.values())
        if model_name is None:
            return models
        return [model for model in models if model.model_name == model_name]

    def production_model(self, model_name: str) -> ModelVersion:
        candidates = [
            model
            for model in self.list_models(model_name)
            if model.stage is ModelStage.PRODUCTION
        ]
        if not candidates:
            raise NotFoundError(f"No production model exists for {model_name!r}.")
        return max(candidates, key=lambda model: model.updated_at)

    def save_prediction(self, prediction: PredictionRecord) -> PredictionRecord:
        with self._lock:
            self._predictions[prediction.id] = prediction
            return prediction

    def get_prediction(self, prediction_id: str) -> PredictionRecord:
        with self._lock:
            try:
                return self._predictions[prediction_id]
            except KeyError as exc:
                raise NotFoundError(f"Prediction {prediction_id!r} was not found.") from exc

    def list_predictions(
        self,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> list[PredictionRecord]:
        with self._lock:
            predictions = list(self._predictions.values())
        if model_name:
            predictions = [item for item in predictions if item.model_name == model_name]
        if model_version:
            predictions = [item for item in predictions if item.model_version == model_version]
        return predictions

    def save_experiment(self, experiment: Experiment) -> Experiment:
        with self._lock:
            self._experiments[experiment.id] = experiment
            return experiment

    def get_experiment(self, experiment_id: str) -> Experiment:
        with self._lock:
            try:
                return self._experiments[experiment_id]
            except KeyError as exc:
                raise NotFoundError(f"Experiment {experiment_id!r} was not found.") from exc

    def active_experiment_for_model(self, model_name: str) -> Experiment | None:
        with self._lock:
            experiments = list(self._experiments.values())
        return next(
            (experiment for experiment in experiments if experiment.model_name == model_name and experiment.active),
            None,
        )
