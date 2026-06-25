from __future__ import annotations

from hashlib import sha256

from ml_model_serving_platform.domain import Experiment, ExperimentVariant, ModelStage, ValidationError
from ml_model_serving_platform.inference import InferenceService
from ml_model_serving_platform.registry import ModelRegistryService
from ml_model_serving_platform.repository import ModelRepository


class ExperimentService:
    def __init__(
        self,
        repository: ModelRepository,
        inference_service: InferenceService,
        registry_service: ModelRegistryService,
    ) -> None:
        self.repository = repository
        self.inference_service = inference_service
        self.registry_service = registry_service

    def create_experiment(
        self,
        model_name: str,
        variants: list[ExperimentVariant],
    ) -> Experiment:
        if sum(variant.traffic_percentage for variant in variants) != 100:
            raise ValidationError("Experiment traffic percentages must sum to 100.")
        for variant in variants:
            self.repository.get_model(model_name, variant.model_version)
        return self.repository.save_experiment(Experiment(model_name=model_name, variants=variants))

    def predict(
        self,
        experiment_id: str,
        entity_id: str,
        features: dict[str, float],
    ):
        experiment = self.repository.get_experiment(experiment_id)
        if not experiment.active:
            raise ValidationError("Experiment is not active.")
        version = self._choose_variant(experiment, entity_id)
        return self.inference_service.predict(
            experiment.model_name,
            features,
            version=version,
            experiment_id=experiment.id,
        )

    def record_ground_truth(self, prediction_id: str, ground_truth: float):
        prediction = self.repository.get_prediction(prediction_id)
        prediction.ground_truth = ground_truth
        return prediction

    def evaluate(self, experiment_id: str) -> dict[str, object]:
        experiment = self.repository.get_experiment(experiment_id)
        predictions = [
            prediction
            for prediction in self.repository.list_predictions(model_name=experiment.model_name)
            if prediction.experiment_id == experiment.id and prediction.ground_truth is not None
        ]
        by_version: dict[str, list[float]] = {variant.model_version: [] for variant in experiment.variants}
        for prediction in predictions:
            ground_truth = prediction.ground_truth
            if ground_truth is not None:
                by_version[prediction.model_version].append(abs(prediction.prediction - ground_truth))

        scores = {
            version: sum(errors) / len(errors)
            for version, errors in by_version.items()
            if errors
        }
        winner = min(scores, key=scores.get) if len(scores) >= 2 else None
        significant = False
        if winner:
            ordered = sorted(scores.values())
            significant = len(ordered) > 1 and ordered[1] - ordered[0] >= 0.05
        return {"scores": scores, "winner": winner, "significant": significant}

    def auto_promote_winner(self, experiment_id: str) -> dict[str, object]:
        result = self.evaluate(experiment_id)
        winner = result["winner"]
        if not winner or not result["significant"]:
            return {"promoted": False, **result}
        experiment = self.repository.get_experiment(experiment_id)
        model = self.repository.get_model(experiment.model_name, str(winner))
        if model.stage is ModelStage.DEV:
            self.registry_service.transition(model.model_name, model.version, ModelStage.STAGING)
        if model.stage is ModelStage.STAGING:
            self.registry_service.transition(model.model_name, model.version, ModelStage.PRODUCTION)
        experiment.active = False
        return {"promoted": True, **result}

    @staticmethod
    def _choose_variant(experiment: Experiment, entity_id: str) -> str:
        bucket = int(sha256(entity_id.encode()).hexdigest(), 16) % 100
        running_total = 0
        for variant in experiment.variants:
            running_total += variant.traffic_percentage
            if bucket < running_total:
                return variant.model_version
        return experiment.variants[-1].model_version
