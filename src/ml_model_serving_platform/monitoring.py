from __future__ import annotations

from statistics import mean

from ml_model_serving_platform.domain import DriftReport, ValidationError
from ml_model_serving_platform.repository import ModelRepository


class MonitoringService:
    def __init__(self, repository: ModelRepository) -> None:
        self.repository = repository

    def prediction_distribution(self, model_name: str, version: str) -> dict[str, float]:
        predictions = self.repository.list_predictions(model_name, version)
        if not predictions:
            raise ValidationError("No predictions exist for this model version.")
        values = [prediction.prediction for prediction in predictions]
        confidences = [prediction.confidence for prediction in predictions]
        return {
            "count": float(len(values)),
            "mean_prediction": mean(values),
            "mean_confidence": mean(confidences),
            "min_prediction": min(values),
            "max_prediction": max(values),
        }

    def detect_drift(
        self,
        model_name: str,
        version: str,
        baseline_mean: float,
        threshold: float = 0.2,
    ) -> DriftReport:
        distribution = self.prediction_distribution(model_name, version)
        current_mean = distribution["mean_prediction"]
        denominator = abs(baseline_mean) or 1.0
        drift_score = abs(current_mean - baseline_mean) / denominator
        low_confidence = distribution["mean_confidence"] < 0.65
        drift_detected = drift_score >= threshold or low_confidence
        alert = None
        if drift_detected:
            alert = "Prediction drift or confidence degradation detected."
        return DriftReport(
            model_name=model_name,
            model_version=version,
            baseline_mean=baseline_mean,
            current_mean=current_mean,
            drift_score=round(drift_score, 4),
            drift_detected=drift_detected,
            alert=alert,
        )

    def trigger_retraining(self, report: DriftReport) -> dict[str, str]:
        if not report.drift_detected:
            return {"status": "skipped", "reason": "No drift detected."}
        return {
            "status": "queued",
            "model_name": report.model_name,
            "model_version": report.model_version,
            "reason": report.alert or "drift",
        }
