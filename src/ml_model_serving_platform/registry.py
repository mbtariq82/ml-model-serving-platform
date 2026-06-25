from __future__ import annotations

from ml_model_serving_platform.domain import (
    ArtifactType,
    ModelStage,
    ModelVersion,
    StageTransitionError,
    ValidationError,
    utcnow,
)
from ml_model_serving_platform.repository import ModelRepository


class ModelRegistryService:
    def __init__(self, repository: ModelRepository) -> None:
        self.repository = repository

    def register(
        self,
        model_name: str,
        version: str,
        artifact_type: ArtifactType,
        artifact_uri: str,
        tags: dict[str, str] | None = None,
        parameters: dict[str, float] | None = None,
    ) -> ModelVersion:
        if not model_name.strip():
            raise ValidationError("model_name is required.")
        if not version.strip():
            raise ValidationError("version is required.")
        if not artifact_uri.strip():
            raise ValidationError("artifact_uri is required.")

        model = ModelVersion(
            model_name=model_name.strip(),
            version=version.strip(),
            artifact_type=artifact_type,
            artifact_uri=artifact_uri.strip(),
            tags=tags or {},
            parameters=parameters or {},
        )
        return self.repository.save_model(model)

    def transition(self, model_name: str, version: str, target_stage: ModelStage) -> ModelVersion:
        model = self.repository.get_model(model_name, version)
        if not self._can_transition(model.stage, target_stage):
            raise StageTransitionError(
                f"Cannot transition from {model.stage.value} to {target_stage.value}."
            )
        model.stage = target_stage
        model.updated_at = utcnow()
        return self.repository.save_model(model)

    @staticmethod
    def _can_transition(current: ModelStage, target: ModelStage) -> bool:
        allowed = {
            ModelStage.DEV: {ModelStage.STAGING, ModelStage.ARCHIVED},
            ModelStage.STAGING: {ModelStage.PRODUCTION, ModelStage.DEV, ModelStage.ARCHIVED},
            ModelStage.PRODUCTION: {ModelStage.ARCHIVED},
            ModelStage.ARCHIVED: set(),
        }
        return target in allowed[current]
