from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from ml_model_serving_platform.domain import (
    ExperimentVariant,
    NotFoundError,
    PlatformError,
    StageTransitionError,
    ValidationError,
)
from ml_model_serving_platform.experiments import ExperimentService
from ml_model_serving_platform.inference import InferenceService
from ml_model_serving_platform.monitoring import MonitoringService
from ml_model_serving_platform.registry import ModelRegistryService
from ml_model_serving_platform.repository import ModelRepository
from ml_model_serving_platform.schemas import (
    BatchPredictRequest,
    CreateExperimentRequest,
    DistributionResponse,
    DriftReportResponse,
    DriftRequest,
    EvaluationResponse,
    ExperimentPredictRequest,
    ExperimentResponse,
    GroundTruthRequest,
    HealthResponse,
    ModelVersionResponse,
    PredictRequest,
    PredictionResponse,
    RegisterModelRequest,
    RetrainingResponse,
    TransitionModelRequest,
)


def create_app(repository: ModelRepository | None = None) -> FastAPI:
    repository = repository or ModelRepository()
    registry_service = ModelRegistryService(repository)
    inference_service = InferenceService(repository)
    monitoring_service = MonitoringService(repository)
    experiment_service = ExperimentService(
        repository,
        inference_service,
        registry_service,
    )

    app = FastAPI(
        title="ML Model Serving Platform",
        version="0.1.0",
        summary="Model registry, inference, experiments, and monitoring API.",
    )
    app.state.repository = repository
    app.state.registry_service = registry_service
    app.state.inference_service = inference_service
    app.state.monitoring_service = monitoring_service
    app.state.experiment_service = experiment_service

    @app.exception_handler(PlatformError)
    async def handle_platform_error(_request: Request, exc: PlatformError) -> JSONResponse:
        status_code = status.HTTP_400_BAD_REQUEST
        if isinstance(exc, NotFoundError):
            status_code = status.HTTP_404_NOT_FOUND
        elif isinstance(exc, StageTransitionError):
            status_code = status.HTTP_409_CONFLICT
        elif isinstance(exc, ValidationError):
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.post("/models", status_code=status.HTTP_201_CREATED, response_model=ModelVersionResponse)
    def register_model(payload: RegisterModelRequest) -> ModelVersionResponse:
        model = registry_service.register(
            payload.model_name,
            payload.version,
            payload.artifact_type,
            payload.artifact_uri,
            tags=payload.tags,
            parameters=payload.parameters,
        )
        return ModelVersionResponse.model_validate(model)

    @app.get("/models", response_model=list[ModelVersionResponse])
    def list_models(model_name: str | None = None) -> list[ModelVersionResponse]:
        return [
            ModelVersionResponse.model_validate(model)
            for model in repository.list_models(model_name)
        ]

    @app.get("/models/{model_name}/versions/{version}", response_model=ModelVersionResponse)
    def get_model(model_name: str, version: str) -> ModelVersionResponse:
        return ModelVersionResponse.model_validate(repository.get_model(model_name, version))

    @app.post("/models/{model_name}/versions/{version}/stage", response_model=ModelVersionResponse)
    def transition_model(
        model_name: str,
        version: str,
        payload: TransitionModelRequest,
    ) -> ModelVersionResponse:
        return ModelVersionResponse.model_validate(
            registry_service.transition(model_name, version, payload.stage)
        )

    @app.post("/models/{model_name}/versions/{version}/warm", response_model=ModelVersionResponse)
    def warm_model(model_name: str, version: str) -> ModelVersionResponse:
        return ModelVersionResponse.model_validate(inference_service.warm_model(model_name, version))

    @app.post("/models/{model_name}/predict", response_model=PredictionResponse)
    def predict(model_name: str, payload: PredictRequest) -> PredictionResponse:
        prediction = inference_service.predict(
            model_name,
            payload.features,
            version=payload.version,
            request_id=payload.request_id,
        )
        return PredictionResponse.model_validate(prediction)

    @app.post("/models/{model_name}/batch-predict", response_model=list[PredictionResponse])
    def batch_predict(model_name: str, payload: BatchPredictRequest) -> list[PredictionResponse]:
        return [
            PredictionResponse.model_validate(prediction)
            for prediction in inference_service.batch_predict(
                model_name,
                payload.feature_rows,
                version=payload.version,
            )
        ]

    @app.post("/experiments", status_code=status.HTTP_201_CREATED, response_model=ExperimentResponse)
    def create_experiment(payload: CreateExperimentRequest) -> ExperimentResponse:
        variants = [
            ExperimentVariant(
                model_version=variant.model_version,
                traffic_percentage=variant.traffic_percentage,
            )
            for variant in payload.variants
        ]
        return ExperimentResponse.model_validate(
            experiment_service.create_experiment(payload.model_name, variants)
        )

    @app.post("/experiments/{experiment_id}/predict", response_model=PredictionResponse)
    def experiment_predict(
        experiment_id: str,
        payload: ExperimentPredictRequest,
    ) -> PredictionResponse:
        return PredictionResponse.model_validate(
            experiment_service.predict(experiment_id, payload.entity_id, payload.features)
        )

    @app.post("/experiments/{experiment_id}/ground-truth", response_model=PredictionResponse)
    def record_ground_truth(
        experiment_id: str,
        payload: GroundTruthRequest,
    ) -> PredictionResponse:
        repository.get_experiment(experiment_id)
        return PredictionResponse.model_validate(
            experiment_service.record_ground_truth(payload.prediction_id, payload.ground_truth)
        )

    @app.get("/experiments/{experiment_id}/evaluation", response_model=EvaluationResponse)
    def evaluate_experiment(experiment_id: str) -> EvaluationResponse:
        return EvaluationResponse(**experiment_service.evaluate(experiment_id))

    @app.post("/experiments/{experiment_id}/auto-promote", response_model=EvaluationResponse)
    def auto_promote(experiment_id: str) -> EvaluationResponse:
        return EvaluationResponse(**experiment_service.auto_promote_winner(experiment_id))

    @app.get(
        "/models/{model_name}/versions/{version}/distribution",
        response_model=DistributionResponse,
    )
    def distribution(model_name: str, version: str) -> DistributionResponse:
        return DistributionResponse(**monitoring_service.prediction_distribution(model_name, version))

    @app.post(
        "/models/{model_name}/versions/{version}/drift",
        response_model=DriftReportResponse,
    )
    def detect_drift(model_name: str, version: str, payload: DriftRequest) -> DriftReportResponse:
        return DriftReportResponse.model_validate(
            monitoring_service.detect_drift(
                model_name,
                version,
                payload.baseline_mean,
                payload.threshold,
            )
        )

    @app.post(
        "/models/{model_name}/versions/{version}/retraining",
        response_model=RetrainingResponse,
    )
    def trigger_retraining(model_name: str, version: str, payload: DriftRequest) -> RetrainingResponse:
        report = monitoring_service.detect_drift(
            model_name,
            version,
            payload.baseline_mean,
            payload.threshold,
        )
        return RetrainingResponse(**monitoring_service.trigger_retraining(report))

    return app


app = create_app()
