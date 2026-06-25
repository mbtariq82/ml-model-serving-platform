from __future__ import annotations

from fastapi.testclient import TestClient

from ml_model_serving_platform.main import create_app


def register_model(client: TestClient, version: str, multiplier: float) -> None:
    response = client.post(
        "/models",
        json={
            "model_name": "risk-model",
            "version": version,
            "artifact_type": "pickle",
            "artifact_uri": f"s3://models/risk/{version}.pkl",
            "parameters": {"multiplier": multiplier},
        },
    )
    assert response.status_code == 201


def promote_model(client: TestClient, version: str) -> None:
    assert client.post(
        f"/models/risk-model/versions/{version}/stage",
        json={"stage": "staging"},
    ).status_code == 200
    assert client.post(
        f"/models/risk-model/versions/{version}/stage",
        json={"stage": "production"},
    ).status_code == 200


def test_api_registers_promotes_and_predicts() -> None:
    client = TestClient(create_app())
    register_model(client, "1.0.0", multiplier=2)
    promote_model(client, "1.0.0")

    prediction = client.post(
        "/models/risk-model/predict",
        json={"features": {"income": 10, "debt": 2}, "request_id": "req-1"},
    )

    assert prediction.status_code == 200
    assert prediction.json()["prediction"] == 24
    assert client.get("/models/risk-model/versions/1.0.0").json()["is_warm"] is True


def test_api_serves_experiments_and_monitoring() -> None:
    client = TestClient(create_app())
    register_model(client, "1.0.0", multiplier=1)
    register_model(client, "2.0.0", multiplier=2)
    promote_model(client, "1.0.0")

    experiment = client.post(
        "/experiments",
        json={
            "model_name": "risk-model",
            "variants": [
                {"model_version": "1.0.0", "traffic_percentage": 50},
                {"model_version": "2.0.0", "traffic_percentage": 50},
            ],
        },
    )
    assert experiment.status_code == 201
    prediction = client.post(
        f"/experiments/{experiment.json()['id']}/predict",
        json={"entity_id": "user-1", "features": {"x": 5}},
    )
    assert prediction.status_code == 200

    batch = client.post(
        "/models/risk-model/batch-predict",
        json={"feature_rows": [{"x": 3}, {"x": 4}], "version": "1.0.0"},
    )
    assert batch.status_code == 200
    distribution = client.get("/models/risk-model/versions/1.0.0/distribution")
    assert distribution.status_code == 200
    assert distribution.json()["count"] >= 2
