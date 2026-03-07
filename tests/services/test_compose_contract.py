from pathlib import Path

import yaml


def test_compose_defines_required_services():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    required = {
        "postgres",
        "redis",
        "redpanda",
        "minio",
        "minio-setup",
        "control-api",
        "market-data-service",
        "execution-service",
        "model-inference-service",
        "portfolio-service",
        "execution-policy-service",
        "replay-service",
        "risk-service",
    }
    assert required.issubset(services)
    assert "healthcheck" in services["postgres"]
    assert "healthcheck" in services["redis"]
