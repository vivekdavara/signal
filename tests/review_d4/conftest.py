"""Review D4: the serving fixtures (a synthetic dataset, a small trained model, Redis under fresh test-<uuid>
namespaces), shared from tests/serving/conftest.py."""

from tests.serving.conftest import client, dataset, fresh, loaded, model, prepared  # noqa: F401
