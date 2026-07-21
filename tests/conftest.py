import pytest


@pytest.fixture(autouse=True)
def reset_api_rate_limiter():
    """Keep SlowAPI's in-memory counters isolated between tests."""
    try:
        from app import api

        api.limiter._storage.reset()
    except (AttributeError, ImportError):
        pass
    yield
    try:
        api.limiter._storage.reset()
    except (AttributeError, UnboundLocalError):
        pass
