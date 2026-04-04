"""FastAPI application for SmartOps AI."""

from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv-core is required. Install dependencies with `pip install -r requirements.txt`."
    ) from exc

try:
    from smartops_ai_env.models import SmartOpsAction, SmartOpsObservation
    from smartops_ai_env.server.environment import SmartOpsEnvironment
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT.parent) not in sys.path:
        sys.path.insert(0, str(ROOT.parent))

    from smartops_ai_env.models import SmartOpsAction, SmartOpsObservation
    from smartops_ai_env.server.environment import SmartOpsEnvironment


app = create_app(
    SmartOpsEnvironment,
    SmartOpsAction,
    SmartOpsObservation,
    env_name="smartops_ai_env",
    max_concurrent_envs=4,
)


def main(host: str | None = None, port: int | None = None) -> None:
    """Entry point used by `python -m` and the OpenEnv validator."""

    import uvicorn
    import argparse

    if host is None or port is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=8000)
        args = parser.parse_args()
        host = args.host
        port = args.port

    assert host is not None
    assert port is not None
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
