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
from fastapi.responses import HTMLResponse

@app.get("/web", response_class=HTMLResponse)
async def web_ui():
    return """
    <html>
    <head>
        <title>SmartOps AI UI</title>
    </head>
    <body style="font-family: Arial; padding: 20px;">
        <h1>🚀 SmartOps AI Environment</h1>

        <button onclick="resetEnv()">Reset</button>
        <br><br>

        <input id="ticket" placeholder="Ticket ID (e.g. B-1001)" />
        <input id="category" placeholder="Category (billing)" />
        <button onclick="stepEnv()">Step</button>

        <h3>Response:</h3>
        <pre id="output"></pre>

        <script>
            async function resetEnv() {
                const res = await fetch('/reset', { method: 'POST' });
                const data = await res.json();
                document.getElementById('output').textContent = JSON.stringify(data, null, 2);
            }

            async function stepEnv() {
                const ticket = document.getElementById('ticket').value;
                const category = document.getElementById('category').value;

                const res = await fetch('/step', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        action: {
                            action_type: "classify_ticket",
                            category: category,
                            ticket_id: ticket
                        }
                    })
                });

                const data = await res.json();
                document.getElementById('output').textContent = JSON.stringify(data, null, 2);
            }
        </script>
    </body>
    </html>
    """