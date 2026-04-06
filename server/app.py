"""FastAPI application for SmartOps AI."""

from __future__ import annotations

from fastapi.responses import HTMLResponse

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


# ✅ IMPORTANT: set concurrency to 1
app = create_app(
    SmartOpsEnvironment,
    SmartOpsAction,
    SmartOpsObservation,
    env_name="smartops_ai_env",
    max_concurrent_envs=1,
)


# ✅ UI ROUTE (added properly)
@app.get("/", response_class=HTMLResponse)
@app.get("/web", response_class=HTMLResponse)
async def web_ui():
    return """
<html>
<head>
    <title>SmartOps AI Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            padding: 30px;
        }

        h1 {
            color: #38bdf8;
        }

        .card {
            background: #1e293b;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }

        input, select {
            padding: 10px;
            margin: 8px;
            border-radius: 6px;
            border: none;
            width: 200px;
        }

        button {
            padding: 10px 15px;
            margin: 8px;
            border-radius: 6px;
            border: none;
            cursor: pointer;
            background: #38bdf8;
            color: black;
            font-weight: bold;
        }

        button:hover {
            background: #0ea5e9;
        }

        .reset-btn {
            background: #22c55e;
        }

        .reset-btn:hover {
            background: #16a34a;
        }

        pre {
            background: #020617;
            padding: 15px;
            border-radius: 10px;
            color: #22c55e;
            overflow-x: auto;
        }
    </style>
</head>

<body>

    <h1>🚀 SmartOps AI Dashboard</h1>

    <div class="card">
        <h3>⚙️ Controls</h3>

        <button class="reset-btn" onclick="resetEnv()">🔄 Reset Environment</button>

        <br>

        <input id="ticket" value="B-1001" />

        <select id="category">
            <option value="billing" selected>billing</option>
            <option value="technical">technical</option>
            <option value="delivery">delivery</option>
            <option value="fraud">fraud</option>
        </select>

        <button onclick="stepEnv()">▶️ Run Step</button>
    </div>

    <div class="card">
        <h3>📊 Environment Response</h3>
        <pre id="output">Click reset to start...</pre>
    </div>

<script>
    async function resetEnv() {
        const res = await fetch('/reset', { method: 'POST' });
        const data = await res.json();
        document.getElementById('output').textContent = JSON.stringify(data, null, 2);
    }

    async function stepEnv() {
        const res = await fetch('/step', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                action: {
                    action_type: "classify_ticket",
                    category: document.getElementById('category').value,
                    ticket_id: document.getElementById('ticket').value
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