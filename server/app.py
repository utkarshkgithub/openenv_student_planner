from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from student_planner.env import StudentPlannerCoreEnv
from student_planner.models import ResetRequest, StepRequest

app = FastAPI(title="Student Planner OpenEnv", version="0.1.0")

_http_env = StudentPlannerCoreEnv()


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "name": "student_planner",
        "status": "ok",
        "openenv": True,
        "endpoints": ["/health", "/reset", "/step", "/state", "/docs", "/web", "/ws"],
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy", "benchmark": "student_planner"}


@app.post("/reset")
async def reset(request: ResetRequest | None = None) -> Dict[str, Any]:
    resolved_request = request or ResetRequest()
    result = _http_env.reset(task_name=resolved_request.task_name, seed=resolved_request.seed)
    return result.model_dump(mode="json")


@app.post("/step")
async def step(request: StepRequest) -> Dict[str, Any]:
    result = _http_env.step(request.action)
    return result.model_dump(mode="json")


@app.get("/state")
async def state() -> Dict[str, Any]:
    return _http_env.state().model_dump(mode="json")


@app.get("/web")
async def web_ui() -> HTMLResponse:
    content = """
    <!doctype html>
    <html>
      <head>
        <meta charset=\"utf-8\" />
        <title>Student Planner OpenEnv</title>
        <style>
          body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin: 2rem; }
          pre { background: #f3f3f3; padding: 1rem; border-radius: 8px; overflow-x: auto; }
        </style>
      </head>
      <body>
        <h1>Student Planner OpenEnv</h1>
        <p>Use the API endpoints <code>/reset</code>, <code>/step</code>, and <code>/state</code>, or connect over <code>/ws</code>.</p>
        <pre>
POST /reset  {"task_name": "single_topic", "seed": 42}
POST /step   {"action": {"action_type": "study", "topic": "genetics", "duration": 20}}
GET  /state
        </pre>
      </body>
    </html>
    """
    return HTMLResponse(content=content)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session_env = StudentPlannerCoreEnv()

    try:
        while True:
            payload = await websocket.receive_json()
            message_type = payload.get("type")

            if message_type == "ping":
                await websocket.send_json({"type": "result", "payload": {"pong": True}})
                continue

            if message_type == "reset":
                result = session_env.reset(task_name=payload.get("task_name"), seed=payload.get("seed"))
                await websocket.send_json({"type": "result", "payload": result.model_dump(mode="json")})
                continue

            if message_type == "step":
                result = session_env.step(payload.get("action", {}))
                await websocket.send_json({"type": "result", "payload": result.model_dump(mode="json")})
                continue

            if message_type == "state":
                await websocket.send_json(
                    {"type": "result", "payload": session_env.state().model_dump(mode="json")}
                )
                continue

            await websocket.send_json(
                {"type": "error", "error": f"unsupported message type '{message_type}'"}
            )

    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json({"type": "error", "error": str(exc)})


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
