from __future__ import annotations

import asyncio
import json
import socket
from typing import Any, Optional

import httpx
import websockets

from .models import StepResult, StudentPlannerAction, StudentPlannerState


class StudentPlannerEnv:
    """Async client for interacting with a Student Planner OpenEnv server."""

    def __init__(self, base_url: str, task_name: Optional[str] = None, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.task_name = task_name
        self.timeout = timeout
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)
        self._ws: Optional[Any] = None
        self._managed_container_id: Optional[str] = None

    async def __aenter__(self) -> "StudentPlannerEnv":
        await self._ensure_ws()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def reset(self, task_name: Optional[str] = None, seed: Optional[int] = None) -> StepResult:
        payload = {
            "type": "reset",
            "task_name": task_name or self.task_name,
            "seed": seed,
        }
        response = await self._send_ws_message(payload)
        return StepResult.model_validate(response)

    async def step(self, action: StudentPlannerAction | dict) -> StepResult:
        if isinstance(action, StudentPlannerAction):
            action_payload = action.model_dump(mode="json")
        else:
            action_payload = action
        response = await self._send_ws_message({"type": "step", "action": action_payload})
        return StepResult.model_validate(response)

    async def state(self) -> StudentPlannerState:
        response = await self._send_ws_message({"type": "state"})
        return StudentPlannerState.model_validate(response)

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

        await self._http.aclose()

        if self._managed_container_id:
            await self._docker_rm_force(self._managed_container_id)
            self._managed_container_id = None

    @classmethod
    async def from_docker_image(
        cls,
        image_name: str,
        task_name: Optional[str] = None,
        host_port: Optional[int] = None,
        startup_timeout: float = 45.0,
    ) -> "StudentPlannerEnv":
        if not image_name:
            raise ValueError("image_name is required")

        port = host_port or cls._find_free_port()
        process = await asyncio.create_subprocess_exec(
            "docker",
            "run",
            "-d",
            "-p",
            f"{port}:7860",
            image_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            error = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"docker run failed: {error}")

        container_id = stdout.decode("utf-8", errors="replace").strip()
        env = cls(base_url=f"http://127.0.0.1:{port}", task_name=task_name)
        env._managed_container_id = container_id
        await env._wait_for_health(timeout_seconds=startup_timeout)
        return env

    @classmethod
    async def from_env(cls, repo_id: str, task_name: Optional[str] = None) -> "StudentPlannerEnv":
        if "/" not in repo_id:
            raise ValueError("repo_id must be in '<namespace>/<name>' format")
        image = f"registry.hf.space/{repo_id.replace('/', '-')}:latest"
        return await cls.from_docker_image(image_name=image, task_name=task_name)

    async def _ensure_ws(self) -> None:
        if self._ws is not None:
            try:
                pong_waiter = await self._ws.ping()
                await pong_waiter
                return
            except Exception:
                try:
                    await self._ws.close()
                except Exception:
                    pass
                self._ws = None
        self._ws = await websockets.connect(self._ws_url(), ping_interval=20, ping_timeout=20)

    async def _send_ws_message(self, payload: dict) -> dict:
        await self._ensure_ws()
        assert self._ws is not None

        await self._ws.send(json.dumps(payload))
        raw = await self._ws.recv()
        data = json.loads(raw)

        message_type = data.get("type")
        if message_type == "error":
            raise RuntimeError(data.get("error", "unknown websocket error"))

        if message_type != "result":
            raise RuntimeError(f"unexpected websocket response type: {message_type}")

        return data.get("payload", {})

    async def _wait_for_health(self, timeout_seconds: float) -> None:
        deadline = asyncio.get_event_loop().time() + timeout_seconds
        while asyncio.get_event_loop().time() < deadline:
            try:
                response = await self._http.get("/health")
                if response.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(1.0)

        raise TimeoutError(f"server at {self.base_url} did not become healthy in time")

    @staticmethod
    async def _docker_rm_force(container_id: str) -> None:
        process = await asyncio.create_subprocess_exec(
            "docker",
            "rm",
            "-f",
            container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()

    def _ws_url(self) -> str:
        if self.base_url.startswith("https://"):
            return "wss://" + self.base_url[len("https://") :] + "/ws"
        if self.base_url.startswith("http://"):
            return "ws://" + self.base_url[len("http://") :] + "/ws"
        raise ValueError("base_url must start with http:// or https://")

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
