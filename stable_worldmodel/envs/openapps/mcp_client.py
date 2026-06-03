"""Synchronous client bridge to an OpenApps MCP server.

The MCP Python SDK is async; gymnasium ``Env`` is sync. This wraps a
stdio MCP client in a dedicated background event-loop thread and exposes
blocking helpers (``reset``/``act``/``reconfigure``/...), marshalling each
call onto the loop via ``run_coroutine_threadsafe``.

Process = session: each client spawns its own ``python -m open_apps.mcp``
subprocess, so N envs in one process are fully independent (no shared
Playwright/FastHTML singletons).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
import threading

import numpy as np
from PIL import Image

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _result(res):
    """Unwrap a scalar/list tool result (FastMCP wraps as structuredContent.result)."""
    sc = getattr(res, "structuredContent", None)
    if isinstance(sc, dict) and "result" in sc:
        return sc["result"]
    texts = [c.text for c in res.content if getattr(c, "type", None) == "text"]
    return texts


class OpenAppsMCPClient:
    """Blocking handle to one OpenApps MCP server subprocess."""

    def __init__(
        self,
        app_name: str,
        *,
        python_executable: str | None = None,
        server_args: list[str] | None = None,
        ready_timeout: float = 120.0,
    ) -> None:
        self.app_name = app_name
        self._python = python_executable or sys.executable
        self._extra_args = server_args or []
        self._loop = asyncio.new_event_loop()
        self._session: ClientSession | None = None
        self._shutdown: asyncio.Event | None = None
        self._ready = threading.Event()
        self._exc: BaseException | None = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=ready_timeout) or self._session is None:
            raise RuntimeError(
                f"OpenApps MCP server ({app_name}) failed to start: {self._exc}"
            )

    # -- event loop / lifecycle -------------------------------------------

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:  # pragma: no cover - surfaced via _exc
            self._exc = e
            self._ready.set()

    async def _serve(self) -> None:
        self._shutdown = asyncio.Event()
        params = StdioServerParameters(
            command=self._python,
            args=["-m", "open_apps.mcp", "--app", self.app_name, *self._extra_args],
            env=os.environ.copy(),
        )
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    await self._shutdown.wait()
        except Exception as e:
            self._exc = e
            self._ready.set()

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def call_tool(self, name: str, args: dict | None = None):
        if self._session is None:
            raise RuntimeError("MCP session not available")
        return self._run(self._session.call_tool(name, args or {}))

    def close(self) -> None:
        try:
            if self._shutdown is not None:
                self._loop.call_soon_threadsafe(self._shutdown.set)
            self._thread.join(timeout=10)
        except Exception:
            pass
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass

    # -- high-level helpers -----------------------------------------------

    def reset(self, seed: int | None = None):
        args = {} if seed is None else {"seed": int(seed)}
        return self._decode_obs(self.call_tool("reset", args))

    def reconfigure(self, **kwargs) -> None:
        args = {k: v for k, v in kwargs.items() if v is not None}
        self.call_tool("reconfigure", args)

    def act(self, action: str, with_reward: bool = True):
        return self._decode_obs(
            self.call_tool("act", {"action": action, "with_reward": with_reward})
        )

    def observe(self):
        return self._decode_obs(self.call_tool("observe", {}))

    def load_task(self, key: str) -> str:
        r = _result(self.call_tool("load_task", {"key": key}))
        return r if isinstance(r, str) else (r[0] if r else "")

    def list_tasks(self, app: str | None = None) -> list[str]:
        r = _result(self.call_tool("list_tasks", {"app": app} if app else {}))
        return list(r) if isinstance(r, list) else [r]

    def list_variants(self, app: str, group: str) -> list[str]:
        r = _result(self.call_tool("list_variants", {"app": app, "group": group}))
        return list(r) if isinstance(r, list) else [r]

    def list_apps(self) -> list[str]:
        r = _result(self.call_tool("list_apps", {}))
        return list(r) if isinstance(r, list) else [r]

    def get_reward(self) -> float:
        r = _result(self.call_tool("get_reward", {}))
        return float(r if not isinstance(r, list) else (r[0] if r else 0.0))

    # -- decoding ----------------------------------------------------------

    @staticmethod
    def _decode_obs(res):
        """-> (img: np.uint8 (H,W,3) | None, meta: dict) from an obs tool result."""
        png = None
        meta: dict = {}
        for c in res.content:
            ctype = getattr(c, "type", None)
            if ctype == "image":
                png = base64.b64decode(c.data)
            elif ctype == "text":
                try:
                    meta = json.loads(c.text)
                except Exception:
                    pass
        img = None
        if png is not None:
            img = np.asarray(
                Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.uint8
            )
        return img, meta
