from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class InferenceTask:
    payload: Any  # whatever the infer_fn expects
    future: asyncio.Future
    enqueued_at: float = field(default_factory=time.monotonic)


class GPUQueue:
    def __init__(self, max_size: int = 8) -> None:
        self._queue: asyncio.Queue[InferenceTask] = asyncio.Queue(maxsize=max_size)
        self._worker_task: asyncio.Task | None = None

    def start(self, infer_fn: Callable) -> None:
        self._worker_task = asyncio.create_task(
            self._worker(infer_fn), name="gpu-worker"
        )
        logger.info("GPU worker started (maxsize=%d)", self._queue.maxsize)

    async def stop(self) -> None:
        if self._worker_task:
            await self._queue.join()
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("GPU worker stopped")

    async def submit(self, payload: Any) -> Any:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put(InferenceTask(payload=payload, future=future))
        return await future

    async def _worker(self, infer_fn: Callable) -> None:
        while True:
            task = await self._queue.get()
            queue_ms = (time.monotonic() - task.enqueued_at) * 1000
            task.payload.queue_ms = queue_ms
            try:
                result = await asyncio.to_thread(infer_fn, task.payload)
                task.future.set_result(result)
            except Exception as exc:
                task.future.set_exception(exc)
            finally:
                self._queue.task_done()
