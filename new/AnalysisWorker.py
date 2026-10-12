
"""
Analysis Queue + Analysis Worker

职责拆分：
- AnalysisQueue  : 带优先级的异步队列 + 背压 + 去重
- AnalysisWorker : 从队列取任务，执行分析，回调 StateManager 生命周期
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from StateManager import SymbolState, StateManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 任务包装（带优先级）
# ---------------------------------------------------------------------------

@dataclass(order=True)
class AnalysisTask:
    """优先级队列元素。score 越高越优先（用 -score 实现）。"""
    priority: float
    enqueued_at: float
    state: SymbolState = field(compare=False)

    @classmethod
    def from_symbol_state(cls, state: SymbolState) -> "AnalysisTask":
        return cls(
            priority=-state.score,          # 负数 → 分数高的先出队
            enqueued_at=time.time(),
            state=state,
        )


# 分析函数类型：接收 SymbolState，返回任意结果（dict / 模型输出等）
AnalyzeFn = Callable[[SymbolState], Awaitable[Any] | Any]
ResultCallback = Callable[[SymbolState, Any], Awaitable[None] | None]


# ---------------------------------------------------------------------------
# Analysis Queue
# ---------------------------------------------------------------------------

class AnalysisQueue:
    """带优先级、去重、背压的异步分析队列。

    Args:
        maxsize: 队列最大长度。0 表示无限。满时 put 会等待（背压）。
        dedup: 同一 symbol 已在队列中时，是否拒绝重复入队。
    """

    def __init__(self, maxsize: int = 0, dedup: bool = True) -> None:
        self._queue: asyncio.PriorityQueue[AnalysisTask] = asyncio.PriorityQueue(
            maxsize=maxsize
        )
        self._dedup = dedup
        self._pending: set[str] = set()          # 已在队列中的 symbol
        self._lock = asyncio.Lock()

    async def put(self, state: SymbolState) -> bool:
        """入队。返回 True 表示成功，False 表示因去重被丢弃。"""
        async with self._lock:
            if self._dedup and state.symbol in self._pending:
                logger.debug("Dedup skip: %s already in queue", state.symbol)
                return False
            self._pending.add(state.symbol)

        task = AnalysisTask.from_symbol_state(state)
        await self._queue.put(task)
        logger.debug(
            "Enqueued %s score=%.2f queue_size=%d",
            state.symbol,
            state.score,
            self._queue.qsize(),
        )
        return True

    async def get(self) -> AnalysisTask:
        """取出优先级最高的任务。"""
        task = await self._queue.get()
        return task

    async def task_done(self, symbol: str) -> None:
        """Worker 处理完后必须调用，用于去重集合清理 + join 计数。"""
        async with self._lock:
            self._pending.discard(symbol)
        self._queue.task_done()

    def qsize(self) -> int:
        return self._queue.qsize()

    def pending_symbols(self) -> set[str]:
        return set(self._pending)

    async def join(self) -> None:
        """等待队列全部处理完。"""
        await self._queue.join()


# ---------------------------------------------------------------------------
# Analysis Worker
# ---------------------------------------------------------------------------

class AnalysisWorker:
    """从 AnalysisQueue 取任务并执行分析。

    Args:
        worker_id: 编号，用于日志。
        queue: 共享的 AnalysisQueue。
        state_manager: 用于 mark_analyzing / mark_cooldown。
        analyze_fn: 实际分析逻辑（EMA / ATR / ML 等）。
        on_result: 分析完成后的回调（写库、发信号等）。
        poll_timeout: get 超时时间，便于优雅退出。
    """

    def __init__(
        self,
        worker_id: int,
        queue: AnalysisQueue,
        state_manager: StateManager,
        analyze_fn: AnalyzeFn,
        on_result: Optional[ResultCallback] = None,
        poll_timeout: float = 1.0,
    ) -> None:
        self.worker_id = worker_id
        self.queue = queue
        self.state_manager = state_manager
        self.analyze_fn = analyze_fn
        self.on_result = on_result
        self.poll_timeout = poll_timeout

        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(
            self._loop(), name=f"AnalysisWorker-{self.worker_id}"
        )
        logger.info("Worker-%d started", self.worker_id)

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Worker-%d stopped", self.worker_id)

    async def _loop(self) -> None:
        while self._running:
            try:
                task = await asyncio.wait_for(
                    self.queue.get(), timeout=self.poll_timeout
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            state = task.state
            symbol = state.symbol

            try:
                await self.state_manager.mark_analyzing(symbol)
                logger.info(
                    "Worker-%d analyzing %s score=%.2f",
                    self.worker_id,
                    symbol,
                    state.score,
                )

                result = await self._run_analyze(state)

                if self.on_result is not None:
                    await self._invoke(self.on_result, state, result)

            except Exception:
                logger.exception(
                    "Worker-%d failed on %s", self.worker_id, symbol
                )
            finally:
                await self.state_manager.mark_cooldown(symbol)
                await self.queue.task_done(symbol)

    async def _run_analyze(self, state: SymbolState) -> Any:
        result = self.analyze_fn(state)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    @staticmethod
    async def _invoke(callback: Callable, *args: Any) -> None:
        try:
            result = callback(*args)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception("Result callback error")


# ---------------------------------------------------------------------------
# Worker Pool 便捷封装
# ---------------------------------------------------------------------------

class AnalysisWorkerPool:
    """管理 1~N 个 AnalysisWorker。"""

    def __init__(
        self,
        queue: AnalysisQueue,
        state_manager: StateManager,
        analyze_fn: AnalyzeFn,
        on_result: Optional[ResultCallback] = None,
        num_workers: int = 2,
    ) -> None:
        if not 1 <= num_workers <= 4:
            raise ValueError("num_workers should be 1~4")

        self.workers = [
            AnalysisWorker(
                worker_id=i + 1,
                queue=queue,
                state_manager=state_manager,
                analyze_fn=analyze_fn,
                on_result=on_result,
            )
            for i in range(num_workers)
        ]

    async def start(self) -> None:
        for w in self.workers:
            await w.start()

    async def stop(self) -> None:
        await asyncio.gather(*(w.stop() for w in self.workers))


# ======================================================================
# 占位分析函数（下一步换成真正的 EMA / ATR / ML）
# ======================================================================











