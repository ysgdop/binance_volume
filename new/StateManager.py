"""
State Manager

职责：
- 维护每个 symbol 的最新价格与状态
- Trigger 判定（价格变动 / 阈值突破等）
- Cooldown 控制（防止同一币短时间重复进入分析）
- Score 计算（优先级，供 Analysis Queue 排序）
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

class SymbolStatus(Enum):
    IDLE = auto()          # 正常跟踪中
    TRIGGERED = auto()     # 已触发，等待进入分析队列
    ANALYZING = auto()     # 正在被 Analysis Worker 处理
    COOLDOWN = auto()      # 冷却中


@dataclass
class SymbolState:
    symbol: str
    price: float = 0.0
    prev_price: float = 0.0
    price_change_pct: float = 0.0          # 相对 prev_price 的涨跌幅 %
    last_update_ts: float = 0.0            # 上次收到行情的时间
    last_trigger_ts: float = 0.0           # 上次触发时间
    status: SymbolStatus = SymbolStatus.IDLE
    score: float = 0.0                     # 优先级分数，越高越先分析
    meta: dict[str, Any] = field(default_factory=dict)  # 扩展字段


TriggerCallback = Callable[[SymbolState], Awaitable[None] | None]


# ---------------------------------------------------------------------------
# State Manager
# ---------------------------------------------------------------------------

class StateManager:
    """维护 symbol 状态，并决定何时触发分析。

    Args:
        price_change_threshold: 相对上次记录价格的涨幅阈值（%），超过才可能触发。
        cooldown_seconds: 同一 symbol 两次触发的最小间隔。
        min_score: score 低于此值不触发。
        on_trigger: 触发时的回调（通常把 symbol 丢进 Analysis Queue）。
    """

    def __init__(
        self,
        price_change_threshold: float = 0.5,
        cooldown_seconds: float = 300.0,      # 5 分钟
        min_score: float = 0.0, # 只做了 是否是涨 大于0
        on_trigger: Optional[TriggerCallback] = None,
    ) -> None:
        self.price_change_threshold = price_change_threshold
        self.cooldown_seconds = cooldown_seconds
        self.min_score = min_score
        self.on_trigger = on_trigger

        self._states: dict[str, SymbolState] = {} # 记录数据
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # 对外主接口（给 WebSocket Stream 调用）
    # ------------------------------------------------------------------

    async def on_ticker(self, item: dict[str, Any]) -> None:
        """WebSocket 每推一条 ticker 就调用这里。

        期望字段（Binance !ticker@arr）:
            s: symbol
            c: last price (str)
        """
        symbol = item.get("s")
        if not symbol:
            return

        try:
            price = float(item["c"])
        except (KeyError, TypeError, ValueError):
            return

        async with self._lock:
            state = self._get_or_create(symbol)
            self._update_price(state, price) # 价格

            if self._should_trigger(state):
                self._mark_triggered(state)
                # 回调放到锁外，避免死锁
                callback = self.on_trigger
                snapshot = self._snapshot(state)
            else:
                callback = None
                snapshot = None

        if callback is not None and snapshot is not None:
            await self._invoke(callback, snapshot)

    # ------------------------------------------------------------------
    # Universe 同步（给 Universe Manager / REST Scanner 调用）
    # ------------------------------------------------------------------

    async def sync_universe(self, symbols: list[str] | set[str]) -> None:
        """只保留当前 Universe 里的 symbol，其余清理掉。"""
        wanted = set(symbols)
        async with self._lock:
            existing = set(self._states.keys())
            for sym in existing - wanted:
                del self._states[sym]
            for sym in wanted - existing:
                self._states[sym] = SymbolState(symbol=sym)
        logger.info(
            "Universe synced: %d symbols (added %d, removed %d)",
            len(wanted),
            len(wanted - existing),
            len(existing - wanted),
        )

    # ------------------------------------------------------------------
    # Analysis Worker 生命周期回调
    # ------------------------------------------------------------------

    async def mark_analyzing(self, symbol: str) -> None:
        async with self._lock:
            state = self._states.get(symbol)
            if state:
                state.status = SymbolStatus.ANALYZING

    async def mark_cooldown(self, symbol: str) -> None:
        """分析结束（无论成功失败）后进入冷却。"""
        async with self._lock:
            state = self._states.get(symbol)
            if state:
                state.status = SymbolStatus.COOLDOWN
                state.last_trigger_ts = time.time()

    async def mark_idle(self, symbol: str) -> None:
        """冷却时间到了，或手动恢复。"""
        async with self._lock:
            state = self._states.get(symbol)
            if state:
                state.status = SymbolStatus.IDLE

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    async def get_state(self, symbol: str) -> Optional[SymbolState]:
        async with self._lock:
            state = self._states.get(symbol)
            return self._snapshot(state) if state else None

    async def get_all_states(self) -> list[SymbolState]:
        async with self._lock:
            return [self._snapshot(s) for s in self._states.values()]

    async def get_triggered(self) -> list[SymbolState]:
        """返回当前 status == TRIGGERED 的列表（按 score 降序）。"""
        async with self._lock:
            triggered = [
                self._snapshot(s)
                for s in self._states.values()
                if s.status == SymbolStatus.TRIGGERED
            ]
        triggered.sort(key=lambda x: x.score, reverse=True)
        return triggered

    # ------------------------------------------------------------------
    # 内部逻辑
    # ------------------------------------------------------------------

    def _get_or_create(self, symbol: str) -> SymbolState:
        if symbol not in self._states:
            self._states[symbol] = SymbolState(symbol=symbol)
        return self._states[symbol]

    def _update_price(self, state: SymbolState, price: float) -> None:
        now = time.time()

        if state.price > 0:
            state.prev_price = state.price
            state.price_change_pct = (price - state.price) / state.price * 100.0
        else:
            # 第一次收到价格，只记录不触发
            state.prev_price = price
            state.price_change_pct = 0.0

        state.price = price
        state.last_update_ts = now
        state.score = self._calc_score(state)

    def _should_trigger(self, state: SymbolState) -> bool:
        # 1. 状态必须是 IDLE（冷却中 / 分析中 / 已触发的不再重复）
        if state.status != SymbolStatus.IDLE:
            # 冷却时间到了自动回到 IDLE
            if state.status == SymbolStatus.COOLDOWN:
                if time.time() - state.last_trigger_ts >= self.cooldown_seconds:
                    state.status = SymbolStatus.IDLE
                else:
                    return False
            else:
                return False

        # 2. 价格涨幅阈值（这里只关心上涨，和你 Scanner 逻辑一致）
        if state.price_change_pct <= self.price_change_threshold:
            return False

        # 3. score 门槛  暂时多余
        if state.score <= self.min_score:
            return False

        return True

    def _mark_triggered(self, state: SymbolState) -> None:
        state.status = SymbolStatus.TRIGGERED
        state.last_trigger_ts = time.time()
        logger.info(
            "TRIGGER %s | price=%.6f change=%.3f%% score=%.2f",
            state.symbol,
            state.price,
            state.price_change_pct,
            state.score,
        )

    def _calc_score(self, state: SymbolState) -> float:
        """简单可扩展的评分函数。

        当前逻辑：涨幅越大分数越高。
        后续可加入成交量、波动率、ML 分数等。
        """
        # 基础分 = 涨幅（只取正值）
        score = max(state.price_change_pct, 0.0)

        # 示例扩展位（先留空，后续接 ATR / 成交量）
        # score += volume_factor
        # score += volatility_factor

        return score

    @staticmethod
    def _snapshot(state: SymbolState) -> SymbolState:
        """返回一份浅拷贝，避免外部修改内部状态。"""
        return SymbolState(
            symbol=state.symbol,
            price=state.price,
            prev_price=state.prev_price,
            price_change_pct=state.price_change_pct,
            last_update_ts=state.last_update_ts,
            last_trigger_ts=state.last_trigger_ts,
            status=state.status,
            score=state.score,
            meta=dict(state.meta),
        )

    @staticmethod
    async def _invoke(callback: TriggerCallback, state: SymbolState) -> None:
        try:
            result = callback(state)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception("on_trigger callback error for %s", state.symbol)


# ======================================================================
# 快速自测
# ======================================================================

async def _demo() -> None:
    logging.basicConfig(level=logging.INFO)

    queue: asyncio.Queue[SymbolState] = asyncio.Queue()

    async def on_trigger(state: SymbolState) -> None:
        await queue.put(state)
        print(f"→ 进入分析队列: {state.symbol} score={state.score:.2f}")

    sm = StateManager(
        price_change_threshold=0.3,
        cooldown_seconds=10,
        on_trigger=on_trigger,
    )

    await sm.sync_universe(["BTCUSDT", "ETHUSDT"])

    # 模拟行情
    await sm.on_ticker({"s": "BTCUSDT", "c": "100000"})
    await sm.on_ticker({"s": "BTCUSDT", "c": "100400"})  # +0.4% → 触发
    await sm.on_ticker({"s": "BTCUSDT", "c": "100500"})  # 已 TRIGGERED，忽略
    await sm.mark_analyzing("BTCUSDT")
    await sm.mark_cooldown("BTCUSDT")

    await asyncio.sleep(11)  # 冷却结束
    await sm.on_ticker({"s": "BTCUSDT", "c": "101000"})  # 再涨 → 再次触发


if __name__ == "__main__":
    asyncio.run(_demo())