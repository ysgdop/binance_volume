


from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable, Optional, Set, Union

import websockets
from websockets.exceptions import ConnectionClosed

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
TickerCallback = Callable[[dict[str, Any]], Union[None, Awaitable[None]]]
RawMessageCallback = Callable[[Any], Union[None, Awaitable[None]]]

logger = logging.getLogger(__name__)


class BinanceStream:
    """Async Binance Futures market stream client.

    Currently implements the all-market ticker stream (`!ticker@arr`).
    Designed so that additional streams (kline, bookTicker, etc.) can be
    added later without refactoring the core connection logic.

    Args:
        url: WebSocket endpoint.
        symbols: Optional set/list of symbols to keep. ``None`` means all.
        on_ticker: Callback invoked for every filtered ticker item.
            Can be a normal function or a coroutine function.
        on_raw_message: Optional callback for the raw parsed message
            (list or dict) before symbol filtering.
        ping_interval: Seconds between application-level pings.
        ping_timeout: Seconds to wait for pong before considering
            the connection dead.
        max_reconnect_delay: Upper bound for exponential backoff (seconds).
        name: Optional name used in log messages (useful when running
            multiple stream instances).
    """

    DEFAULT_URL = "wss://fstream.binance.com/market/ws/!ticker@arr"

    def __init__(
        self,
        url: str = DEFAULT_URL,
        symbols: Optional[list[str] | set[str]] = None,
        on_ticker: Optional[TickerCallback] = None,
        on_raw_message: Optional[RawMessageCallback] = None,
        ping_interval: float = 180.0,
        ping_timeout: float = 60.0,
        max_reconnect_delay: float = 60.0,
        name: str = "BinanceStream",
    ) -> None:
        self.url = url
        self.name = name

        self._symbols: Optional[Set[str]] = set(symbols) if symbols else None
        self.on_ticker = on_ticker
        self.on_raw_message = on_raw_message

        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.max_reconnect_delay = max_reconnect_delay

        # Runtime state
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._connected = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None

        # Latest ticker cache: symbol -> ticker dict
        self.latest: dict[str, dict[str, Any]] = {}

        # Reconnect bookkeeping
        self._reconnect_attempts = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the stream in the background.

        Safe to call multiple times; subsequent calls are no-ops while
        the stream is already running.
        """
        if self._running:
            logger.warning("[%s] Already running", self.name)
            return

        self._running = True
        self._task = asyncio.create_task(self._run_forever(), name=f"{self.name}-main")
        logger.info("[%s] Started", self.name)

    async def stop(self) -> None:
        """Gracefully stop the stream and cancel background tasks."""
        if not self._running:
            return

        logger.info("[%s] Stopping...", self.name)
        self._running = False
        self._connected.clear()

        await self._cancel_task(self._ping_task)
        self._ping_task = None

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        await self._cancel_task(self._task)
        self._task = None
        logger.info("[%s] Stopped", self.name)

    async def wait_connected(self, timeout: Optional[float] = None) -> bool:
        """Wait until the WebSocket is connected.

        Args:
            timeout: Maximum seconds to wait. ``None`` means wait forever.

        Returns:
            True if connected within the timeout, False otherwise.
        """
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def update_symbols(self, symbols: Optional[list[str] | set[str]]) -> None:
        """Dynamically change the symbol filter.

        Args:
            symbols: New symbol set. ``None`` means receive all symbols.
        """
        self._symbols = set(symbols) if symbols else None
        logger.info(
            "[%s] Symbol filter updated → %s",
            self.name,
            "ALL" if self._symbols is None else sorted(self._symbols),
        )

    def get_price(self, symbol: str) -> Optional[float]:
        """Return the latest price for ``symbol``, or ``None``."""
        item = self.latest.get(symbol)
        if item is None:
            return None
        try:
            return float(item["c"])
        except (KeyError, TypeError, ValueError):
            return None

    def get_ticker(self, symbol: str) -> Optional[dict[str, Any]]:
        """Return the latest full ticker dict for ``symbol``, or ``None``."""
        return self.latest.get(symbol)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    # ------------------------------------------------------------------
    # Internal – connection loop
    # ------------------------------------------------------------------
    # 重连
    async def _run_forever(self) -> None:
        """Main loop: connect → consume → reconnect on failure."""
        while self._running:
            try:
                await self._connect_and_consume()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("[%s] Unexpected error: %s", self.name, exc)

            if not self._running:
                break

            delay = self._next_backoff()
            logger.warning(
                "[%s] Reconnecting in %.1fs (attempt %d)",
                self.name,
                delay,
                self._reconnect_attempts,
            )
            await asyncio.sleep(delay)

    async def _connect_and_consume(self) -> None:
        """Establish one connection and read messages until it drops."""
        logger.info("[%s] Connecting to %s", self.name, self.url)

        async with websockets.connect(
            self.url,
            ping_interval=None,          # we handle ping ourselves
            ping_timeout=None,
            close_timeout=5,
            max_size=10 * 1024 * 1024,   # 10 MB – !ticker@arr can be large
        ) as ws:
            self._ws = ws
            self._connected.set()
            self._reconnect_attempts = 0
            logger.info("[%s] Connected", self.name)

            self._ping_task = asyncio.create_task(
                self._heartbeat_loop(), name=f"{self.name}-ping"
            )

            try:
                async for raw in ws:
                    if not self._running:
                        break
                    await self._handle_raw_message(raw)
            finally:
                self._connected.clear()
                await self._cancel_task(self._ping_task)
                self._ping_task = None
                self._ws = None
                logger.info("[%s] Connection closed", self.name)

    # ------------------------------------------------------------------
    # Internal – message handling
    # ------------------------------------------------------------------

    async def _handle_raw_message(self, raw: str | bytes) -> None:
        """Parse one WebSocket message and dispatch callbacks."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("[%s] JSON decode error: %s", self.name, exc)
            return

        # Optional raw callback (before filtering)
        if self.on_raw_message is not None:
            await self._invoke_callback(self.on_raw_message, data)

        # !ticker@arr → list of tickers
        if isinstance(data, list):
            for item in data:
                await self._process_ticker_item(item)
        elif isinstance(data, dict):
            # Future-proof: single ticker / other stream types
            await self._process_ticker_item(data)

    async def _process_ticker_item(self, item: dict[str, Any]) -> None:
        """Filter, cache and dispatch a single ticker object."""
        symbol = item.get("s")
        if not symbol:
            return

        if self._symbols is not None and symbol not in self._symbols:
            return

        self.latest[symbol] = item

        if self.on_ticker is not None:
            await self._invoke_callback(self.on_ticker, item)

    # ------------------------------------------------------------------
    # Internal – heartbeat
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Send application-level pings and detect silent disconnects."""
        while self._running and self._ws is not None:
            try:
                await asyncio.sleep(self.ping_interval)
                if self._ws is None:
                    break

                pong_waiter = await self._ws.ping()
                await asyncio.wait_for(pong_waiter, timeout=self.ping_timeout)
                logger.debug("[%s] Pong received", self.name)

            except asyncio.TimeoutError:
                logger.warning("[%s] Ping timeout – closing connection", self.name)
                if self._ws is not None:
                    await self._ws.close()
                break
            except ConnectionClosed:
                break
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("[%s] Heartbeat error: %s", self.name, exc)
                break

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _next_backoff(self) -> float:
        """Exponential backoff with upper bound."""
        self._reconnect_attempts += 1
        delay = min(2 ** self._reconnect_attempts, self.max_reconnect_delay)
        return delay

    @staticmethod
    async def _invoke_callback(callback: Callable, *args: Any) -> None:
        """Invoke a sync or async callback uniformly."""
        try:
            result = callback(*args)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            logger.exception("Callback error: %s", exc)

    @staticmethod
    async def _cancel_task(task: Optional[asyncio.Task]) -> None:
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass










# ======================================================================
# Example usage
# ======================================================================

async def _demo() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    def sync_handler(item: dict) -> None:
        print(f"[sync] {item['s']} = {item['c']}")

    async def async_handler(item: dict) -> None:
        # 模拟异步处理（写库 / 发通知等）
        await asyncio.sleep(0)
        print(f"[async] {item['s']} = {item['c']}")

    stream = BinanceStream(
        symbols=["BTCUSDT", "ETHUSDT"],
        on_ticker=async_handler,          # 也可传 sync_handler
        name="TickerStream",
    )

    stream.start()
    connected = await stream.wait_connected(timeout=10)
    print("Connected:", connected)

    # 运行 30 秒后动态改过滤列表
    await asyncio.sleep(30)
    stream.update_symbols(["BTCUSDT", "SOLUSDT"])

    await asyncio.sleep(30)
    print("BTC price:", stream.get_price("BTCUSDT"))

    await stream.stop()


if __name__ == "__main__":
    asyncio.run(_demo())