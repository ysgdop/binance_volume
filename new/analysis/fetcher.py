

# rank_pro/analysis/fetcher.py
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Optional
import pandas as pd
import httpx
import logging

logger = logging.getLogger(__name__)


class KlineFetcher:
    """
    带缓存 + 按 symbol 细粒度锁的 K线获取器
    - 不同币种互不阻塞
    - 同一币种的并发请求会合并
    """

    def __init__(self, cache_ttl_seconds: int = 15 * 60):
        self.cache: Dict[str, tuple[pd.DataFrame, float]] = {}
        self.cache_ttl = cache_ttl_seconds
        
        # 细粒度锁：每个 symbol 一把独立的锁
        self._locks: Dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()          # 仅用于保护 _locks 字典本身

    async def _get_lock(self, symbol: str) -> asyncio.Lock:
        """安全获取（或创建）某个 symbol 的专属锁"""
        async with self._locks_guard:
            if symbol not in self._locks:
                self._locks[symbol] = asyncio.Lock()
            return self._locks[symbol]

    async def fetch(
        self,
        symbol: str,
        interval: str = "4h",
        days: int = 40,
        limit: int = 1000,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        now = time.time()

        # 1. 快速路径：缓存命中且未过期（无锁）
        if not force_refresh and symbol in self.cache:
            df, ts = self.cache[symbol]
            if now - ts < self.cache_ttl:
                return df.copy()

        # 2. 获取该 symbol 的专属锁
        lock = await self._get_lock(symbol)
        
        async with lock:
            # 双重检查（防止等待锁期间已经被其他协程更新）
            if not force_refresh and symbol in self.cache:
                df, ts = self.cache[symbol]
                if now - ts < self.cache_ttl:
                    return df.copy()

            # 真正请求
            df = await self._fetch_from_binance(symbol, interval, days, limit)
            if not df.empty:
                self.cache[symbol] = (df, now)
            return df

    async def _fetch_from_binance(
        self,
        symbol: str,
        interval: str,
        days: int,
        limit: int
    ) -> pd.DataFrame:
        print(f"   📥 正在拉取 {symbol} ...")
        end_time = int(time.time() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        all_klines = []
        current_start = start_time

        async with httpx.AsyncClient(
            base_url="https://fapi.binance.com",
            timeout=httpx.Timeout(20.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=5),
        ) as client:
            while current_start < end_time:
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "limit": limit,
                    "startTime": current_start,
                }

                for attempt in range(6):
                    try:
                        resp = await client.get("/fapi/v1/klines", params=params)

                        if resp.status_code == 429:
                            retry = int(resp.headers.get("Retry-After", 3))
                            await asyncio.sleep(retry)
                            continue

                        resp.raise_for_status()
                        data = resp.json()
                        if not data:
                            break

                        all_klines.extend(data)
                        current_start = data[-1][6] + 1
                        break

                    except Exception as e:
                        if attempt == 5:
                            logger.error(f"{symbol} 最终失败: {e}")
                            return pd.DataFrame()
                        await asyncio.sleep(2 ** attempt * 0.5)

                await asyncio.sleep(0.2)

        if not all_klines:
            return pd.DataFrame()

        df = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'count', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        float_cols = ['open', 'high', 'low', 'close', 'volume', 'count', 'taker_buy_base']
        df[float_cols] = df[float_cols].astype(float)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.set_index('open_time').sort_index()

        print(f"   ✅ {symbol} 完成 | {len(df)} 条记录")
        return df

    def clear_cache(self, symbol: Optional[str] = None):
        if symbol:
            self.cache.pop(symbol, None)
        else:
            self.cache.clear()