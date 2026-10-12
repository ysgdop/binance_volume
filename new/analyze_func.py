
import asyncio
from typing import Any, Awaitable, Callable, Optional
import pandas as pd
import httpx
from datetime import datetime, timedelta
import time


async def fetch_klines(symbol: str, interval: str = "4h", days: int = 30, limit: int = 1000) -> pd.DataFrame:
    """单个品种获取K线"""
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
                    resp = await client.get("/fapi/v1/klines", params=params)   # ← fixed endpoint

                    if resp.status_code == 429:
                        retry = int(resp.headers.get("Retry-After", 3))
                        print(f"   ⏳ {symbol} 限流，等待 {retry}秒...")
                        await asyncio.sleep(retry)
                        continue

                    resp.raise_for_status()
                    data = resp.json()

                    if not data:
                        break

                    all_klines.extend(data)
                    current_start = data[-1][6] + 1  # close time
                    break

                except Exception as e:
                    if attempt == 5:
                        print(f"   ❌ {symbol} 最终失败: {e}")
                        return pd.DataFrame()
                    await asyncio.sleep(2 ** attempt * 0.5)

            await asyncio.sleep(0.25)

    if not all_klines:
        print(f"   ⚠️ {symbol} 未获取到任何数据")
        return pd.DataFrame()

    # 数据处理
    df = pd.DataFrame(all_klines, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','quote_volume','count','taker_buy_base',
        'taker_buy_quote','ignore'
    ])

    float_cols = ['open', 'high', 'low', 'close', 'volume', 'count', 'taker_buy_base']
    df[float_cols] = df[float_cols].astype(float)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.set_index('open_time').sort_index()

    print(f"   ✅ {symbol} 完成 | {len(df)} 条记录")
    return df