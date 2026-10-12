

import asyncio
import httpx


class BinanceScanner:

    def __init__(
        self,
        price_change = 0.5,
        min_24h_volume=10_000_000,   # 2000万U
    ):

        self.url = (
            "https://fapi.binance.com/fapi/v1/ticker/24hr"
        )

        self.min_24h_volume = min_24h_volume
        self.price_change = price_change
        self.old_price: dict[str, float] = {}
        # http client
        self.client = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            timeout=10,
            http2=True,
            verify=False,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.aclose()

    async def fetch_ticker_24h(self):
        
        try:

            resp = await self.client.get(self.url)

            if resp.status_code != 200:
                print("request failed:", resp.text)
                return []

            data = resp.json()

            return data

        except Exception as e:

            print("fetch error:", e)

            return []
        
    async def filter_symbols(self, data):

        result = []

        for item in data:

            try:
                symbol = item["symbol"]
                price = float(item["lastPrice"])

                # 只要 USDT 永续
                if not symbol.endswith("USDT"):
                    continue

                # 排除季度合约
                if "_" in symbol:
                    continue

                # 流动性过滤
                quote_volume = float(item["quoteVolume"])

                if quote_volume < self.min_24h_volume:
                    continue

                # 获取上一次价格
                old_price = self.old_price.get(symbol)

                # 第一次出现
                if old_price is None:
                    self.old_price[symbol] = price
                    result.append(item)
                    continue

                # 计算价格变化百分比
                price_change_pct = (
                    (price - old_price)
                    / old_price
                    * 100
                )
                
                # 价格上涨不足阈值  不用绝对值
                if price_change_pct <= self.price_change:
                    continue
                self.old_price[symbol] = price
                result.append(item)

            except Exception as e:
                print(f"filter error: {e}")

        

        return result

    async def _run(self):
        date = await self.fetch_ticker_24h()
        return await self.filter_symbols(date)
