

# rank_pro/analysis/analyzer.py
import logging
from typing import Optional
import pandas as pd

from .result import AnalyzeResult
from .fetcher import KlineFetcher
from .feature_engine import FeatureEngine
from .rule_builder import RuleBuilder
from .scorer import SignalScorer

logger = logging.getLogger(__name__)


# rank_pro/analysis/feature_cache.py
import time
from typing import Dict, Optional, Tuple

import asyncio


class FeatureCache:
    """
    缓存 FeatureEngine.build() 的结果
    - 按 symbol 细粒度锁
    - 默认与 K线缓存相同的 TTL（15分钟）
    """

    def __init__(self, cache_ttl_seconds: int = 15 * 60):
        self.cache: Dict[str, Tuple[pd.DataFrame, float]] = {}
        self.cache_ttl = cache_ttl_seconds
        self._locks: Dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def _get_lock(self, symbol: str) -> asyncio.Lock:
        async with self._locks_guard:
            if symbol not in self._locks:
                self._locks[symbol] = asyncio.Lock()
            return self._locks[symbol]

    async def get_or_build(
        self,
        symbol: str,
        raw_df: pd.DataFrame,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        获取缓存的特征，或重新计算后写入缓存
        """
        now = time.time()

        # 快速路径
        if not force_refresh and symbol in self.cache:
            feature_df, ts = self.cache[symbol]
            if now - ts < self.cache_ttl:
                return feature_df.copy()

        lock = await self._get_lock(symbol)
        async with lock:
            # 双重检查
            if not force_refresh and symbol in self.cache:
                feature_df, ts = self.cache[symbol]
                if now - ts < self.cache_ttl:
                    return feature_df.copy()

            # 真正计算
            from .feature_engine import FeatureEngine
            feature_df = FeatureEngine(raw_df).build()
            
            self.cache[symbol] = (feature_df, now)
            return feature_df.copy()

    def clear(self, symbol: Optional[str] = None):
        if symbol:
            self.cache.pop(symbol, None)
        else:
            self.cache.clear()

    def invalidate(self, symbol: str):
        """主动失效某个币种的特征缓存（当K线更新时调用）"""
        self.cache.pop(symbol, None)




class Analyzer:
    """
    实时分析总入口（带双层缓存）
    - K线缓存（网络）
    - 特征缓存（CPU）
    """

    def __init__(
        self,
        score_threshold: float = 70,
        cache_ttl: int = 15 * 60          # 统一 TTL
    ):
        self.fetcher = KlineFetcher(cache_ttl_seconds=cache_ttl)
        self.feature_cache = FeatureCache(cache_ttl_seconds=cache_ttl)
        self.scorer = SignalScorer(score_threshold=score_threshold)


    async def analyze(self, symbol: str, force_refresh: bool = False) -> AnalyzeResult:
        try:
            # 1. 获取原始 K线
            raw_df = await self.fetcher.fetch(
                symbol,
                interval="4h",
                days=40,
                force_refresh=force_refresh
            )

            if raw_df.empty or len(raw_df) < 100:
                return AnalyzeResult(
                    symbol=symbol,
                    signal=0,
                    score=0.0,
                    close=0.0,
                    status="insufficient_data",
                    bars=len(raw_df)
                )

            # 2. 获取特征
            feature_df = await self.feature_cache.get_or_build(
                symbol=symbol,
                raw_df=raw_df,
                force_refresh=force_refresh
            )

            # 3. 业务规则
            rules = RuleBuilder(feature_df).build_all()

            # 4. 评分
            logger.info("%s scoring...", symbol)
            scored = self.scorer.score(feature_df, rules)

            # 5. 只取最新一根
            latest = scored.iloc[-1]

            logger.info(
                "%s score=%.1f signal=%d",
                symbol,
                latest["total_score"],
                latest["signal"],
            )

            return AnalyzeResult(
                symbol=symbol,
                signal=int(latest["signal"]),
                score=float(latest["total_score"]),
                close=float(latest["close"]),
                status="ok",

                # 原有特征
                ema7_angle=float(latest.get("ema7_angle", 0)),
                volume_ratio=float(latest.get("volume_ratio", 0)),
                taker_ratio=float(latest.get("taker_ratio", 0)),
                count_ratio=float(latest.get("count_ratio", 0)),
                price_breakout=float(latest.get("price_breakout", 0)),
                macd_bull_attack=int(latest.get("macd_bull_attack", 0)),
                macd_bottom_ignite=int(latest.get("macd_bottom_ignite", 0)),
                trend_up=int(latest.get("trend_up", 0)),

                # 四维分数
                trend_score=float(latest.get("trend_score", 0)),
                money_score=float(latest.get("money_score", 0)),
                momentum_score=float(latest.get("momentum_score", 0)),
                breakout_score=float(latest.get("breakout_score", 0)),

                # 新增特征
                atr_expansion=float(latest.get("atr_expansion", 1.0)),
                rolling_high_break=float(latest.get("rolling_high_break", 0)),
                volume_percentile=float(latest.get("volume_percentile", 0.5)),

                bars=len(scored)
            )

        except Exception as e:
            logger.exception("分析 %s 失败", symbol)
            return AnalyzeResult(
                symbol=symbol,
                signal=0,
                score=0.0,
                close=0.0,
                status="error",
                error=str(e)
            )
    
    
    def clear_cache(self, symbol: Optional[str] = None):
        """手动清理缓存（可用于定时任务或测试）"""
        self.fetcher.clear_cache(symbol)
        self.feature_cache.clear(symbol)