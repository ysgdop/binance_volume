
# signal_service.py
"""
信号下游处理中心（单文件版）
职责：去重 → 策略过滤 → 写入数据库（DuckDB）
"""

import time
import asyncio
import logging
from typing import Dict, Optional
from pathlib import Path

import duckdb
from analysis.result import AnalyzeResult

logger = logging.getLogger(__name__)


# ====================== 1. 去重器 ======================
class SignalDeduplicator:
    """同一币种在冷却时间内只允许通过一次"""

    def __init__(self, cooldown_seconds: int = 4 * 3600):
        self.cooldown = cooldown_seconds
        self._last_seen: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def is_duplicate(self, symbol: str) -> bool:
        async with self._lock:
            now = time.time()
            last = self._last_seen.get(symbol, 0)
            return (now - last) < self.cooldown

    async def mark(self, symbol: str) -> None:
        async with self._lock:
            self._last_seen[symbol] = time.time()

    async def clear(self, symbol: Optional[str] = None) -> None:
        async with self._lock:
            if symbol:
                self._last_seen.pop(symbol, None)
            else:
                self._last_seen.clear()


# ====================== 2. 策略过滤器 ======================
class SignalStrategy:
    """基于四维评分的策略过滤器"""

    def __init__(
        self,
        min_total_score: float = 70,
        min_trend_score: float = 12,
        min_money_score: float = 10,
        min_momentum_score: float = 8,
        require_breakout: bool = False,
    ):
        self.min_total_score = min_total_score
        self.min_trend_score = min_trend_score
        self.min_money_score = min_money_score
        self.min_momentum_score = min_momentum_score
        self.require_breakout = require_breakout

    def should_take(self, result: AnalyzeResult) -> bool:
        if result.score < self.min_total_score:
            return False
        if result.trend_score < self.min_trend_score:
            return False
        if result.money_score < self.min_money_score:
            return False
        if result.momentum_score < self.min_momentum_score:
            return False
        if self.require_breakout and result.breakout_score < 5:
            return False
        return True


# ====================== 3. 数据库写入（DuckDB） ======================
class SignalDB:
    """信号写入 DuckDB"""

    def __init__(self, db_path: Optional[str] = None, con: Optional[duckdb.DuckDBPyConnection] = None):
        """
        两种使用方式：
        1. 传 db_path（推荐）→ 自动创建/连接文件
        2. 传已有的 con（可复用 DBManager 的连接）
        
        如果两者都为 None，则 save 会直接跳过
        """
        self._own_connection = False

        if con is not None:
            self.con = con
        elif db_path is not None:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self.con = duckdb.connect(db_path)
            self._own_connection = True
        else:
            self.con = None

        if self.con is not None:
            self._init_table()

    def _init_table(self):
        """创建 signals 表（如果不存在）"""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                symbol          VARCHAR PRIMARY KEY,
                signal          VARCHAR,
                score           DOUBLE,
                close           DOUBLE,
                ema7_angle      DOUBLE,
                ema_compression_ratio DOUBLE,
                volume_ratio    DOUBLE,
                taker_ratio     DOUBLE,
                count_ratio     DOUBLE,
                price_breakout  BOOLEAN,
                macd_bull_attack BOOLEAN,
                macd_bottom_ignite BOOLEAN,
                above_zero      BOOLEAN,
                trend_up        BOOLEAN,
                bars            INTEGER,
                created_at      TIMESTAMP DEFAULT current_timestamp
            )
        """)
        # 兼容没有 sequence 的情况（DuckDB 新版本支持更好）
        try:
            self.con.execute("CREATE SEQUENCE IF NOT EXISTS signals_id_seq")
        except Exception:
            pass

        logger.info("✅ signals 表已就绪（DuckDB）")

    def _save_sync(self, result: AnalyzeResult) -> None:
        """同步写入（在线程中执行）"""
        self.con.execute(
            """
            INSERT INTO signals (
                symbol,
                signal,
                score,
                close,
                ema7_angle,
                ema_compression_ratio,
                volume_ratio,
                taker_ratio,
                count_ratio,
                price_breakout,
                macd_bull_attack,
                macd_bottom_ignite,
                above_zero,
                trend_up,
                bars,
                created_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, current_timestamp
            )
            """,
            [
                result.symbol,
                result.signal,
                result.score,
                result.close,
                result.ema7_angle,
                result.ema_compression_ratio,
                result.volume_ratio,
                result.taker_ratio,
                result.count_ratio,
                result.price_breakout,
                result.macd_bull_attack,
                result.macd_bottom_ignite,
                result.above_zero,
                result.trend_up,
                result.bars,
            ],
        )

    async def save(self, result: AnalyzeResult) -> None:
        if self.con is None:
            logger.debug("DuckDB 未初始化，跳过写入 | %s", result.symbol)
            return

        try:
            # 放到线程池，避免阻塞事件循环
            await asyncio.to_thread(self._save_sync, result)
            logger.debug("DB 写入成功 | %s | score=%.1f", result.symbol, result.score)

        except Exception as e:
            logger.exception("DB 写入失败 | %s | %s", result.symbol, e)

    def close(self):
        if self._own_connection and self.con is not None:
            self.con.close()
            self.con = None


# ====================== 4. 总服务入口 ======================
class SignalService:
    """
    信号下游处理中心
    流程：去重 → 策略 → 写数据库
    """

    def __init__(
        self,
        db: Optional[SignalDB] = None,
        deduplicator: Optional[SignalDeduplicator] = None,
        strategy: Optional[SignalStrategy] = None,
    ):
        self.db = db or SignalDB(db_path=None)  # 默认不写库
        self.deduplicator = deduplicator or SignalDeduplicator(cooldown_seconds=4 * 3600)
        self.strategy = strategy or SignalStrategy()

    async def handle(self, state, result: AnalyzeResult) -> None:
        """Worker 的 on_result 直接指向这里"""
        try:
            # 1. 基础检查
            if result.status != "ok" or not result.is_signal:
                return

            # 2. 去重
            if await self.deduplicator.is_duplicate(result.symbol):
                logger.info("⏭️  去重跳过 | %s", result.symbol)
                return

            # 3. 策略过滤
            if not self.strategy.should_take(result):
                logger.info(
                    "🚫 策略过滤 | %s | score=%.1f | angle=%.1f | vol=%.2f",
                    result.symbol, result.score, result.ema7_angle, result.volume_ratio
                )
                return

            # 4. 写入数据库
            await self.db.save(result)

            # 5. 标记已处理
            await self.deduplicator.mark(result.symbol)

            logger.info(
                "✅ 信号已入库 | %s | Score=%.1f | Close=%.6f | Angle=%.1f | Vol=%.2f",
                result.symbol,
                result.score,
                result.close,
                result.ema7_angle,
                result.volume_ratio,
            )

        except Exception as e:
            logger.exception("SignalService 处理失败 | %s | %s", result.symbol, e)