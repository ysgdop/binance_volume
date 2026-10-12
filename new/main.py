





# main.py
import asyncio
import logging
from typing import Set

from BinanceScanner import BinanceScanner
from BinanceStream import BinanceStream
from StateManager import StateManager, SymbolState
from TimerManager import TimerManager
from AnalysisWorker import AnalysisQueue, AnalysisWorkerPool

# 分析层
from analysis.analyzer import Analyzer
from analysis.result import AnalyzeResult

# 信号层
from signal_server import SignalDB,SignalDeduplicator,SignalService,SignalStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


# ====================== 全局实例 ======================
analyzer = Analyzer(score_threshold=70, cache_ttl=15 * 60)


async def real_analyze(state: SymbolState) -> AnalyzeResult:
    """Worker 真正调用的分析函数"""
    return await analyzer.analyze(state.symbol)


async def run_once(scanner: BinanceScanner) -> list:
    """执行一次 REST 扫描"""
    return await scanner._run()

from utils.config import DB_FILE

async def main():
    

    signal_service = SignalService(
        db=SignalDB(DB_FILE),          # 没有数据库就传 None
        deduplicator=SignalDeduplicator(cooldown_seconds=4 * 3600),
        strategy=SignalStrategy(
            min_total_score=70,             # 总分门槛  scorer.py
            min_trend_score=12,             # 趋势维度最低分
            min_money_score=0,             # 资金维度最低分
            min_momentum_score=8,           # 动能维度最低分
            require_breakout=False,         # 是否强制要求突破分
        ),
    )

    async with BinanceScanner() as scanner:
        timer = TimerManager()

        # 1. Queue + StateManager
        analysis_queue = AnalysisQueue(maxsize=200, dedup=True)

        state_manager = StateManager(
            price_change_threshold=0.5,
            cooldown_seconds=300,
            on_trigger=analysis_queue.put,
        )

        # 2. Worker Pool（已正确接入 SignalService）
        worker_pool = AnalysisWorkerPool(
            queue=analysis_queue,
            state_manager=state_manager,
            analyze_fn=real_analyze,
            on_result=signal_service.handle,      # ← 正确入口
            num_workers=2,
        )
        await worker_pool.start()

        # 3. 第一次 REST 扫描，同步 universe (写入数据)
        result = await run_once(scanner)  
        symbols: Set[str] = {item["symbol"] for item in result}
        await state_manager.sync_universe(symbols)
        logger.info("初始 universe 大小: %d", len(symbols))

        # 4. WebSocket 实时行情
        stream = BinanceStream(
            symbols=symbols,
            on_ticker=state_manager.on_ticker,
            name="TickerStream",
        )
        await stream.start()
        await stream.wait_connected(timeout=10)
        logger.info("WebSocket 已连接")

        # 5. 定时刷新 Universe（每 4 小时）
        async def scheduled_job():
            try:
                result = await run_once(scanner)
                new_symbols = {item["symbol"] for item in result}
                stream.update_symbols(new_symbols)
                await state_manager.sync_universe(new_symbols)
                logger.info("Universe 已刷新，当前数量: %d", len(new_symbols))
            except Exception as e:
                logger.exception("定时刷新 Universe 失败: %s", e)

        # 定时进行
        timer.add_cron_job(
            scheduled_job,
            hour="0,4,8,12,16,20",
            minute=0,
            second=0,
            job_id="rest_polling",
        )
        timer.start()

        # 6. 主循环
        try:
            logger.info("系统已启动，等待信号...")
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            logger.info("收到取消信号，开始优雅退出...")
        finally:
            await stream.stop()
            await worker_pool.stop()
            # 如果有数据库连接池，在这里关闭
            # await db_pool.close()
            logger.info("系统已安全退出")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("用户手动中断")

