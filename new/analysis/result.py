

# rank_pro/analysis/result.py
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class AnalyzeResult:
    symbol: str
    signal: int                     # 0 or 1
    score: float
    close: float
    status: str = "ok"              # ok / insufficient_data / error

    # 关键特征（方便下游使用）
    # analysis/result.py 增加字
    
    # 四维分数
    trend_score: float = 0.0
    money_score: float = 0.0
    momentum_score: float = 0.0
    breakout_score: float = 0.0
    
    # 新增特征
    atr_expansion: float = 1.0
    rolling_high_break: float = 0.0
    volume_percentile: float = 0.5
    ema7_angle: float = 0.0
    ema_compression_ratio: float = 0.0
    volume_ratio: float = 0.0
    taker_ratio: float = 0.0
    count_ratio: float = 0.0
    price_breakout: float = 0.0

    macd_bull_attack: int = 0
    macd_bottom_ignite: int = 0
    above_zero: int = 0
    trend_up: int = 0

    bars: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_signal(self) -> bool:
        return self.signal == 1 and self.status == "ok"