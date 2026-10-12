

# analysis/scorer.py
import pandas as pd
import numpy as np
from typing import Dict


class SignalScorer:
    """
    四维加权评分（总分 100）
    Trend 30 + Money 30 + Momentum 25 + Breakout 15
    """

    def __init__(self, score_threshold: float = 70):
        self.score_threshold = score_threshold

    def score(self, df: pd.DataFrame, rules: Dict[str, pd.Series]) -> pd.DataFrame:
        result = df.copy()
        for k, v in rules.items():
            result[k] = v

        # ==================== 1. 趋势维度 (30分) ====================
        trend_score = pd.Series(0.0, index=result.index)

        # macd_bull_attack 最高权重
        trend_score += result['macd_bull_attack'] * 15
        # 底部点火
        trend_score += result['macd_bottom_ignite'] * 12
        # 趋势向上
        trend_score += result['trend_up'] * 8
        # 柱子为正（辅助）
        trend_score += result.get('hist_positive', 0) * 5

        trend_score = trend_score.clip(0, 30)
        
        # ==================== 2. 资金维度 (30分) ====================
        money_score = pd.Series(0.0, index=result.index)

        # 量比
        money_score += np.where(result['volume_ratio'] > 3.0, 12,
                         np.where(result['volume_ratio'] > 2.2, 8,
                         np.where(result['volume_ratio'] > 1.8, 4, 0)))

        # 主动买入
        money_score += np.where(result['taker_ratio'] > 2.5, 10,
                         np.where(result['taker_ratio'] > 1.8, 6, 0))

        # 笔数
        money_score += np.where(result['count_ratio'] > 3.0, 8,
                         np.where(result['count_ratio'] > 2.2, 5, 0))

        money_score = money_score.clip(0, 30)

        # ==================== 3. 动能维度 (25分) ====================
        momentum_score = pd.Series(0.0, index=result.index)

        momentum_score += result.get('acc_spike', 0) * 10
        momentum_score += result.get('hist_positive', 0) * 5

        # ema7 角度
        momentum_score += np.where(result['ema7_angle'] > 45, 10,
                            np.where(result['ema7_angle'] > 30, 6,
                            np.where(result['ema7_angle'] > 15, 3, 0)))

        momentum_score = momentum_score.clip(0, 25)

        # ==================== 4. 突破维度 (15分) ====================
        breakout_score = pd.Series(0.0, index=result.index)

        # 价格相对均线突破
        breakout_score += np.where(result['price_breakout'] > 0.05, 6,
                            np.where(result['price_breakout'] > 0.02, 3, 0))

        # 滚动新高突破
        breakout_score += result.get('rolling_high_break', 0) * 5

        # ATR 扩张
        breakout_score += np.where(result.get('atr_expansion', 1) > 1.5, 4,
                            np.where(result.get('atr_expansion', 1) > 1.2, 2, 0))

        breakout_score = breakout_score.clip(0, 15)

        # ==================== 总分 ====================
        total_score = trend_score + money_score + momentum_score + breakout_score

        result['trend_score'] = trend_score
        result['money_score'] = money_score
        result['momentum_score'] = momentum_score
        result['breakout_score'] = breakout_score
        result['total_score'] = total_score

        # 最终信号
        result['signal'] = (total_score >= self.score_threshold).astype(np.int8)
      
        return result