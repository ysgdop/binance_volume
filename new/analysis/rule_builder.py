

# rank_pro/analysis/rule_builder.py
import pandas as pd
import numpy as np
from typing import Dict


class RuleBuilder:
    """只负责把指标转换成业务规则"""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def build_all(self) -> Dict[str, pd.Series]:
        """返回所有规则（连续值 + 0/1信号）"""
        rules = {}

        # EMA 相关
        ema_comp, ema7_ang, price_brk = self._build_ema_rules()
        rules['ema_compression_ratio'] = ema_comp
        rules['ema7_angle'] = ema7_ang
        rules['price_breakout'] = price_brk

        # 量能
        rules['volume_ratio'] = self._build_volume_rules()
        rules['taker_ratio'] = self._build_taker_rules()
        rules['count_ratio'] = self._build_count_rules()

        # MACD 规则
        rules.update(self._build_macd_rules())

        
        rules['atr_expansion'] = self._build_atr_expansion()
        rules['rolling_high_break'] = self._build_rolling_high_break()
        rules['volume_percentile'] = self._build_volume_percentile()

        return rules

    def _build_ema_rules(self, lookback: int = 3):
        atr_smooth = self.df['atr'] + 1e-8
        ema_cols = ['ema_20', 'ema_30', 'ema_60']

        ema_max = self.df[ema_cols].max(axis=1)
        ema_min = self.df[ema_cols].min(axis=1)
        ema_compression_ratio = (ema_max - ema_min) / atr_smooth

        price_change = self.df['ema_7'] - self.df['ema_7'].shift(lookback)
        standardized_slope = (price_change / atr_smooth) / lookback
        ema7_angle = np.degrees(np.arctan(standardized_slope))

        ema_ref = self.df[['ema_20', 'ema_30']].mean(axis=1)
        price_breakout = (self.df['close'] - ema_ref) / (ema_ref + 1e-8)

        safe = max(60, lookback)
        ema_compression_ratio.iloc[:safe] = 0
        ema7_angle.iloc[:safe] = 0
        price_breakout.iloc[:safe] = 0

        return ema_compression_ratio, ema7_angle, price_breakout

    def _build_volume_rules(self):
        ratio = self.df['volume_ma_5'] / (self.df['volume_ma_20'] + 1e-8)
        ratio.iloc[:20] = 0
        return ratio

    def _build_taker_rules(self):
        ma = self.df['taker_buy_base'].rolling(20, min_periods=1).mean()
        ratio = self.df['taker_buy_base'] / (ma.shift(1) + 1e-8)
        ratio.iloc[:21] = 0
        return ratio

    def _build_count_rules(self):
        ma = self.df['count'].rolling(20, min_periods=1).mean()
        ratio = self.df['count'] / (ma.shift(1) + 1e-8)
        ratio.iloc[:21] = 0
        return ratio

    def _build_macd_rules(self, lookback_window: int = 120, spike_k: float = 2.2):
        df = self.df

        golden_cross = ((df['macd_dif'] > df['macd_dea']) &
                        (df['macd_dif'].shift(1) <= df['macd_dea'].shift(1))).astype(np.int8)

        above_zero = (df['macd_dif'] > 0).astype(np.int8)
        below_zero = (df['macd_dif'] <= 0).astype(np.int8)
        hist_positive = (df['macd_hist'] > 0).astype(np.int8)

        hist_slope = df['macd_hist'].diff()
        acc = hist_slope.diff()
        acc_std = acc.rolling(lookback_window, min_periods=20).std()
        acc_spike = (acc.abs() > spike_k * (acc_std + 1e-8)).astype(np.int8)
        acc_positive = (acc > 0).astype(np.int8)

        trend_up = ((df['macd_dif'] > df['macd_dea']) &
                    (df['macd_dea'] > df['macd_dea'].shift(1))).astype(np.int8)

        macd_bull_attack = (
            (above_zero == 1) & (hist_positive == 1) &
            (acc_positive == 1) & (acc_spike == 1) & (trend_up == 1)
        ).astype(np.int8)

        macd_bottom_ignite = (
            (below_zero == 1) & (golden_cross == 1) & (acc_spike == 1)
        ).astype(np.int8)

        safe = max(lookback_window, 60)
        for s in [golden_cross, macd_bull_attack, macd_bottom_ignite]:
            s.iloc[:safe] = 0

        return {
            "golden_cross": golden_cross,
            "above_zero": above_zero,
            "below_zero": below_zero,
            "hist_positive": hist_positive,
            "acc_spike": acc_spike,
            "acc_positive": acc_positive,
            "trend_up": trend_up,
            "macd_bull_attack": macd_bull_attack,
            "macd_bottom_ignite": macd_bottom_ignite,
        }

    # 在 RuleBuilder 中新增以下方法

    def _build_atr_expansion(self, lookback: int = 20) -> pd.Series:
        """ATR 扩张程度：当前 ATR / 过去 lookback 根 ATR 均值"""
        atr = self.df['atr']
        atr_ma = atr.rolling(lookback, min_periods=5).mean()
        expansion = atr / (atr_ma + 1e-8)
        expansion.iloc[:lookback] = 1.0
        return expansion


    def _build_rolling_high_break(self, window: int = 20) -> pd.Series:
        """是否突破过去 window 根最高价（收盘价）"""
        rolling_high = self.df['close'].rolling(window, min_periods=5).max().shift(1)
        break_ = (self.df['close'] > rolling_high).astype(np.float32)
        break_.iloc[:window] = 0
        return break_


    def _build_volume_percentile(self, window: int = 50) -> pd.Series:
        """当前成交量在过去 window 根中的分位数（0~1）"""
        def percentile_rank(x):
            if len(x) < 5:
                return 0.5
            return (x.rank().iloc[-1] - 1) / (len(x) - 1)

        vol_pct = self.df['volume'].rolling(window, min_periods=10).apply(percentile_rank, raw=False)
        vol_pct.iloc[:window] = 0.5
        return vol_pct