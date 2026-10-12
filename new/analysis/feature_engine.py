

# rank_pro/analysis/feature_engine.py
import pandas as pd
import numpy as np


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period, min_periods=1).mean()
    return atr.bfill().fillna(1e-8)


class FeatureEngine:
    """只负责生成技术指标，不产生任何业务规则"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare()

    def _prepare(self):
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'count', 'taker_buy_base']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        self.df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def build(self) -> pd.DataFrame:
        """一键生成所有基础指标"""
        self.calc_ema([7, 20, 30, 60, 120, 250])
        self.calc_volume([5, 20])
        self.calc_macd()
        self.df['atr'] = calculate_atr(self.df, 14)
        return self.df

    def calc_ema(self, periods):
        for p in periods:
            self.df[f'ema_{p}'] = self.df['close'].ewm(span=p, adjust=False).mean()

    def calc_volume(self, periods):
        for p in periods:
            self.df[f'volume_ma_{p}'] = self.df['volume'].rolling(p, min_periods=1).mean()

    def calc_macd(self):
        ema12 = self.df['close'].ewm(span=12, adjust=False).mean()
        ema26 = self.df['close'].ewm(span=26, adjust=False).mean()
        self.df['macd_dif'] = ema12 - ema26
        self.df['macd_dea'] = self.df['macd_dif'].ewm(span=9, adjust=False).mean()
        self.df['macd_hist'] = (self.df['macd_dif'] - self.df['macd_dea']) * 2