

# 技术指标  open, high, low, close, volume, count, taker_buy_base
# 零轴上方不是必要条件

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period, min_periods=1).mean()
    return atr.bfill().fillna(1e-8)


class FeatureEngine:
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


class FeatureRuleBuilder(FeatureEngine):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

    def run_base(self):
        self.calc_ema([7, 20, 30, 60, 120, 250])
        self.calc_volume([5, 20])
        self.calc_macd()

    def build_ema_rules(self, lookback: int = 3):
        if 'atr' not in self.df.columns:
            self.df['atr'] = calculate_atr(self.df, period=14)

        ema_cols = ['ema_20', 'ema_30', 'ema_60']
        for col in ema_cols + ['ema_7']:
            if col not in self.df.columns:
                raise ValueError(f"缺少列: {col}，请先执行 run_base()")

        atr_smooth = self.df['atr'] + 1e-8

        # 均线压缩
        ema_max = self.df[ema_cols].max(axis=1)
        ema_min = self.df[ema_cols].min(axis=1)
        ema_compression_ratio = (ema_max - ema_min) / atr_smooth

        # EMA7 角度（ATR标准化）
        price_change = self.df['ema_7'] - self.df['ema_7'].shift(lookback)
        standardized_slope = (price_change / atr_smooth) / lookback
        ema7_angle = np.degrees(np.arctan(standardized_slope))

        # 价格相对均线突破
        ema_ref = self.df[['ema_20', 'ema_30']].mean(axis=1)
        price_breakout = (self.df['close'] - ema_ref) / (ema_ref + 1e-8)

        # 前期不稳定区域置零
        safe = max(60, lookback)
        ema_compression_ratio.iloc[:safe] = 0
        ema7_angle.iloc[:safe] = 0
        price_breakout.iloc[:safe] = 0

        return ema_compression_ratio, ema7_angle, price_breakout

    def build_volume_rules(self):
        for col in ['volume_ma_5', 'volume_ma_20']:
            if col not in self.df.columns:
                raise ValueError(f"缺少列: {col}")

        volume_ratio = self.df['volume_ma_5'] / (self.df['volume_ma_20'] + 1e-8)
        volume_ratio.iloc[:20] = 0
        return volume_ratio

    def build_macd_rules(self, lookback_window: int = 100, spike_k: float = 2.0, fill_window: int = 60):
        df = self.df
        required = ['macd_dif', 'macd_dea', 'macd_hist']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"缺少MACD列: {col}")

        # 基础状态
        golden_cross = ((df['macd_dif'] > df['macd_dea']) &
                        (df['macd_dif'].shift(1) <= df['macd_dea'].shift(1))).astype(np.int8)

        dead_cross = ((df['macd_dif'] < df['macd_dea']) &
                      (df['macd_dif'].shift(1) >= df['macd_dea'].shift(1))).astype(np.int8)

        above_zero = (df['macd_dif'] > 0).astype(np.int8)
        below_zero = (df['macd_dif'] <= 0).astype(np.int8)
        hist_positive = (df['macd_hist'] > 0).astype(np.int8)

        # 动能
        hist_slope = df['macd_hist'].diff()
        acc = hist_slope.diff()
        acc_std = acc.rolling(lookback_window, min_periods=20).std()
        acc_spike = (acc.abs() > spike_k * (acc_std + 1e-8)).astype(np.int8)
        acc_positive = (acc > 0).astype(np.int8)

        # 趋势
        trend_up = ((df['macd_dif'] > df['macd_dea']) &
                    (df['macd_dea'] > df['macd_dea'].shift(1))).astype(np.int8)

        trend_down = ((df['macd_dif'] < df['macd_dea']) &
                      (df['macd_dea'] < df['macd_dea'].shift(1))).astype(np.int8)

        # 组合信号
        macd_bull_attack = (
            (above_zero == 1) &
            (hist_positive == 1) &
            (acc_positive == 1) &
            (acc_spike == 1) &
            (trend_up == 1)
        ).astype(np.int8)

        macd_bottom_ignite = (
            (below_zero == 1) &
            (golden_cross == 1) &
            (acc_spike == 1)
        ).astype(np.int8)

        macd_momentum = (
            (hist_positive == 1) &
            (acc_positive == 1)
        ).astype(np.int8)

        macd_strong_bull = (
            (macd_bull_attack == 1) &
            (df['macd_hist'] > df['macd_hist'].rolling(10, min_periods=1).mean())
        ).astype(np.int8)

        # 前期置零
        safe = max(lookback_window, fill_window, 30)
        for s in [golden_cross, dead_cross, macd_bull_attack,
                  macd_bottom_ignite, macd_momentum, macd_strong_bull]:
            s.iloc[:safe] = 0

        return {
            "golden_cross": golden_cross,
            "dead_cross": dead_cross,
            "above_zero": above_zero,
            "below_zero": below_zero,
            "hist_positive": hist_positive,
            "hist_slope": hist_slope,
            "acc": acc,
            "acc_spike": acc_spike,
            "acc_positive": acc_positive,
            "trend_up": trend_up,
            "trend_down": trend_down,
            "macd_bull_attack": macd_bull_attack,
            "macd_bottom_ignite": macd_bottom_ignite,
            "macd_momentum": macd_momentum,
            "macd_strong_bull": macd_strong_bull,
        }

    def build_taker_rules(self):
        if 'taker_buy_base' not in self.df.columns:
            raise ValueError("缺少列: taker_buy_base")

        if 'taker_buy_ma_20' in self.df.columns:
            taker_buy_ma_20 = self.df['taker_buy_ma_20']
        else:
            taker_buy_ma_20 = self.df['taker_buy_base'].rolling(20, min_periods=1).mean()

        past_taker_ma20 = taker_buy_ma_20.shift(1)
        taker_ratio = self.df['taker_buy_base'] / (past_taker_ma20 + 1e-8)
        taker_ratio.iloc[:21] = 0
        return taker_ratio

    def build_count_rules(self):
        if 'count' not in self.df.columns:
            raise ValueError("缺少列: count")

        if 'count_ma_20' in self.df.columns:
            count_ma_20 = self.df['count_ma_20']
        else:
            count_ma_20 = self.df['count'].rolling(20, min_periods=1).mean()

        past_count_ma20 = count_ma_20.shift(1)
        count_ratio = self.df['count'] / (past_count_ma20 + 1e-8)
        count_ratio.iloc[:21] = 0
        return count_ratio


def evaluate_market_signals(
    builder_df: pd.DataFrame,
    ema_comp: pd.Series,
    ema7_ang: pd.Series,
    price_brk: pd.Series,
    vol_rat: pd.Series,
    taker_rat: pd.Series,
    count_rat: pd.Series,
    macd_r: Dict[str, pd.Series],
    score_threshold: float = 70
) -> pd.DataFrame:
    """
    优化后的评分引擎
    - 零轴上方不再是硬性必要条件
    - 支持主升浪攻击 + 底部点火两条路径
    """
    df = builder_df.assign(
        ema_compression_ratio=ema_comp,
        ema7_angle=ema7_ang,
        price_breakout=price_brk,
        volume_ratio=vol_rat,
        taker_ratio=taker_rat,
        count_ratio=count_rat,
        **macd_r
    )

    # ==================== 硬过滤条件（已去掉零轴上方强制要求） ====================
    # 路径1：趋势延续（主升浪）
    hard_trend = (
        (df['macd_dif'] > df['macd_dea']) &
        (df['trend_up'] == 1) &
        (df['hist_positive'] == 1) &
        (df['volume_ratio'] > 2.2) &
        (df['taker_ratio'] > 1.8) &
        (df['count_ratio'] > 2.5)
    )

    # 路径2：底部点火（允许在零轴下方）
    hard_bottom = (
        (df['macd_bottom_ignite'] == 1) &
        (df['volume_ratio'] > 2.5) &
        (df['taker_ratio'] > 2.0) &
        (df['count_ratio'] > 2.8)
    )

    hard_condition = hard_trend | hard_bottom

    # ==================== 软评分 ====================
    score = pd.Series(0.0, index=df.index)
    mask = hard_condition

    if mask.any():
        # 核心信号高权重
        score.loc[mask] += df.loc[mask, 'macd_bull_attack'] * 28
        score.loc[mask] += df.loc[mask, 'macd_bottom_ignite'] * 32   # 底部点火给更高分
        score.loc[mask] += df.loc[mask, 'macd_strong_bull'] * 10

        # 动能爆发
        score.loc[mask] += df.loc[mask, 'acc_spike'] * 18

        # 量能
        score.loc[mask] += (df.loc[mask, 'volume_ratio'] > 2.5) * 12
        score.loc[mask] += (df.loc[mask, 'taker_ratio'] > 2.0) * 10
        score.loc[mask] += (df.loc[mask, 'count_ratio'] > 3.0) * 8

        # 角度与突破
        score.loc[mask] += (df.loc[mask, 'ema7_angle'] > 35) * 10
        score.loc[mask] += (df.loc[mask, 'price_breakout'] > 0.03) * 6

        # 零轴上方额外加分（非必须，但加分）
        score.loc[mask] += df.loc[mask, 'above_zero'] * 5

    df['total_score'] = score
    df['signal'] = ((df['total_score'] >= score_threshold) & hard_condition).astype(np.int8)

    return df


def process_single_file(
    builder_df: pd.DataFrame,
    holding_period: int = 180,
    score_threshold: float = 70
) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
    try:
        df = builder_df.copy()

        builder = FeatureRuleBuilder(df)
        builder.run_base()

        ema_compression, ema7_angle, price_breakout = builder.build_ema_rules()
        volume_ratio = builder.build_volume_rules()
        macd_rules = builder.build_macd_rules(lookback_window=120, spike_k=2.2, fill_window=60)
        taker_ratio = builder.build_taker_rules()
        count_ratio = builder.build_count_rules()

        result_df = evaluate_market_signals(
            builder.df,
            ema_compression, ema7_angle, price_breakout,
            volume_ratio, taker_ratio, count_ratio, macd_rules,
            score_threshold=score_threshold
        )

        # 计算未来最大收益（回测用）
        result_df = result_df.reset_index(drop=True)
        result_df['max_return_in_180'] = 0.0

        signals_idx = result_df[result_df['signal'] == 1].index

        for idx in signals_idx:
            end_idx = min(idx + holding_period, len(result_df) - 1)
            if end_idx > idx:
                future_close = result_df['close'].iloc[idx+1:end_idx+1]
                if not future_close.empty:
                    max_price = future_close.max()
                    entry_price = result_df['close'].iloc[idx]
                    result_df.loc[idx, 'max_return_in_180'] = (max_price / entry_price - 1) * 100

        triggers = result_df[result_df['signal'] == 1]
        win_rate = 0.0
        avg_max = 0.0
        total_pnl = 0.0

        if not triggers.empty:
            wins = (triggers['max_return_in_180'] >= 10).sum()
            win_rate = round(wins / len(triggers) * 100, 2)
            avg_max = round(triggers['max_return_in_180'].mean(), 3)
            total_pnl = round(triggers['max_return_in_180'].sum(), 3)

        summary = {
            'rows': len(result_df),
            'signals': len(triggers),
            'win_rate': win_rate,
            'avg_max_return': avg_max,
            'total_max_pnl': total_pnl,
        }

        return summary, result_df

    except Exception as e:
        print(f"❌ DataFrame 处理失败: {e}")
        return None, None


# ====================== 批量回测 ======================
def main_batch():
    folder_path = Path("rank_pro/data/sample_data")

    if not folder_path.exists():
        print(f"❌ 文件夹不存在: {folder_path}")
        return

    # 调试用：只跑单个文件
    csv_files = list(folder_path.glob("DYMUSDT.csv"))
    # csv_files = list(folder_path.glob("*.csv"))  # 全部文件

    print(f"找到 {len(csv_files)} 个CSV文件，开始回测...\n")

    results = []

    for file in csv_files:
        print(f"正在处理 → {file.name}")
        try:
            df = pd.read_csv(file)

            # 可选：如果有时间列
            # if 'open_time' in df.columns:
            #     df['open_time'] = pd.to_datetime(df['open_time'])

            summary, result_df = process_single_file(
                builder_df=df,
                holding_period=180,
                score_threshold=70          # 已根据新规则适当降低
            )

            if summary:
                summary['symbol'] = file.stem
                results.append(summary)

                if result_df is not None and result_df['signal'].sum() > 0:
                    print(f"\n📍 {file.stem} 触发信号位置：")
                    triggers = result_df[result_df['signal'] == 1].copy()

                    show_cols = [
                        'close', 'total_score', 'ema7_angle',
                        'ema_compression_ratio', 'macd_bull_attack',
                        'macd_bottom_ignite', 'max_return_in_180'
                    ]
                    if 'open_time' in triggers.columns:
                        show_cols = ['open_time'] + show_cols

                    print(triggers[show_cols].round(3).to_string())
                    print("-" * 90)

        except Exception as e:
            print(f"❌ 处理 {file.name} 失败: {e}")

    if results:
        summary_df = pd.DataFrame(results)
        print("\n" + "=" * 90)
        print("🏆 回测总汇总")
        print("=" * 90)
        print(summary_df.to_string(index=False))

        summary_df.to_csv("batch_backtest_summary.csv", index=False)
        print(f"\n✅ 汇总已保存至: batch_backtest_summary.csv")

    print("\n回测完成！")


if __name__ == "__main__":
    main_batch()