"""

factor_engine.py — 因子引擎

20+ 内置因子：动量、反转、波动率、量价相关、MACD/RSI/布林带、趋势强度等
自定义表达式沙箱：支持 ret/ma/ema/std/rank/ts_rank/corr/delta/delay/log 等算子
全套评估指标：IC序列（Spearman）、IR、分层收益（单调性检验）、多空夏普/回撤、换手率、自相关

================
股票因子研究引擎
  - 数据标准化与预处理
  - 基础因子库（量价 / 技术 / 基本面代理）
  - 自定义表达式因子（支持 eval 安全沙箱）
  - 因子评估：IC序列、IR、分层收益、换手率、自相关

 研究主题
    │
    ▼
┌──────────────┐     提案 JSON      ┌──────────────┐
│  ProposerAgent│ ─────────────────▶ │  CriticAgent │
│  （提案者）   │ ◀───────────────── │  （批判者）  │
└──────────────┘   批判+建议修订     └──────────────┘
        │ 多轮迭代（最多N轮，连续2轮通过则早停）
        ▼
┌──────────────┐
│ MediatorAgent│  ← 综合裁定：采纳 / 不采纳
└──────────────┘
        │
        ▼
  FactorEngine 评估（IC / IR / 分层收益 / 多空回测）

  
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from scipy import stats

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────────────────

@dataclass
class FactorResult:
    """单因子评估结果"""
    name: str
    description: str
    expression: str                  # 计算表达式或伪代码
    economic_logic: str              # 经济学逻辑
    ic_series: pd.Series = field(default_factory=pd.Series)
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ir: float = 0.0                  # Information Ratio = IC均值/IC标准差
    icir: float = 0.0               # 同 IR，部分文献叫法
    ic_positive_ratio: float = 0.0  # IC>0 的占比
    turnover: float = 0.0           # 月均换手率
    layer_returns: pd.DataFrame = field(default_factory=pd.DataFrame)  # 分层收益
    long_short_return: float = 0.0  # 多空组合年化收益
    max_drawdown: float = 0.0       # 多空组合最大回撤
    sharpe: float = 0.0             # 多空组合夏普
    autocorr: float = 0.0           # 因子一阶自相关（衡量稳定性）
    coverage: float = 0.0           # 有效值覆盖率
    raw_factor: pd.DataFrame = field(default_factory=pd.DataFrame)     # 原始因子面板
    debate_log: List[Dict] = field(default_factory=list)               # 辩论日志


# ──────────────────────────────────────────────────────────────────
# 因子引擎
# ──────────────────────────────────────────────────────────────────

class FactorEngine:
    """
    因子引擎
    接收标准化行情 DataFrame，提供：
    1. 内置因子计算
    2. 自定义表达式因子
    3. 因子评估全套指标
    """

    REQUIRED_COLS = ["date", "symbol", "open", "high", "low", "close", "volume"]

    def __init__(self, price_df: pd.DataFrame):
        """
        Args:
            price_df: 宽表，列至少含 date/symbol/open/high/low/close/volume
        """
        self.raw = price_df.copy()
        self._validate()
        self.raw["date"] = pd.to_datetime(self.raw["date"])
        self.raw = self.raw.sort_values(["symbol", "date"]).reset_index(drop=True)
        # 以 (date, symbol) 为索引的面板
        self.panel: pd.DataFrame = self.raw.set_index(["date", "symbol"])

    # ── 校验 ──────────────────────────────────────────────────────

    def _validate(self):
        missing = [c for c in self.REQUIRED_COLS if c not in self.raw.columns]
        if missing:
            raise ValueError(f"数据缺少必要列: {missing}")

    # ── 基础序列计算（按 symbol 分组）────────────────────────────

    def _by_symbol(self, func, *col_args) -> pd.Series:
        """对每个 symbol 单独计算时序函数，返回与 raw 对齐的 Series"""
        return self.raw.groupby("symbol", group_keys=False).apply(
            lambda g: func(g, *col_args)
        )

    # ──────────────────────────────────────────────────────────────
    # 内置基础因子
    # ──────────────────────────────────────────────────────────────

    def compute_builtin(self, factor_name: str) -> pd.DataFrame:
        """
        计算内置因子，返回 (date, symbol) -> factor_value 的 DataFrame
        """
        fn = getattr(self, f"_factor_{factor_name}", None)
        if fn is None:
            raise ValueError(f"未知内置因子: {factor_name}，请用 list_builtins() 查看支持列表")
        series = fn()
        return self._to_panel(series, factor_name)

    def _to_panel(self, series: pd.Series, name: str) -> pd.DataFrame:
        """将对齐到 raw 的 Series 转为 pivot 面板 (date × symbol)"""
        df = self.raw[["date", "symbol"]].copy()
        df[name] = series.values
        return df.pivot(index="date", columns="symbol", values=name)

    # ---- 量价因子 ------------------------------------------------

    def _factor_momentum_1m(self) -> pd.Series:
        """1月动量：过去20日收益率"""
        return self._by_symbol(lambda g: g["close"].pct_change(20))

    def _factor_momentum_3m(self) -> pd.Series:
        """3月动量：过去60日收益率"""
        return self._by_symbol(lambda g: g["close"].pct_change(60))

    def _factor_momentum_6m(self) -> pd.Series:
        """6月动量（跳过最近1月）：过去120日收益 - 过去20日收益"""
        def _calc(g):
            r120 = g["close"].pct_change(120)
            r20  = g["close"].pct_change(20)
            return r120 - r20
        return self._by_symbol(_calc)

    def _factor_reversal_1w(self) -> pd.Series:
        """短期反转：过去5日收益率取负"""
        return self._by_symbol(lambda g: -g["close"].pct_change(5))

    def _factor_volatility_20(self) -> pd.Series:
        """20日波动率（收益率标准差）"""
        return self._by_symbol(lambda g: g["close"].pct_change().rolling(20).std())

    def _factor_volatility_60(self) -> pd.Series:
        """60日波动率"""
        return self._by_symbol(lambda g: g["close"].pct_change().rolling(60).std())

    def _factor_volume_ma5_ratio(self) -> pd.Series:
        """量比：当日成交量 / 5日均量"""
        return self._by_symbol(
            lambda g: g["volume"] / g["volume"].rolling(5).mean().replace(0, np.nan)
        )

    def _factor_volume_ma20_ratio(self) -> pd.Series:
        """20日量比"""
        return self._by_symbol(
            lambda g: g["volume"] / g["volume"].rolling(20).mean().replace(0, np.nan)
        )

    def _factor_turnover_rate_proxy(self) -> pd.Series:
        """换手率代理：成交量/60日均成交量，衡量流动性变化"""
        return self._by_symbol(
            lambda g: g["volume"] / g["volume"].rolling(60).mean().replace(0, np.nan)
        )

    def _factor_price_to_high_52w(self) -> pd.Series:
        """52周价格强度：收盘价 / 252日最高价"""
        return self._by_symbol(
            lambda g: g["close"] / g["high"].rolling(252).max().replace(0, np.nan)
        )

    def _factor_price_to_low_52w(self) -> pd.Series:
        """52周低点距离：收盘价 / 252日最低价 - 1"""
        return self._by_symbol(
            lambda g: g["close"] / g["low"].rolling(252).min().replace(0, np.nan) - 1
        )

    # ---- 技术指标因子 -------------------------------------------

    def _factor_rsi_14(self) -> pd.Series:
        """RSI(14)"""
        def _rsi(g):
            d = g["close"].diff()
            gain = d.clip(lower=0).rolling(14).mean()
            loss = (-d.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            return 100 - 100 / (1 + rs)
        return self._by_symbol(_rsi)

    def _factor_macd_signal(self) -> pd.Series:
        """MACD 柱状值（DIF - DEA），衡量短期动量加速"""
        def _macd(g):
            ema12 = g["close"].ewm(span=12, adjust=False).mean()
            ema26 = g["close"].ewm(span=26, adjust=False).mean()
            dif   = ema12 - ema26
            dea   = dif.ewm(span=9, adjust=False).mean()
            return dif - dea
        return self._by_symbol(_macd)

    def _factor_bollinger_position(self) -> pd.Series:
        """布林带相对位置：(close - 下轨) / (上轨 - 下轨)，0~1"""
        def _bp(g):
            ma  = g["close"].rolling(20).mean()
            std = g["close"].rolling(20).std()
            upper = ma + 2 * std
            lower = ma - 2 * std
            band  = (upper - lower).replace(0, np.nan)
            return (g["close"] - lower) / band
        return self._by_symbol(_bp)

    def _factor_atr_ratio(self) -> pd.Series:
        """ATR比率：14日ATR / 收盘价，衡量相对波动"""
        def _atr(g):
            h, l, c = g["high"], g["low"], g["close"]
            tr = pd.concat([h - l,
                            (h - c.shift()).abs(),
                            (l - c.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            return atr / c.replace(0, np.nan)
        return self._by_symbol(_atr)

    def _factor_trend_strength(self) -> pd.Series:
        """趋势强度：60日线性回归斜率 × 60 / 收盘价均值（标准化）"""
        def _ts(g):
            def slope(x):
                if x.isna().any() or len(x) < 10:
                    return np.nan
                t = np.arange(len(x))
                k, *_ = np.polyfit(t, x.values, 1)
                return k * len(x) / x.mean()
            return g["close"].rolling(60).apply(slope, raw=False)
        return self._by_symbol(_ts)

    def _factor_price_acceleration(self) -> pd.Series:
        """价格加速度：5日动量 - 20日动量（动量加速）"""
        def _pa(g):
            return g["close"].pct_change(5) - g["close"].pct_change(20)
        return self._by_symbol(_pa)

    def _factor_volume_price_corr(self) -> pd.Series:
        """量价相关性：20日窗口内收益率与成交量的相关系数"""
        def _vpc(g):
            ret = g["close"].pct_change()
            vol = g["volume"]
            return ret.rolling(20).corr(vol)
        return self._by_symbol(_vpc)

    def _factor_high_low_ratio(self) -> pd.Series:
        """振幅因子：(最高-最低)/前收，衡量日内博弈强度"""
        return self._by_symbol(
            lambda g: (g["high"] - g["low"]) / g["close"].shift(1).replace(0, np.nan)
        )

    def _factor_open_gap(self) -> pd.Series:
        """隔夜跳空：开盘价/前收 - 1"""
        return self._by_symbol(
            lambda g: g["open"] / g["close"].shift(1).replace(0, np.nan) - 1
        )

    def _factor_intraday_return(self) -> pd.Series:
        """日内收益：收盘价/开盘价 - 1"""
        return self._by_symbol(
            lambda g: g["close"] / g["open"].replace(0, np.nan) - 1
        )

    def _factor_volume_trend(self) -> pd.Series:
        """成交量趋势：5日均量 / 20日均量"""
        return self._by_symbol(
            lambda g: g["volume"].rolling(5).mean() / g["volume"].rolling(20).mean().replace(0, np.nan)
        )

    # ──────────────────────────────────────────────────────────────
    # 自定义表达式因子
    # ──────────────────────────────────────────────────────────────

    def compute_expression(self, expression: str, name: str = "custom") -> pd.DataFrame:
        """
        执行 LLM 生成的因子表达式（受限沙箱）

        expression 中可使用的变量：
            close, open, high, low, volume  —— pd.Series（按 symbol 分组后的时序）
        可调用函数：
            ret(n)         — n日收益率
            ma(x, n)       — n日简单均线
            ema(x, n)      — n日指数均线
            std(x, n)      — n日滚动标准差
            rank(x)        — 截面排名（0-1）
            ts_rank(x, n)  — 时序滚动排名（0-1）
            corr(x, y, n)  — n日滚动相关
            delta(x, n)    — x.diff(n)
            delay(x, n)    — x.shift(n)
            log(x)         — np.log(x)
            abs(x)         — x.abs()
            sign(x)        — np.sign(x)
        """
        results = []
        for sym, grp in self.raw.groupby("symbol", sort=False):
            grp = grp.sort_values("date").reset_index(drop=True)
            close  = grp["close"].copy()
            open_  = grp["open"].copy()
            high   = grp["high"].copy()
            low    = grp["low"].copy()
            volume = grp["volume"].copy()

            # ---- 沙箱函数 ----
            def ret(n):              return close.pct_change(n)
            def ma(x, n):            return x.rolling(n).mean()
            def ema(x, n):           return x.ewm(span=n, adjust=False).mean()
            def std(x, n):           return x.rolling(n).std()
            def rank(x):             return x.rank(pct=True)
            def ts_rank(x, n):       return x.rolling(n).apply(lambda v: pd.Series(v).rank(pct=True).iloc[-1], raw=False)
            def corr(x, y, n):       return x.rolling(n).corr(y)
            def delta(x, n):         return x.diff(n)
            def delay(x, n):         return x.shift(n)
            def log(x):              return np.log(x.replace(0, np.nan))
            def abs_(x):             return x.abs()
            def sign(x):             return np.sign(x)

            _ns = {
                "close": close, "open": open_, "high": high,
                "low": low, "volume": volume,
                "ret": ret, "ma": ma, "ema": ema, "std": std,
                "rank": rank, "ts_rank": ts_rank, "corr": corr,
                "delta": delta, "delay": delay, "log": log,
                "abs": abs_, "sign": sign,
                "np": np, "pd": pd,
            }

            try:
                val = eval(expression, {"__builtins__": {}}, _ns)  # noqa: S307
                if isinstance(val, pd.Series):
                    val = val.values
                results.append(pd.DataFrame({
                    "date":   grp["date"].values,
                    "symbol": sym,
                    name:     val,
                }))
            except Exception as e:
                # 单个标的计算失败不中断整体
                results.append(pd.DataFrame({
                    "date":   grp["date"].values,
                    "symbol": sym,
                    name:     np.nan,
                }))

        all_df = pd.concat(results, ignore_index=True)
        return all_df.pivot(index="date", columns="symbol", values=name)

    # ──────────────────────────────────────────────────────────────
    # 因子评估
    # ──────────────────────────────────────────────────────────────

    def evaluate(
        self,
        factor_panel: pd.DataFrame,
        forward_days: int = 20,
        n_layers: int = 5,
        factor_name: str = "factor",
    ) -> FactorResult:
        """
        全面评估因子

        Args:
            factor_panel : pivot 表 (date × symbol)
            forward_days : 预测未来几个交易日收益（默认 20）
            n_layers     : 分层数量
            factor_name  : 因子名称

        Returns:
            FactorResult
        """
        # 计算远期收益
        close_panel = self._close_panel()
        fwd_ret = close_panel.pct_change(forward_days).shift(-forward_days)

        # 对齐
        common_dates   = factor_panel.index.intersection(fwd_ret.index)
        common_symbols = factor_panel.columns.intersection(fwd_ret.columns)
        f = factor_panel.loc[common_dates, common_symbols]
        r = fwd_ret.loc[common_dates, common_symbols]

        # ---- IC 序列 ----
        ic_list = []
        for dt in common_dates:
            fi = f.loc[dt].dropna()
            ri = r.loc[dt].dropna()
            idx = fi.index.intersection(ri.index)
            if len(idx) < 5:
                ic_list.append((dt, np.nan))
                continue
            rho, _ = stats.spearmanr(fi[idx], ri[idx])
            ic_list.append((dt, rho))

        ic_s = pd.Series(
            [v for _, v in ic_list],
            index=[d for d, _ in ic_list],
            name="IC"
        ).dropna()

        ic_mean = float(ic_s.mean())
        ic_std  = float(ic_s.std())
        ir      = ic_mean / ic_std if ic_std > 0 else 0.0
        ic_pos  = float((ic_s > 0).mean())

        # ---- 分层收益 ----
        layer_ret_list = []
        for dt in common_dates:
            fi = f.loc[dt].dropna()
            ri = r.loc[dt].dropna()
            idx = fi.index.intersection(ri.index)
            if len(idx) < n_layers * 2:
                continue
            labels = pd.qcut(fi[idx], n_layers, labels=False, duplicates="drop")
            for lbl in range(n_layers):
                syms = labels[labels == lbl].index
                if len(syms) > 0:
                    layer_ret_list.append({
                        "date":  dt,
                        "layer": lbl + 1,
                        "ret":   float(ri[syms].mean()),
                    })

        if layer_ret_list:
            layer_df   = pd.DataFrame(layer_ret_list)
            layer_mean = layer_df.groupby("layer")["ret"].mean().rename("平均收益")
            layer_std  = layer_df.groupby("layer")["ret"].std().rename("收益标准差")
            layer_info = pd.concat([layer_mean, layer_std], axis=1)
        else:
            layer_info = pd.DataFrame()

        # ---- 多空收益（第 n 层做多，第 1 层做空）----
        ls_ret_series = pd.Series(dtype=float)
        if layer_ret_list:
            ld = pd.DataFrame(layer_ret_list)
            top_ret = ld[ld["layer"] == n_layers].set_index("date")["ret"]
            bot_ret = ld[ld["layer"] == 1].set_index("date")["ret"]
            idx_c   = top_ret.index.intersection(bot_ret.index)
            ls_ret_series = (top_ret[idx_c] - bot_ret[idx_c])
            annual_factor = 252 / forward_days
            ls_annual    = float(ls_ret_series.mean() * annual_factor)
            cum          = (1 + ls_ret_series).cumprod()
            peak         = cum.cummax()
            dd           = ((cum - peak) / peak)
            max_dd       = float(dd.min())
            ls_sharpe    = float(
                ls_ret_series.mean() / ls_ret_series.std() * np.sqrt(annual_factor)
            ) if ls_ret_series.std() > 0 else 0.0
        else:
            ls_annual = max_dd = ls_sharpe = 0.0

        # ---- 换手率（相邻期因子排名变化）----
        def _turnover(panel: pd.DataFrame) -> float:
            ranks = panel.rank(axis=1, pct=True)
            diff  = ranks.diff().abs().mean(axis=1)
            return float(diff.mean())

        turnover = _turnover(f)

        # ---- 覆盖率 ----
        coverage = float(f.notna().mean().mean())

        # ---- 一阶自相关 ----
        flat_ac = []
        for sym in f.columns:
            s = f[sym].dropna()
            if len(s) > 2:
                flat_ac.append(s.autocorr(1))
        autocorr = float(np.nanmean(flat_ac)) if flat_ac else 0.0

        return FactorResult(
            name=factor_name,
            description="",
            expression="",
            economic_logic="",
            ic_series=ic_s,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ir=ir,
            icir=ir,
            ic_positive_ratio=ic_pos,
            turnover=turnover,
            layer_returns=layer_info,
            long_short_return=ls_annual,
            max_drawdown=max_dd,
            sharpe=ls_sharpe,
            autocorr=autocorr,
            coverage=coverage,
            raw_factor=factor_panel,
        )

    def _close_panel(self) -> pd.DataFrame:
        return self.raw.pivot(index="date", columns="symbol", values="close")

    # ──────────────────────────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def list_builtins() -> List[str]:
        """列出所有内置因子名称"""
        return [
            "momentum_1m", "momentum_3m", "momentum_6m",
            "reversal_1w",
            "volatility_20", "volatility_60",
            "volume_ma5_ratio", "volume_ma20_ratio",
            "turnover_rate_proxy",
            "price_to_high_52w", "price_to_low_52w",
            "rsi_14", "macd_signal", "bollinger_position",
            "atr_ratio", "trend_strength", "price_acceleration",
            "volume_price_corr", "high_low_ratio",
            "open_gap", "intraday_return", "volume_trend",
        ]

    @staticmethod
    def generate_demo_data(
        n_stocks: int = 30,
        n_days: int = 500,
        seed: int = 42,
    ) -> pd.DataFrame:
        """生成演示行情数据"""
        np.random.seed(seed)
        symbols = [f"SH{60000 + i:04d}" for i in range(n_stocks)]
        dates   = pd.date_range("2022-01-01", periods=n_days, freq="B")
        rows    = []
        for sym in symbols:
            base  = np.random.uniform(10, 100)
            price = base * np.exp(np.cumsum(np.random.randn(n_days) * 0.015))
            price = np.maximum(price, 1.0)
            for i, dt in enumerate(dates):
                o = price[i] * (1 + np.random.randn() * 0.003)
                h = price[i] * (1 + abs(np.random.randn()) * 0.008)
                l = price[i] * (1 - abs(np.random.randn()) * 0.008)
                v = np.random.randint(100_000, 5_000_000)
                rows.append({
                    "date":   dt,
                    "symbol": sym,
                    "open":   round(o, 2),
                    "high":   round(max(o, h, price[i]), 2),
                    "low":    round(min(o, l, price[i]), 2),
                    "close":  round(price[i], 2),
                    "volume": v,
                })
        return pd.DataFrame(rows)
