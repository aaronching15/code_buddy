# -*- coding: utf-8 -*-
"""
backtest_engine.py
回测引擎：逐日 BS 重定价 + 到期结算 + Sharpe/MDD/胜率统计
兼容 Python 3.7
"""

import datetime
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bs_engine import bs_price, settlement_pnl


# ─── 核心回测函数 ────────────────────────────────────────────────────────────

def run_backtest(
    positions: List[Dict],
    price_path: pd.DataFrame,
    r: float = 0.02,
    refresh_iv: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """
    对一组卖出持仓运行逐日回测。

    Parameters
    ----------
    positions : list of dict
        每条 dict 至少包含:
          strike_price, call_put (C/P), exp_date (date),
          open_price, iv, multiplier, quantity
    price_path : DataFrame
        列：date (date), close (float)，按日期升序排列
    r : float
        无风险利率
    refresh_iv : bool
        是否用当天价格反解 IV（当前版本固定用开仓 IV，保留接口）

    Returns
    -------
    df_daily : DataFrame
        逐日明细，列：date, spot, [每条持仓的 pnl], total_pnl, cumulative_pnl
    stats : dict
        绩效统计：total_return, annualized_return, max_drawdown,
                  sharpe, win_rate, profit_loss_ratio, total_premium
    """
    if price_path.empty:
        return pd.DataFrame(), {}

    price_path = price_path.sort_values("date").reset_index(drop=True)
    dates  = list(price_path["date"])
    closes = dict(zip(price_path["date"], price_path["close"]))

    # 确认最早/最晚到期日
    for pos in positions:
        if isinstance(pos["exp_date"], str):
            pos["exp_date"] = datetime.date.fromisoformat(pos["exp_date"])

    # ── 逐日循环 ────────────────────────────────────────────────────────────
    daily_rows = []
    for d in dates:
        spot = closes[d]
        row = {"date": d, "spot": spot}
        day_total = 0.0

        for i, pos in enumerate(positions):
            K        = float(pos["strike_price"])
            cp       = str(pos["call_put"]).upper()
            opt_type = "call" if cp == "C" else "put"
            exp_date = pos["exp_date"]
            T_days   = max((exp_date - d).days, 0)
            sigma    = float(pos["iv"])
            mult     = int(pos.get("multiplier", 10000))
            qty      = int(pos.get("quantity", 1))
            open_px  = float(pos["open_price"])

            if T_days <= 0:
                # 到期结算
                intrinsic = settlement_pnl(spot, K, open_px, opt_type, mult)
                pos_pnl = intrinsic * qty
            else:
                current_px = bs_price(spot, K, T_days, r, sigma, opt_type)
                # 卖方：开仓价 - 当前价 → 正 = 赚钱
                pos_pnl = (open_px - current_px) * mult * qty

            col = "pos_{}_{}{}".format(i, cp, K)
            row[col] = round(pos_pnl, 4)
            day_total += pos_pnl

        row["total_pnl"] = round(day_total, 4)
        daily_rows.append(row)

    df_daily = pd.DataFrame(daily_rows)
    df_daily["cumulative_pnl"] = df_daily["total_pnl"].cumsum()

    # ── 绩效统计 ────────────────────────────────────────────────────────────
    stats = _calc_stats(df_daily, positions, r)
    return df_daily, stats


def _calc_stats(df: pd.DataFrame, positions: List[Dict], r: float) -> Dict:
    """计算绩效统计指标"""
    if df.empty or "cumulative_pnl" not in df.columns:
        return {}

    n_days = len(df)

    # total_premium：收到的初始权利金总额（卖方基准）
    # open_price 可能为 0 或 NaN（期权链未填写），用 bs_price 兜底，再用 0.01 兜底
    total_premium = 0.0
    for p in positions:
        px = float(p.get("open_price") or 0)
        if px <= 0:
            px = float(p.get("bs_price") or 0)
        if px <= 0:
            px = 0.01  # 极端兜底，避免除零
        total_premium += px * int(p.get("multiplier", 10000)) * int(p.get("quantity", 1))

    final_pnl    = float(df["cumulative_pnl"].iloc[-1])
    total_return = final_pnl / total_premium if total_premium > 0 else 0.0

    # annualized：避免 NaN / Inf
    if math.isfinite(total_return) and total_return > -1:
        annualized = (1 + total_return) ** (252.0 / max(n_days, 1)) - 1
    else:
        annualized = total_return

    # 最大回撤
    cum = df["cumulative_pnl"].values
    peak = np.maximum.accumulate(cum)
    drawdown = (peak - cum) / (np.abs(peak) + 1e-9)
    max_dd = float(np.max(drawdown))

    # Sharpe：日度收益 / 波动
    daily_pnl = df["total_pnl"].values
    if len(daily_pnl) > 1 and np.std(daily_pnl) > 0:
        excess = daily_pnl - r / 252 * total_premium
        sharpe = float(np.mean(excess) / np.std(excess) * math.sqrt(252))
    else:
        sharpe = 0.0

    # 胜率：逐日盈利天数 / 总天数
    win_days  = int(np.sum(daily_pnl > 0))
    loss_days = int(np.sum(daily_pnl < 0))
    win_rate  = win_days / max(n_days, 1)

    # 盈亏比
    avg_win  = float(np.mean(daily_pnl[daily_pnl > 0])) if win_days > 0  else 0.0
    avg_loss = float(np.mean(np.abs(daily_pnl[daily_pnl < 0]))) if loss_days > 0 else 1.0
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    return {
        "total_premium":     round(total_premium, 2),
        "final_pnl":         round(final_pnl, 2),
        "total_return_pct":  round(total_return * 100, 2),
        "annualized_pct":    round(annualized * 100, 2),
        "max_drawdown_pct":  round(max_dd * 100, 2),
        "sharpe":            round(sharpe, 3),
        "win_rate_pct":      round(win_rate * 100, 2),
        "profit_loss_ratio": round(pl_ratio, 3),
        "n_days":            n_days,
    }


# ─── 多组合批量回测 ──────────────────────────────────────────────────────────

def batch_backtest(
    strangle_pairs: List[Dict],
    price_path: pd.DataFrame,
    r: float = 0.02,
) -> List[Dict]:
    """
    对多个 Strangle 组合批量运行回测，返回结果列表。
    每个元素：{"label": str, "df_daily": DataFrame, "stats": dict}
    """
    results = []
    for i, pair in enumerate(strangle_pairs):
        call_K = float(pair["call_leg"]["strike_price"])
        put_K  = float(pair["put_leg"]["strike_price"])
        label  = "Call{}/Put{}".format(call_K, put_K)

        positions = []
        for leg_key in ("call_leg", "put_leg"):
            leg = pair[leg_key]
            positions.append({
                "strike_price": float(leg["strike_price"]),
                "call_put":     str(leg["call_put"]).upper(),
                "exp_date":     leg["exp_date"],
                "open_price":   float(leg.get("open_price", leg.get("bs_price", 0.03))),
                "iv":           float(leg.get("iv", 0.20)),
                "multiplier":   int(leg.get("multiplier", 10000)),
                "quantity":     1,
            })

        df_daily, stats = run_backtest(positions, price_path, r=r)
        results.append({"label": label, "df_daily": df_daily, "stats": stats})
    return results


# ─── 自测 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== backtest_engine 自测 ===")

    today   = datetime.date.today()
    exp_date = datetime.date(today.year, today.month, 28)
    if exp_date <= today:
        nm = today.replace(day=1) + datetime.timedelta(days=32)
        exp_date = nm.replace(day=28)

    # 构造价格路径（模拟 20 天横盘）
    dates  = [today + datetime.timedelta(days=i) for i in range(20)]
    closes = [3.0 + 0.02 * math.sin(i * 0.5) for i in range(20)]  # 轻微震荡
    price_path = pd.DataFrame({"date": dates, "close": closes})

    positions = [
        {
            "strike_price": 3.2, "call_put": "C", "exp_date": exp_date,
            "open_price": 0.030, "iv": 0.20, "multiplier": 10000, "quantity": 1,
        },
        {
            "strike_price": 2.8, "call_put": "P", "exp_date": exp_date,
            "open_price": 0.025, "iv": 0.22, "multiplier": 10000, "quantity": 1,
        },
    ]

    df_daily, stats = run_backtest(positions, price_path, r=0.02)
    print("逐日盈亏（前 5 行）:")
    print(df_daily[["date", "spot", "total_pnl", "cumulative_pnl"]].head(5).to_string())
    print("\n绩效统计:")
    for k, v in stats.items():
        print("  {:25s}: {}".format(k, v))

    assert "sharpe" in stats, "stats 应含 sharpe"
    assert "max_drawdown_pct" in stats, "stats 应含 max_drawdown_pct"
    assert len(df_daily) == 20, "逐日结果行数应=20"
    print("\n[OK] backtest_engine 自测完成")
