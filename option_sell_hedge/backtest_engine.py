# -*- coding: utf-8 -*-
"""
backtest_engine.py
回测引擎：逐日 BS 重定价 + Delta 对冲 + 到期结算 + Sharpe/MDD/胜率统计
兼容 Python 3.7
"""

import datetime
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bs_engine import bs_price, bs_price_and_greeks, settlement_pnl


# ─── 核心回测函数 ────────────────────────────────────────────────────────────

def run_backtest(
    positions: List[Dict],
    price_path: pd.DataFrame,
    r: float = 0.02,
    refresh_iv: bool = False,
    delta_hedge: bool = True,
    hedge_threshold: float = 0.05,
    initial_capital: float = 1_000_000.0,
) -> Tuple[pd.DataFrame, Dict, List[Dict]]:
    """
    对一组卖出持仓运行逐日回测（含 Delta 对冲）。

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
        是否用当天价格反解 IV（保留接口）
    delta_hedge : bool
        是否模拟 Delta 对冲（每日 ETF 买卖）
    hedge_threshold : float
        当净 delta 绝对值 > threshold 时才执行对冲（默认 0.05）
    initial_capital : float
        期初本金（元），用于计算收益率和 Sharpe

    Returns
    -------
    df_daily : DataFrame
        逐日明细
    stats : dict
        绩效统计
    trade_log : list of dict
        每笔交易记录（开仓/对冲/平仓/到期结算）
    """
    if price_path.empty:
        return pd.DataFrame(), {}, []

    price_path = price_path.sort_values("date").reset_index(drop=True)
    dates  = list(price_path["date"])
    closes = dict(zip(price_path["date"], price_path["close"]))

    # 统一 exp_date 为 date 类型
    for pos in positions:
        if isinstance(pos["exp_date"], str):
            pos["exp_date"] = datetime.date.fromisoformat(pos["exp_date"])

    # ── 开仓日（第一天） ─────────────────────────────────────────────────────
    open_date = dates[0]
    open_spot = closes[open_date]

    # 收取的初始权利金
    total_premium = 0.0
    for pos in positions:
        px   = float(pos["open_price"])
        mult = int(pos.get("multiplier", 10000))
        qty  = int(pos.get("quantity", 1))
        total_premium += px * mult * qty

    # 初始化 ETF 对冲仓位（份数，正=多头，负=空头）
    etf_shares = 0.0
    etf_cost   = 0.0  # 历史买卖 ETF 的累计现金流（买入为负）

    trade_log = []

    # ── 记录开仓 ────────────────────────────────────────────────────────────
    trade_log.append({
        "日期":     open_date,
        "类型":     "开仓",
        "操作":     "卖出期权组合",
        "标的":     "期权",
        "方向":     "卖出",
        "数量":     sum(int(p.get("quantity", 1)) for p in positions),
        "价格":     round(total_premium / max(sum(int(p.get("quantity", 1)) for p in positions), 1), 4),
        "金额":     round(total_premium, 2),
        "说明":     "卖出 Call{} + Put{}，收取权利金 {:.2f} 元".format(
            "/".join(str(p["strike_price"]) for p in positions if str(p.get("call_put","")).upper()=="C"),
            "/".join(str(p["strike_price"]) for p in positions if str(p.get("call_put","")).upper()=="P"),
            total_premium,
        ),
        "累计盈亏": 0.0,
        "净Delta":  0.0,
    })

    # ── 逐日循环 ────────────────────────────────────────────────────────────
    daily_rows   = []
    prev_opt_val = total_premium  # 上一天期权组合市值（卖方视角：收入为正基准）

    for d in dates:
        spot = closes[d]
        row  = {"date": d, "spot": spot}
        day_opt_val  = 0.0  # 今天期权组合的理论市值（卖方须付出的金额）
        net_delta    = 0.0  # 今天组合净 delta（卖方：需要反向对冲）

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
                # 到期结算：按内在价值
                intrinsic  = max(spot - K, 0) if opt_type == "call" else max(K - spot, 0)
                current_px = intrinsic
                pos_delta  = 0.0
            else:
                res = bs_price_and_greeks(spot, K, T_days, r, sigma,
                                          option_type=opt_type, short_pos=False)
                current_px = res["price"]
                pos_delta  = res["delta"]

            # 卖方：期权组合今日市值（若需平仓需付出的成本）
            day_opt_val += current_px * mult * qty
            # 卖方净 delta：持有 short call 时 delta 为负（若 spot 上涨需买入 ETF 对冲）
            sign = 1.0 if opt_type == "call" else -1.0
            net_delta += (-sign) * abs(pos_delta) * mult * qty

            col = "pos_{}_{}{}".format(i, cp, K)
            # 当天该腿的盈亏 = 开仓价 - 今日价（卖方持有，时间价值流向卖方）
            row[col] = round((open_px - current_px) * mult * qty, 4)

        # ── Delta 对冲 ──────────────────────────────────────────────────────
        hedge_pnl    = 0.0
        hedge_action = None

        if delta_hedge and abs(net_delta) > hedge_threshold:
            # 目标 ETF 持仓 = -net_delta（对冲组合净 delta）
            target_shares = -net_delta
            delta_diff    = target_shares - etf_shares

            if abs(delta_diff) > 0.01:
                trade_val   = delta_diff * spot  # 正=买入 ETF，负=卖出 ETF
                etf_cost   += trade_val
                etf_shares  = target_shares
                direction   = "买入" if delta_diff > 0 else "卖出"
                hedge_action = {
                    "日期":     d,
                    "类型":     "Delta对冲",
                    "操作":     "{}ETF".format(direction),
                    "标的":     "ETF",
                    "方向":     direction,
                    "数量":     round(abs(delta_diff), 4),
                    "价格":     round(spot, 4),
                    "金额":     round(abs(trade_val), 2),
                    "说明":     "净Delta={:.4f}，{}ETF {:.4f}份@{:.4f}，ETF总持仓={:.4f}份".format(
                        net_delta, direction, abs(delta_diff), spot, etf_shares,
                    ),
                }

        # ETF 仓位当天盈亏（持有收益，不是交易盈亏）
        # 用上一日收盘价计算（首日用开仓价）
        prev_date = dates[dates.index(d) - 1] if dates.index(d) > 0 else d
        prev_spot = closes.get(prev_date, spot)
        etf_daily_pnl = etf_shares * (spot - prev_spot)

        # 期权组合盈亏 = 开仓价值 - 今日价值
        opt_daily_pnl = sum(
            row.get("pos_{}_{}{}".format(i, str(p["call_put"]).upper(), p["strike_price"]), 0.0)
            for i, p in enumerate(positions)
        )

        row["opt_pnl"]   = round(opt_daily_pnl, 4)
        row["etf_pnl"]   = round(etf_daily_pnl, 4)
        row["total_pnl"] = round(opt_daily_pnl + etf_daily_pnl, 4)
        row["net_delta"] = round(net_delta, 4)
        row["etf_shares"] = round(etf_shares, 4)
        daily_rows.append(row)

        # 记录对冲交易
        if hedge_action is not None:
            trade_log.append(hedge_action)

    df_daily = pd.DataFrame(daily_rows)
    df_daily["cumulative_pnl"] = df_daily["total_pnl"].cumsum()

    # ── 到期平仓记录 ────────────────────────────────────────────────────────
    exp_dates = set(pos["exp_date"] for pos in positions)
    for exp_d in sorted(exp_dates):
        if exp_d in closes:
            final_spot = closes[exp_d]
        elif dates:
            final_spot = closes[dates[-1]]
        else:
            continue

        settle_pnl = 0.0
        settle_desc_parts = []
        for pos in positions:
            if pos["exp_date"] != exp_d:
                continue
            K        = float(pos["strike_price"])
            cp       = str(pos["call_put"]).upper()
            opt_type = "call" if cp == "C" else "put"
            mult     = int(pos.get("multiplier", 10000))
            qty      = int(pos.get("quantity", 1))
            open_px  = float(pos["open_price"])
            intrinsic = max(final_spot - K, 0) if opt_type == "call" else max(K - final_spot, 0)
            leg_settle = (open_px - intrinsic) * mult * qty
            settle_pnl += leg_settle
            settle_desc_parts.append(
                "{}K={} 内在值{:.4f} 盈亏{:.2f}".format(cp, K, intrinsic, leg_settle)
            )

        # 平仓 ETF 对冲仓位
        etf_close_pnl = 0.0
        etf_close_desc = ""
        if abs(etf_shares) > 0.01:
            etf_close_pnl  = etf_shares * final_spot - etf_cost
            direction_close = "卖出" if etf_shares > 0 else "买入"
            etf_close_desc  = "，平仓ETF {:.4f}份@{:.4f}，对冲盈亏{:.2f}".format(
                abs(etf_shares), final_spot, etf_close_pnl,
            )
            trade_log.append({
                "日期":     exp_d,
                "类型":     "平仓对冲",
                "操作":     "{}ETF".format(direction_close),
                "标的":     "ETF",
                "方向":     direction_close,
                "数量":     round(abs(etf_shares), 4),
                "价格":     round(final_spot, 4),
                "金额":     round(abs(etf_shares) * final_spot, 2),
                "说明":     "到期平仓ETF对冲仓位{etf_close_desc}".format(
                    etf_close_desc=etf_close_desc),
            })

        cum_pnl = float(df_daily["cumulative_pnl"].iloc[-1]) if not df_daily.empty else 0.0
        trade_log.append({
            "日期":     exp_d,
            "类型":     "到期结算",
            "操作":     "期权到期",
            "标的":     "期权",
            "方向":     "结算",
            "数量":     len([p for p in positions if p["exp_date"] == exp_d]),
            "价格":     round(final_spot, 4),
            "金额":     round(settle_pnl, 2),
            "说明":     "现货收盘{:.4f}；{}{}".format(
                final_spot, "；".join(settle_desc_parts), etf_close_desc,
            ),
            "累计盈亏": round(cum_pnl, 2),
            "净Delta":  0.0,
        })

    # 补全 trade_log 中的累计盈亏字段
    _cum = 0.0
    if not df_daily.empty:
        date_to_cum = dict(zip(df_daily["date"], df_daily["cumulative_pnl"]))
        for rec in trade_log:
            d_rec = rec["日期"]
            if d_rec in date_to_cum:
                rec["累计盈亏"] = round(float(date_to_cum[d_rec]), 2)
            elif "累计盈亏" not in rec:
                rec["累计盈亏"] = 0.0
            if "净Delta" not in rec:
                rec["净Delta"] = 0.0

    # ── 绩效统计 ────────────────────────────────────────────────────────────
    stats = _calc_stats(df_daily, positions, r, total_premium, initial_capital)
    return df_daily, stats, trade_log


def _calc_stats(
    df: pd.DataFrame,
    positions: List[Dict],
    r: float,
    total_premium: float,
    initial_capital: float,
) -> Dict:
    """计算绩效统计指标"""
    if df.empty or "cumulative_pnl" not in df.columns:
        return {}

    n_days = len(df)

    # 避免 total_premium 为 0 导致除零
    if total_premium <= 0:
        total_premium = 0.01
    # 以初始本金为基准计算收益率
    base = initial_capital if initial_capital > 0 else total_premium

    final_pnl    = float(df["cumulative_pnl"].iloc[-1])
    total_return = final_pnl / base

    # annualized：避免 NaN / Inf
    if math.isfinite(total_return) and total_return > -1:
        annualized = (1 + total_return) ** (252.0 / max(n_days, 1)) - 1
    else:
        annualized = total_return

    # 最大回撤
    cum  = df["cumulative_pnl"].values
    peak = np.maximum.accumulate(cum)
    drawdown = (peak - cum) / (np.abs(peak) + 1e-9)
    max_dd = float(np.max(drawdown))

    # Sharpe：日度收益 / 波动
    daily_pnl = df["total_pnl"].values
    if len(daily_pnl) > 1 and np.std(daily_pnl) > 0:
        excess = daily_pnl - r / 252 * base
        sharpe = float(np.mean(excess) / np.std(excess) * math.sqrt(252))
    else:
        sharpe = 0.0

    # 胜率：逐日盈利天数
    win_days  = int(np.sum(daily_pnl > 0))
    loss_days = int(np.sum(daily_pnl < 0))
    win_rate  = win_days / max(n_days, 1)

    # 盈亏比
    avg_win  = float(np.mean(daily_pnl[daily_pnl > 0])) if win_days  > 0 else 0.0
    avg_loss = float(np.mean(np.abs(daily_pnl[daily_pnl < 0]))) if loss_days > 0 else 1.0
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    return {
        "total_premium":     round(total_premium, 2),
        "initial_capital":   round(initial_capital, 2),
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
    initial_capital: float = 1_000_000.0,
) -> List[Dict]:
    """
    对多个 Strangle 组合批量运行回测，返回结果列表。
    每个元素：{"label": str, "df_daily": DataFrame, "stats": dict, "trade_log": list}
    """
    results = []
    for i, pair in enumerate(strangle_pairs):
        call_K = float(pair["call_leg"]["strike_price"])
        put_K  = float(pair["put_leg"]["strike_price"])
        label  = "Call{}/Put{}".format(call_K, put_K)

        positions = []
        for leg_key in ("call_leg", "put_leg"):
            leg = pair[leg_key]
            open_px = float(leg.get("open_price") or 0)
            if open_px <= 0:
                open_px = float(leg.get("bs_price") or 0.03)
            positions.append({
                "strike_price": float(leg["strike_price"]),
                "call_put":     str(leg["call_put"]).upper(),
                "exp_date":     leg["exp_date"],
                "open_price":   open_px,
                "iv":           float(leg.get("iv") or 0.20),
                "multiplier":   int(leg.get("multiplier") or 10000),
                "quantity":     1,
            })

        df_daily, stats, trade_log = run_backtest(
            positions, price_path, r=r, initial_capital=initial_capital,
        )
        results.append({
            "label":     label,
            "df_daily":  df_daily,
            "stats":     stats,
            "trade_log": trade_log,
        })
    return results


# ─── 自测 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== backtest_engine 自测 ===")

    today    = datetime.date.today()
    exp_date = datetime.date(today.year, today.month, 28)
    if exp_date <= today:
        nm = today.replace(day=1) + datetime.timedelta(days=32)
        exp_date = nm.replace(day=28)

    dates  = [today + datetime.timedelta(days=i) for i in range(20)]
    closes = [3.0 + 0.02 * math.sin(i * 0.5) for i in range(20)]
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

    df_daily, stats, trade_log = run_backtest(
        positions, price_path, r=0.02, delta_hedge=True, initial_capital=1_000_000.0,
    )
    print("逐日盈亏（前 5 行）:")
    print(df_daily[["date", "spot", "opt_pnl", "etf_pnl", "total_pnl", "cumulative_pnl"]].head(5).to_string())
    print("\n绩效统计:")
    for k, v in stats.items():
        print("  {:25s}: {}".format(k, v))
    print("\n交易记录（{}笔）:".format(len(trade_log)))
    for rec in trade_log:
        print("  {}  [{}]  {}".format(rec["日期"], rec["类型"], rec["说明"]))

    assert "sharpe" in stats
    assert len(df_daily) == 20
    assert len(trade_log) >= 1
    print("\n[OK] backtest_engine 自测完成")
