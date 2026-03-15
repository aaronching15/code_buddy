# -*- coding: utf-8 -*-
"""
backtest_engine.py
回测引擎：逐日 BS 重定价 + Delta 对冲 + 到期结算 + Sharpe/MDD/胜率统计
支持功能：
  - 真实期权收盘价（akshare）或 BS 理论价
  - 每日浮盈亏、浮盈亏比例
  - 资金占用模型（25% 名义本金保证金）控制最大建仓手数
  - 详细交易日志（开仓/Delta对冲/平仓/到期结算）
兼容 Python 3.7
"""

import datetime
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bs_engine import bs_price, bs_price_and_greeks, settlement_pnl


# ─── 资金占用计算 ──────────────────────────────────────────────────────────────

def calc_margin_per_lot(
    spot: float,
    multiplier: int = 10000,
    margin_ratio: float = 0.25,
) -> float:
    """
    计算卖出 1 手期权所需保证金（名义本金法）。
    保证金 = spot * multiplier * margin_ratio
    示例：spot=4.68, mult=10000, ratio=0.25 → 11700 元
    """
    return spot * multiplier * margin_ratio


def calc_max_lots(
    capital: float,
    spot: float,
    multiplier: int = 10000,
    margin_ratio: float = 0.25,
    n_legs: int = 2,
) -> int:
    """
    计算给定本金下最多可建仓的手数（每腿）。
    对于 Strangle：同时卖 1 手 call + 1 手 put，占用 2 手保证金。
    返回每腿手数（call 手数 = put 手数）。

    公式：
        margin_per_lot = spot * multiplier * margin_ratio
        total_margin   = n_legs * margin_per_lot * qty_per_leg
        qty_per_leg    = floor(capital / (n_legs * margin_per_lot))
    """
    margin_per_lot = calc_margin_per_lot(spot, multiplier, margin_ratio)
    if margin_per_lot <= 0:
        return 1
    # 每腿手数（call/put 各一半）
    qty = int(capital / (n_legs * margin_per_lot))
    return max(qty, 1)


def apply_capital_model(
    positions: List[Dict],
    spot: float,
    initial_capital: float,
    margin_ratio: float = 0.25,
    n_legs: int = 2,
) -> Tuple[List[Dict], Dict]:
    """
    根据本金和资金占用模型，自动设置每腿手数，返回调整后的 positions 和资金概览 dict。

    资金占用说明（Strangle）：
        - 名义本金 = spot × multiplier（1 手合约对应的标的市值）
        - 保证金   = 名义本金 × margin_ratio（默认 25%）
        - 每腿手数 = floor(capital / (n_legs × margin_per_lot))
        - 保证金总占用 = n_legs × margin_per_lot × qty_per_leg
    """
    if not positions:
        return positions, {}

    mult = int(positions[0].get("multiplier", 10000))
    margin_per_lot = calc_margin_per_lot(spot, mult, margin_ratio)
    qty_per_leg    = calc_max_lots(initial_capital, spot, mult, margin_ratio, n_legs)

    total_margin  = n_legs * margin_per_lot * qty_per_leg
    margin_pct    = total_margin / initial_capital * 100 if initial_capital > 0 else 0
    free_capital  = initial_capital - total_margin

    adjusted = []
    for pos in positions:
        p = dict(pos)
        p["quantity"] = qty_per_leg
        adjusted.append(p)

    capital_info = {
        "spot":             round(spot, 4),
        "multiplier":       mult,
        "margin_ratio":     margin_ratio,
        "margin_per_lot":   round(margin_per_lot, 2),
        "qty_per_leg":      qty_per_leg,
        "n_legs":           n_legs,
        "total_margin":     round(total_margin, 2),
        "margin_pct":       round(margin_pct, 2),
        "free_capital":     round(free_capital, 2),
        "initial_capital":  round(initial_capital, 2),
    }
    return adjusted, capital_info


# ─── 核心回测函数 ─────────────────────────────────────────────────────────────

def run_backtest(
    positions: List[Dict],
    price_path: pd.DataFrame,
    r: float = 0.02,
    refresh_iv: bool = False,
    delta_hedge: bool = True,
    hedge_threshold: float = 0.05,
    initial_capital: float = 1_000_000.0,
    margin_ratio: float = 0.25,
    use_real_close: bool = False,
    etf_symbol: str = "",
) -> Tuple[pd.DataFrame, Dict, List[Dict]]:
    """
    对一组卖出持仓运行逐日回测（含 Delta 对冲 + 资金占用模型）。

    Parameters
    ----------
    positions : list of dict
        每条 dict 至少包含:
          ts_code, strike_price, call_put (C/P), exp_date (date),
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
        当净 delta 绝对值 > threshold 时才执行对冲
    initial_capital : float
        期初本金（元）
    margin_ratio : float
        保证金占用比例（默认 25%）
    use_real_close : bool
        是否尝试通过 akshare 拉取当日期权真实收盘价（网络可用时使用）
    etf_symbol : str
        ETF 代码，use_real_close=True 时用于拉取行情

    Returns
    -------
    df_daily : DataFrame
        逐日明细（含浮盈亏、浮盈亏比例、net_delta、etf_shares）
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

    # ── 开仓日（第一天） ──────────────────────────────────────────────────────
    open_date = dates[0]
    open_spot = closes[open_date]

    # ── 资金占用模型：自动计算手数 ─────────────────────────────────────────────
    positions, capital_info = apply_capital_model(
        positions, open_spot, initial_capital,
        margin_ratio=margin_ratio, n_legs=len(positions),
    )

    # 收取的初始权利金（含手数）
    total_premium = 0.0
    for pos in positions:
        px   = float(pos["open_price"])
        mult = int(pos.get("multiplier", 10000))
        qty  = int(pos.get("quantity", 1))
        total_premium += px * mult * qty

    # 初始化 ETF 对冲仓位（份数，正=多头，负=空头）
    etf_shares = 0.0
    etf_cost   = 0.0   # 买卖 ETF 的累计现金流（买入为负）

    trade_log = []

    # ── 记录开仓 ─────────────────────────────────────────────────────────────
    qty_total = sum(int(p.get("quantity", 1)) for p in positions)
    call_legs = [p for p in positions if str(p.get("call_put","")).upper() == "C"]
    put_legs  = [p for p in positions if str(p.get("call_put","")).upper() == "P"]
    qty_per_leg = int(positions[0].get("quantity", 1)) if positions else 1

    trade_log.append({
        "日期":     open_date,
        "类型":     "开仓",
        "操作":     "卖出期权组合",
        "标的":     "期权",
        "方向":     "卖出",
        "数量":     qty_total,
        "价格":     round(total_premium / max(qty_total, 1), 4),
        "金额":     round(total_premium, 2),
        "说明":     (
            "卖出 Call{} × {}手 + Put{} × {}手，"
            "收取权利金 {:.2f} 元；"
            "保证金占用 {:.2f} 元（{:.1f}%本金），"
            "余可用资金 {:.2f} 元".format(
                "/".join(str(p["strike_price"]) for p in call_legs),
                qty_per_leg,
                "/".join(str(p["strike_price"]) for p in put_legs),
                qty_per_leg,
                total_premium,
                capital_info.get("total_margin", 0),
                capital_info.get("margin_pct", 0),
                capital_info.get("free_capital", 0),
            )
        ),
        "累计盈亏": 0.0,
        "净Delta":  0.0,
    })

    # ── 逐日循环 ─────────────────────────────────────────────────────────────
    daily_rows   = []
    # open_price_total：开仓时期权组合总价值（卖方收到的权利金）
    # 每日浮盈亏 = 开仓总价值 - 今日市值（负数=亏损，正数=盈利）
    open_value_total = total_premium   # 按手数计算的开仓总权利金

    for d in dates:
        spot = closes[d]
        row  = {"date": d, "spot": spot}
        day_opt_val  = 0.0   # 今天期权组合的市值（卖方须付出的金额）
        net_delta    = 0.0   # 今天组合净 delta

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
            code     = str(pos.get("ts_code", ""))

            # ── 获取当日期权价格 ────────────────────────────────────────────
            current_px = None

            # 尝试真实收盘价（仅在 use_real_close=True 且非到期日）
            if use_real_close and code and T_days > 0:
                try:
                    from data_loader import load_option_daily_close
                    real_px, _err = load_option_daily_close(code, d)
                    if real_px is not None and real_px > 0:
                        current_px = real_px
                except Exception:
                    pass

            # 回退到 BS 理论价
            if current_px is None:
                if T_days <= 0:
                    intrinsic  = max(spot - K, 0) if opt_type == "call" else max(K - spot, 0)
                    current_px = intrinsic
                else:
                    res = bs_price_and_greeks(spot, K, T_days, r, sigma,
                                              option_type=opt_type, short_pos=False)
                    current_px = res["price"]

            # Greeks（用于 delta 对冲）
            if T_days > 0:
                res = bs_price_and_greeks(spot, K, T_days, r, sigma,
                                          option_type=opt_type, short_pos=False)
                pos_delta = res["delta"]
            else:
                pos_delta = 0.0

            # 卖方组合当日市值
            day_opt_val += current_px * mult * qty

            # 卖方净 delta（short call → delta 为负，short put → delta 为正）
            sign = 1.0 if opt_type == "call" else -1.0
            net_delta += (-sign) * abs(pos_delta) * mult * qty

            col_bs  = "pos_{}_{}{}".format(i, cp, K)
            col_mkt = "mkt_{}_{}{}".format(i, cp, K)
            # 该腿浮盈亏（相对开仓价，按今日理论/市场价）
            row[col_bs]  = round((open_px - current_px) * mult * qty, 4)
            row[col_mkt] = round(current_px, 6)   # 今日期权价格

        # ── Delta 对冲 ───────────────────────────────────────────────────────
        hedge_action = None
        if delta_hedge and abs(net_delta) > hedge_threshold:
            target_shares = -net_delta
            delta_diff    = target_shares - etf_shares
            if abs(delta_diff) > 0.01:
                trade_val   = delta_diff * spot
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

        # ETF 仓位当日盈亏（价格变动收益）
        idx       = dates.index(d)
        prev_spot = closes[dates[idx - 1]] if idx > 0 else spot
        etf_daily_pnl = etf_shares * (spot - prev_spot)

        # 期权组合当日盈亏（按持仓浮盈亏）
        pos_keys = ["pos_{}_{}{}".format(i, str(p["call_put"]).upper(), p["strike_price"])
                    for i, p in enumerate(positions)]
        opt_daily_pnl = sum(row.get(k, 0.0) for k in pos_keys)

        # 浮盈亏（相对开仓价的累计盈亏）
        float_pnl      = open_value_total - day_opt_val  # 正=盈利
        float_pnl_pct  = float_pnl / open_value_total * 100 if open_value_total > 0 else 0.0
        float_pnl_cap  = float_pnl / initial_capital * 100 if initial_capital > 0 else 0.0

        row["opt_pnl"]          = round(opt_daily_pnl, 4)
        row["etf_pnl"]          = round(etf_daily_pnl, 4)
        row["total_pnl"]        = round(opt_daily_pnl + etf_daily_pnl, 4)
        row["net_delta"]        = round(net_delta, 4)
        row["etf_shares"]       = round(etf_shares, 4)
        row["opt_mkt_val"]      = round(day_opt_val, 4)        # 期权今日市值（须平仓成本）
        row["float_pnl"]        = round(float_pnl, 4)         # 浮盈亏（元）
        row["float_pnl_pct"]    = round(float_pnl_pct, 4)     # 浮盈亏 / 权利金 (%)
        row["float_pnl_cap_pct"]= round(float_pnl_cap, 4)     # 浮盈亏 / 本金 (%)
        daily_rows.append(row)

        if hedge_action is not None:
            trade_log.append(hedge_action)

    df_daily = pd.DataFrame(daily_rows)
    df_daily["cumulative_pnl"] = df_daily["total_pnl"].cumsum()

    # ── 到期平仓记录 ──────────────────────────────────────────────────────────
    exp_dates = set(pos["exp_date"] for pos in positions)
    for exp_d in sorted(exp_dates):
        if exp_d in closes:
            final_spot = closes[exp_d]
        elif dates:
            final_spot = closes[dates[-1]]
        else:
            continue

        settle_pnl_val = 0.0
        settle_desc_parts = []
        for pos in positions:
            if pos["exp_date"] != exp_d:
                continue
            K         = float(pos["strike_price"])
            cp        = str(pos["call_put"]).upper()
            opt_type  = "call" if cp == "C" else "put"
            mult      = int(pos.get("multiplier", 10000))
            qty       = int(pos.get("quantity", 1))
            open_px   = float(pos["open_price"])
            intrinsic = max(final_spot - K, 0) if opt_type == "call" else max(K - final_spot, 0)
            leg_settle = (open_px - intrinsic) * mult * qty
            settle_pnl_val += leg_settle
            settle_desc_parts.append(
                "{}K={} 内在值{:.4f} 盈亏{:.2f}×{}手".format(
                    cp, K, intrinsic, leg_settle / max(qty, 1), qty)
            )

        # 平仓 ETF 对冲仓位
        etf_close_pnl  = 0.0
        etf_close_desc = ""
        if abs(etf_shares) > 0.01:
            etf_close_pnl   = etf_shares * final_spot - etf_cost
            direction_close = "卖出" if etf_shares > 0 else "买入"
            etf_close_desc  = "，平仓ETF {:.4f}份@{:.4f}，对冲累计盈亏{:.2f}".format(
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
                "说明":     "到期平仓ETF对冲{}".format(etf_close_desc),
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
            "金额":     round(settle_pnl_val, 2),
            "说明":     "现货收盘{:.4f}；{}{}".format(
                final_spot,
                "；".join(settle_desc_parts),
                etf_close_desc,
            ),
            "累计盈亏": round(cum_pnl, 2),
            "净Delta":  0.0,
        })

    # 补全 trade_log 中的累计盈亏/净Delta 字段
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

    # ── 绩效统计 ──────────────────────────────────────────────────────────────
    stats = _calc_stats(df_daily, positions, r, total_premium, initial_capital)
    stats.update({
        "margin_per_lot":   capital_info.get("margin_per_lot", 0),
        "qty_per_leg":      capital_info.get("qty_per_leg", 1),
        "total_margin":     capital_info.get("total_margin", 0),
        "margin_pct":       capital_info.get("margin_pct", 0),
        "free_capital":     capital_info.get("free_capital", 0),
    })
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

    if total_premium <= 0:
        total_premium = 0.01
    base = initial_capital if initial_capital > 0 else total_premium

    final_pnl    = float(df["cumulative_pnl"].iloc[-1])
    total_return = final_pnl / base

    if math.isfinite(total_return) and total_return > -1:
        annualized = (1 + total_return) ** (252.0 / max(n_days, 1)) - 1
    else:
        annualized = total_return

    # 最大回撤
    cum      = df["cumulative_pnl"].values
    peak     = np.maximum.accumulate(cum)
    drawdown = (peak - cum) / (np.abs(peak) + 1e-9)
    max_dd   = float(np.max(drawdown))

    # Sharpe
    daily_pnl = df["total_pnl"].values
    if len(daily_pnl) > 1 and np.std(daily_pnl) > 0:
        excess = daily_pnl - r / 252 * base
        sharpe = float(np.mean(excess) / np.std(excess) * math.sqrt(252))
    else:
        sharpe = 0.0

    # 胜率
    win_days  = int(np.sum(daily_pnl > 0))
    loss_days = int(np.sum(daily_pnl < 0))
    win_rate  = win_days / max(n_days, 1)

    # 盈亏比
    avg_win  = float(np.mean(daily_pnl[daily_pnl > 0]))  if win_days  > 0 else 0.0
    avg_loss = float(np.mean(np.abs(daily_pnl[daily_pnl < 0]))) if loss_days > 0 else 1.0
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    # 最终浮盈亏（以最后一日的 float_pnl 为准）
    final_float_pnl     = float(df["float_pnl"].iloc[-1])     if "float_pnl"     in df.columns else final_pnl
    final_float_pct_prem = float(df["float_pnl_pct"].iloc[-1]) if "float_pnl_pct" in df.columns else 0.0
    final_float_pct_cap  = float(df["float_pnl_cap_pct"].iloc[-1]) if "float_pnl_cap_pct" in df.columns else 0.0

    return {
        "total_premium":       round(total_premium, 2),
        "initial_capital":     round(initial_capital, 2),
        "final_pnl":           round(final_pnl, 2),
        "final_float_pnl":     round(final_float_pnl, 2),
        "final_float_pct_prem":round(final_float_pct_prem, 2),
        "final_float_pct_cap": round(final_float_pct_cap, 2),
        "total_return_pct":    round(total_return * 100, 2),
        "annualized_pct":      round(annualized * 100, 2),
        "max_drawdown_pct":    round(max_dd * 100, 2),
        "sharpe":              round(sharpe, 3),
        "win_rate_pct":        round(win_rate * 100, 2),
        "profit_loss_ratio":   round(pl_ratio, 3),
        "n_days":              n_days,
    }


# ─── 多组合批量回测 ────────────────────────────────────────────────────────────

def batch_backtest(
    strangle_pairs: List[Dict],
    price_path: pd.DataFrame,
    r: float = 0.02,
    initial_capital: float = 1_000_000.0,
    margin_ratio: float = 0.25,
    delta_hedge: bool = True,
    hedge_threshold: float = 0.05,
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
                "ts_code":      str(leg.get("ts_code", "")),
                "strike_price": float(leg["strike_price"]),
                "call_put":     str(leg["call_put"]).upper(),
                "exp_date":     leg["exp_date"],
                "open_price":   open_px,
                "iv":           float(leg.get("iv") or 0.20),
                "multiplier":   int(leg.get("multiplier") or 10000),
                "quantity":     1,
            })

        df_daily, stats, trade_log = run_backtest(
            positions, price_path, r=r,
            initial_capital=initial_capital,
            margin_ratio=margin_ratio,
            delta_hedge=delta_hedge,
            hedge_threshold=hedge_threshold,
        )
        results.append({
            "label":     label,
            "df_daily":  df_daily,
            "stats":     stats,
            "trade_log": trade_log,
        })
    return results


# ─── 自测 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== backtest_engine 自测 ===")

    today    = datetime.date.today()
    exp_date = datetime.date(today.year, today.month, 28)
    if exp_date <= today:
        nm = today.replace(day=1) + datetime.timedelta(days=32)
        exp_date = nm.replace(day=28)

    dates  = [today + datetime.timedelta(days=i) for i in range(20)]
    closes = [4.68 + 0.05 * math.sin(i * 0.5) for i in range(20)]
    price_path = pd.DataFrame({"date": dates, "close": closes})

    positions = [
        {
            "ts_code":      "510050C3200",
            "strike_price": 5.0, "call_put": "C", "exp_date": exp_date,
            "open_price": 0.030, "iv": 0.20, "multiplier": 10000, "quantity": 1,
        },
        {
            "ts_code":      "510050P4300",
            "strike_price": 4.3, "call_put": "P", "exp_date": exp_date,
            "open_price": 0.025, "iv": 0.22, "multiplier": 10000, "quantity": 1,
        },
    ]

    df_daily, stats, trade_log = run_backtest(
        positions, price_path, r=0.02,
        delta_hedge=True, initial_capital=1_000_000.0, margin_ratio=0.25,
    )

    print("资金模型：")
    for k in ("qty_per_leg", "margin_per_lot", "total_margin", "margin_pct", "free_capital"):
        print("  {:25s}: {}".format(k, stats.get(k, "N/A")))

    print("\n逐日盈亏（前5行）:")
    cols = ["date", "spot", "opt_pnl", "etf_pnl", "total_pnl",
            "cumulative_pnl", "float_pnl", "float_pnl_pct", "float_pnl_cap_pct"]
    cols = [c for c in cols if c in df_daily.columns]
    print(df_daily[cols].head(5).to_string())

    print("\n绩效统计:")
    for k, v in stats.items():
        print("  {:30s}: {}".format(k, v))

    print("\n交易记录（{}笔）:".format(len(trade_log)))
    for rec in trade_log:
        print("  {}  [{}]  {}".format(rec["日期"], rec["类型"], rec["说明"]))

    assert "sharpe" in stats
    assert "float_pnl" in df_daily.columns
    assert stats["qty_per_leg"] > 0
    print("\n[OK] backtest_engine 自测完成")
