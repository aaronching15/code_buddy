# -*- coding: utf-8 -*-
"""
strangle_builder.py
Strangle 选仓逻辑：OTM 筛选、Delta 范围过滤、组合构建
兼容 Python 3.7
"""

import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bs_engine import bs_price_and_greeks


# ─── 核心筛选函数 ────────────────────────────────────────────────────────────

def enrich_greeks(
    df: pd.DataFrame,
    spot: float,
    r: float = 0.02,
    as_of: Optional[datetime.date] = None,
) -> pd.DataFrame:
    """
    对期权列表 DataFrame 补充 BS 定价和 Greeks（买方视角）。
    输入 df 必须包含：strike_price, call_put, exp_date, iv
    如果 iv 缺失则用 0.20 填充（仅用于 Greeks 估算）。
    返回新增列：bs_price, delta, gamma, vega, theta
    """
    if as_of is None:
        as_of = datetime.date.today()

    records = []
    for _, row in df.iterrows():
        K         = float(row["strike_price"])
        iv        = float(row["iv"]) if pd.notna(row.get("iv")) else 0.20
        exp_date  = row["exp_date"]
        if isinstance(exp_date, str):
            exp_date = datetime.date.fromisoformat(exp_date)
        T_days = max((exp_date - as_of).days, 0)
        cp     = str(row.get("call_put", "C")).upper()
        opt_type = "call" if cp == "C" else "put"

        res = bs_price_and_greeks(spot, K, T_days, r, iv, option_type=opt_type, short_pos=False)
        records.append({
            "bs_price": res["price"],
            "delta":    res["delta"],
            "gamma":    res["gamma"],
            "vega":     res["vega"],
            "theta":    res["theta"],
        })

    enriched = df.copy()
    for key in ["bs_price", "delta", "gamma", "vega", "theta"]:
        enriched[key] = [r[key] for r in records]
    return enriched


def filter_otm_options(
    df: pd.DataFrame,
    spot: float,
    delta_min: float = 0.10,
    delta_max: float = 0.30,
) -> pd.DataFrame:
    """
    筛选 OTM 期权：
    - OTM call：strike_price > spot，且 delta in [delta_min, delta_max]
    - OTM put ：strike_price < spot，且 |delta| in [delta_min, delta_max]

    如果 df 没有 delta 列，调用 enrich_greeks 前需先传入 spot/r/as_of。
    本函数假设 df 已有 delta 列（由 enrich_greeks 补充或手动上传）。
    """
    if "delta" not in df.columns:
        raise ValueError("df 缺少 delta 列，请先调用 enrich_greeks()")

    # OTM call
    mask_call = (
        (df["call_put"].str.upper() == "C") &
        (df["strike_price"] > spot) &
        (df["delta"].abs() >= delta_min) &
        (df["delta"].abs() <= delta_max)
    )
    # OTM put
    mask_put = (
        (df["call_put"].str.upper() == "P") &
        (df["strike_price"] < spot) &
        (df["delta"].abs() >= delta_min) &
        (df["delta"].abs() <= delta_max)
    )
    result = df[mask_call | mask_put].copy()
    result["otm_pct"] = ((result["strike_price"] - spot) / spot * 100).round(2)
    return result.reset_index(drop=True)


def build_strangle_pairs(
    df_otm: pd.DataFrame,
    mode: str = "nearest_delta",
) -> List[Dict]:
    """
    从筛选后的 OTM 期权列表中构建 Strangle 对。
    mode:
      - "nearest_delta" : 选 |delta| 最接近的 call + put 各一
      - "all_pairs"     : 所有 call x put 的笛卡尔积（用于批量回测）

    返回 list of dict，每个 dict 含 call_leg / put_leg 两个 row dict。
    """
    calls = df_otm[df_otm["call_put"].str.upper() == "C"].copy()
    puts  = df_otm[df_otm["call_put"].str.upper() == "P"].copy()

    if calls.empty or puts.empty:
        return []

    if mode == "nearest_delta":
        # 选 delta 最接近 0.20 的各一支
        target = 0.20
        calls["_dist"] = (calls["delta"].abs() - target).abs()
        puts["_dist"]  = (puts["delta"].abs()  - target).abs()
        best_call = calls.nsmallest(1, "_dist").iloc[0].to_dict()
        best_put  = puts.nsmallest(1, "_dist").iloc[0].to_dict()
        return [{"call_leg": best_call, "put_leg": best_put}]

    elif mode == "all_pairs":
        pairs = []
        for _, c_row in calls.iterrows():
            for _, p_row in puts.iterrows():
                pairs.append({"call_leg": c_row.to_dict(), "put_leg": p_row.to_dict()})
        return pairs

    else:
        raise ValueError("未知 mode：{}".format(mode))


def strangle_to_positions(pair: Dict, quantity: int = 1) -> List[Dict]:
    """
    将 Strangle 对转换为持仓列表（卖出方向）。
    返回 list of position dict，供 backtest_engine 和 monitor 使用。
    """
    positions = []
    for leg_key in ("call_leg", "put_leg"):
        leg = pair[leg_key]
        # open_price 优先用实际成交价，其次用 BS 理论价，最后用 0.01 兜底（避免除零）
        open_px = float(leg.get("open_price") or 0)
        if open_px <= 0:
            open_px = float(leg.get("bs_price") or 0)
        if open_px <= 0:
            open_px = 0.01

        pos = {
            "ts_code":      leg.get("ts_code", ""),
            "strike_price": float(leg["strike_price"]),
            "call_put":     str(leg["call_put"]).upper(),
            "exp_date":     leg["exp_date"],
            "open_price":   open_px,
            "bs_price":     float(leg.get("bs_price") or open_px),
            "iv":           float(leg.get("iv") or 0.20),
            "multiplier":   int(leg.get("multiplier") or 10000),
            "quantity":     quantity,
            "direction":    "short",  # 卖方
        }
        positions.append(pos)
    return positions


def summarize_strangle(pair: Dict, spot: float) -> Dict:
    """
    计算 Strangle 组合的关键指标：
    - 总权利金（call premium + put premium）
    - 盈利区间 [lower_breakeven, upper_breakeven]
    - 最大收益
    """
    call_leg = pair["call_leg"]
    put_leg  = pair["put_leg"]

    call_K   = float(call_leg["strike_price"])
    put_K    = float(put_leg["strike_price"])
    call_prem = float(call_leg.get("open_price", call_leg.get("bs_price", 0)))
    put_prem  = float(put_leg.get("open_price",  put_leg.get("bs_price", 0)))
    mult      = int(call_leg.get("multiplier", 10000))

    total_premium    = (call_prem + put_prem) * mult
    upper_breakeven  = call_K + call_prem + put_prem
    lower_breakeven  = put_K  - call_prem - put_prem
    profit_range_pct = (upper_breakeven - lower_breakeven) / spot * 100

    return {
        "call_strike":       call_K,
        "put_strike":        put_K,
        "call_premium":      round(call_prem, 4),
        "put_premium":       round(put_prem,  4),
        "total_premium":     round(total_premium, 2),
        "upper_breakeven":   round(upper_breakeven, 4),
        "lower_breakeven":   round(lower_breakeven, 4),
        "profit_range_pct":  round(profit_range_pct, 2),
        "max_profit":        round(total_premium, 2),
        "call_delta":        round(float(call_leg.get("delta", 0)), 4),
        "put_delta":         round(float(put_leg.get("delta",  0)), 4),
    }


# ─── 自测 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import io
    print("=== strangle_builder 自测 ===")

    # 构造一批虚拟期权
    today = datetime.date.today()
    exp   = datetime.date(today.year, today.month, 28)
    if exp <= today:
        next_month = today.replace(day=1) + datetime.timedelta(days=32)
        exp = next_month.replace(day=28)

    spot = 3.00
    r    = 0.02

    rows = []
    for K in [2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3]:
        for cp in ["C", "P"]:
            rows.append({
                "ts_code":      "510050",
                "strike_price": K,
                "call_put":     cp,
                "exp_date":     exp,
                "open_price":   0.03,
                "iv":           0.20,
                "multiplier":   10000,
            })
    df = pd.DataFrame(rows)

    # 补充 Greeks
    df_enriched = enrich_greeks(df, spot=spot, r=r)
    print("Greeks 补充完毕，delta 范围: [{:.3f}, {:.3f}]".format(
        df_enriched["delta"].min(), df_enriched["delta"].max()
    ))

    # 筛选 OTM
    df_otm = filter_otm_options(df_enriched, spot=spot, delta_min=0.10, delta_max=0.30)
    print("OTM 筛选结果: {} 支（delta 0.10~0.30）".format(len(df_otm)))
    assert len(df_otm) > 0, "OTM 筛选结果为空"

    # 构建 Strangle
    pairs = build_strangle_pairs(df_otm, mode="nearest_delta")
    assert len(pairs) == 1, "nearest_delta 应返回 1 对"
    summary = summarize_strangle(pairs[0], spot=spot)
    print("Strangle 摘要:")
    for k, v in summary.items():
        print("  {:25s}: {}".format(k, v))

    assert summary["upper_breakeven"] > spot, "上方盈亏平衡点应 > 当前价"
    assert summary["lower_breakeven"] < spot, "下方盈亏平衡点应 < 当前价"

    # 转持仓
    positions = strangle_to_positions(pairs[0])
    assert len(positions) == 2, "应有 2 条持仓（call + put）"
    assert all(p["direction"] == "short" for p in positions), "方向应为 short"
    print("\n持仓方向:", [p["direction"] for p in positions])
    print("[OK] strangle_builder 自测完成")
