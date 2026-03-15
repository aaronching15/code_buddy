# -*- coding: utf-8 -*-
"""
monitor.py
实时监控页面：Portfolio Greeks 汇总、预警逻辑
供 app.py 的"实时监控"页面调用
兼容 Python 3.7
"""

import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from bs_engine import bs_price_and_greeks, portfolio_greeks


# ─── 预警阈值常量 ─────────────────────────────────────────────────────────────
ALERT_NET_DELTA      = 0.30   # |Net Delta| 超此值建议 Delta 对冲
ALERT_DAYS_TO_EXPIRY = 5      # 到期剩余天数 <= 此值时提示滚仓
ALERT_LOSS_RATIO     = 1.5    # 亏损超过 初始权利金 * 此倍数时预警


# ─── 持仓重定价 ───────────────────────────────────────────────────────────────

def reprice_positions(
    positions: List[Dict],
    spot: float,
    r: float = 0.02,
    as_of: Optional[datetime.date] = None,
) -> pd.DataFrame:
    """
    对持仓列表重定价，返回含 Greeks 和浮动盈亏的 DataFrame（卖方视角）。
    """
    if as_of is None:
        as_of = datetime.date.today()

    rows = []
    for pos in positions:
        K        = float(pos["strike_price"])
        cp       = str(pos["call_put"]).upper()
        opt_type = "call" if cp == "C" else "put"
        exp_date = pos["exp_date"]
        if isinstance(exp_date, str):
            exp_date = datetime.date.fromisoformat(exp_date)
        T_days  = max((exp_date - as_of).days, 0)
        sigma   = float(pos.get("iv", 0.20))
        mult    = int(pos.get("multiplier", 10000))
        qty     = int(pos.get("quantity", 1))
        open_px = float(pos.get("open_price", 0))

        # 卖方视角 Greeks
        res = bs_price_and_greeks(spot, K, T_days, r, sigma, opt_type, short_pos=True)

        # 浮动盈亏 = (开仓价 - 当前价) * 乘数 * 数量
        current_px  = res["price"]
        float_pnl   = (open_px - current_px) * mult * qty
        prem_income = open_px * mult * qty  # 初始权利金收入

        row = {
            "合约":      pos.get("ts_code", "{}{}{}".format(pos.get("ts_code",""), cp, K)),
            "方向":      "卖出",
            "类型":      "认购" if cp == "C" else "认沽",
            "行权价":    K,
            "到期日":    exp_date.strftime("%Y-%m-%d"),
            "剩余天数":  T_days,
            "开仓价":    round(open_px, 4),
            "当前价":    round(current_px, 4),
            "IV":        round(sigma * 100, 2),
            "Delta":     round(res["delta"], 4),
            "Gamma":     round(res["gamma"], 4),
            "Vega":      round(res["vega"],  4),
            "Theta":     round(res["theta"], 4),
            "数量":      qty,
            "浮动盈亏":  round(float_pnl, 2),
            "权利金":    round(prem_income, 2),
            # 内部字段，用于预警
            "_float_pnl":   float_pnl,
            "_prem_income": prem_income,
            "_T_days":      T_days,
            "_delta_raw":   res["delta"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


def calc_portfolio_greeks(df: pd.DataFrame) -> Dict:
    """
    汇总组合 Net Greeks（从已含 Greeks 列的 DataFrame 中计算）。
    """
    if df.empty:
        return {"net_delta": 0, "net_gamma": 0, "net_vega": 0, "net_theta": 0,
                "total_pnl": 0, "total_premium": 0}

    qty_col = "数量" if "数量" in df.columns else None
    qty = df[qty_col].values if qty_col else np.ones(len(df))

    return {
        "net_delta":     round(float(np.sum(df["Delta"].values * qty)), 4),
        "net_gamma":     round(float(np.sum(df["Gamma"].values * qty)), 4),
        "net_vega":      round(float(np.sum(df["Vega"].values  * qty)), 4),
        "net_theta":     round(float(np.sum(df["Theta"].values * qty)), 4),
        "total_pnl":     round(float(df["浮动盈亏"].sum()), 2),
        "total_premium": round(float(df["权利金"].sum()),   2),
    }


# ─── 预警逻辑 ─────────────────────────────────────────────────────────────────

def get_alerts(df: pd.DataFrame, net_greeks: Dict) -> List[Dict]:
    """
    生成预警信息列表。每条 dict: {level, title, message}
    level: "error" | "warning" | "info"
    """
    alerts = []

    # 1. Net Delta 偏离过大
    nd = abs(net_greeks.get("net_delta", 0))
    if nd > ALERT_NET_DELTA:
        alerts.append({
            "level":   "error",
            "title":   "Delta 风险",
            "message": "Net Delta = {:.4f}，超过阈值 ±{:.2f}，建议执行 Delta 对冲".format(
                net_greeks["net_delta"], ALERT_NET_DELTA),
        })

    # 2. 到期日临近
    if not df.empty and "_T_days" in df.columns:
        near_exp = df[df["_T_days"] <= ALERT_DAYS_TO_EXPIRY]
        if not near_exp.empty:
            contracts = near_exp["合约"].tolist()
            alerts.append({
                "level":   "warning",
                "title":   "滚仓预警",
                "message": "以下合约剩余 {} 天内到期，建议及时滚仓：{}".format(
                    ALERT_DAYS_TO_EXPIRY, ", ".join(str(c) for c in contracts)),
            })

    # 3. 亏损超阈值
    total_pnl  = net_greeks.get("total_pnl", 0)
    total_prem = net_greeks.get("total_premium", 1)
    if total_pnl < 0 and total_prem > 0:
        loss_ratio = abs(total_pnl) / total_prem
        if loss_ratio >= ALERT_LOSS_RATIO:
            alerts.append({
                "level":   "error",
                "title":   "亏损预警",
                "message": "组合亏损 {:.0f} 元，已达初始权利金的 {:.1f} 倍，请评估止损".format(
                    abs(total_pnl), loss_ratio),
            })

    return alerts


# ─── Streamlit 页面渲染 ───────────────────────────────────────────────────────

def render_monitor_page(positions: List[Dict], spot: float, r: float = 0.02):
    """
    渲染实时监控页面，直接调用此函数即可（在 app.py 的监控页中调用）。
    """
    if not positions:
        st.info("暂无持仓数据，请先在「持仓管理」页面添加持仓，或在「数据导入」页面上传持仓文件。")
        return

    as_of = datetime.date.today()
    df = reprice_positions(positions, spot, r=r, as_of=as_of)
    net = calc_portfolio_greeks(df)
    alerts = get_alerts(df, net)

    # ── 顶部指标卡片 ──────────────────────────────────────────────────────────
    st.markdown("### 组合 Greeks 实时汇总")
    c1, c2, c3, c4 = st.columns(4)
    delta_color = "normal" if abs(net["net_delta"]) <= ALERT_NET_DELTA else "inverse"
    c1.metric("Net Delta",  "{:.4f}".format(net["net_delta"]),  delta_color=delta_color)
    c2.metric("Net Gamma",  "{:.4f}".format(net["net_gamma"]))
    c3.metric("Net Vega",   "{:.4f}".format(net["net_vega"]))
    c4.metric("Net Theta",  "{:.4f}".format(net["net_theta"]),  delta_color="normal")

    st.markdown("---")
    p1, p2 = st.columns(2)
    p1.metric("组合浮动盈亏", "{:+,.0f} 元".format(net["total_pnl"]),
              delta_color="normal" if net["total_pnl"] >= 0 else "inverse")
    p2.metric("初始权利金收入", "{:,.0f} 元".format(net["total_premium"]))

    # ── 预警区域 ──────────────────────────────────────────────────────────────
    if alerts:
        st.markdown("---")
        st.markdown("### ⚠️ 风险预警")
        for alert in alerts:
            if alert["level"] == "error":
                st.error("**{}** — {}".format(alert["title"], alert["message"]))
            elif alert["level"] == "warning":
                st.warning("**{}** — {}".format(alert["title"], alert["message"]))
            else:
                st.info("**{}** — {}".format(alert["title"], alert["message"]))

    # ── 持仓明细表格 ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 持仓明细")

    display_cols = ["合约", "方向", "类型", "行权价", "到期日", "剩余天数",
                    "开仓价", "当前价", "IV", "Delta", "Gamma", "Vega", "Theta",
                    "数量", "浮动盈亏", "权利金"]
    display_df = df[[c for c in display_cols if c in df.columns]].copy()

    # 颜色渐变样式（浮动盈亏列红绿）
    def color_pnl(val):
        try:
            v = float(val)
            if v > 0:
                return "color: #00cc44"
            elif v < 0:
                return "color: #ff4444"
        except Exception:
            pass
        return ""

    def color_delta(val):
        try:
            v = float(val)
            intensity = min(abs(v) / 0.5, 1.0)
            if v > 0:
                g = int(100 + 155 * intensity)
                return "color: rgb(0, {}, 0)".format(g)
            else:
                r = int(100 + 155 * intensity)
                return "color: rgb({}, 0, 0)".format(r)
        except Exception:
            return ""

    styled = display_df.style \
        .applymap(color_pnl, subset=["浮动盈亏"]) \
        .applymap(color_delta, subset=["Delta"]) \
        .format({
            "行权价": "{:.3f}",
            "开仓价": "{:.4f}",
            "当前价": "{:.4f}",
            "IV":     "{:.1f}%",
            "Delta":  "{:.4f}",
            "Gamma":  "{:.4f}",
            "Vega":   "{:.4f}",
            "Theta":  "{:.4f}",
            "浮动盈亏": "{:+,.0f}",
            "权利金":  "{:,.0f}",
        })

    st.dataframe(styled, width='stretch')

    # ── 时间戳 ────────────────────────────────────────────────────────────────
    st.caption("数据时间：{}  |  当前价：{:.4f}  |  无风险利率：{:.1f}%".format(
        as_of.strftime("%Y-%m-%d"), spot, r * 100))


# ─── 自测（无 Streamlit）────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== monitor 自测（无 Streamlit）===")

    today    = datetime.date.today()
    exp_date = datetime.date(today.year, today.month, 28)
    if exp_date <= today:
        nm = today.replace(day=1) + datetime.timedelta(days=32)
        exp_date = nm.replace(day=28)

    positions = [
        {"ts_code": "50C300", "strike_price": 3.2, "call_put": "C",
         "exp_date": exp_date, "open_price": 0.030, "iv": 0.20,
         "multiplier": 10000, "quantity": 1},
        {"ts_code": "50P280", "strike_price": 2.8, "call_put": "P",
         "exp_date": exp_date, "open_price": 0.025, "iv": 0.22,
         "multiplier": 10000, "quantity": 1},
    ]

    df = reprice_positions(positions, spot=3.0, r=0.02)
    net = calc_portfolio_greeks(df)
    print("Net Greeks:", net)
    assert "net_delta" in net
    assert "total_pnl" in net

    alerts = get_alerts(df, net)
    print("预警数量:", len(alerts))
    for a in alerts:
        print(" [{}] {} - {}".format(a["level"], a["title"], a["message"]))

    print("[OK] monitor 自测完成")
