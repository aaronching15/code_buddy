# -*- coding: utf-8 -*-
"""
app.py
Strangle 卖出对冲策略研究与监控系统 - Streamlit 主入口
4 页面：回测模拟 / 实时监控 / 持仓管理 / 数据导入
兼容 Python 3.7 + Streamlit 1.23.1
"""

import datetime
import math
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from bs_engine import bs_price_and_greeks, implied_vol
from backtest_engine import run_backtest, batch_backtest, calc_max_lots, calc_margin_per_lot
from data_loader import (
    UNDERLYING_MAP,
    build_template_df,
    load_options_from_excel,
    load_price_path_from_excel,
    load_spot_from_akshare,
    load_option_chain_from_akshare,
    get_latest_spot,
    save_positions,
    load_positions,
    save_fetched_data,
    load_fetched_data,
    list_saved_files,
)
from strangle_builder import (
    enrich_greeks,
    filter_otm_options,
    build_strangle_pairs,
    strangle_to_positions,
    summarize_strangle,
)
from monitor import render_monitor_page


# ─── 页面基础配置 ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Strangle 卖出策略",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 深色金融风格 CSS
st.markdown("""
<style>
    .main { background-color: #0e1117; color: #fafafa; }
    .stMetric { background: #1a1d23; border-radius: 8px; padding: 10px; }
    .alert-card-error   { border: 2px solid #ff4444; border-radius: 8px;
                          padding: 12px; margin: 6px 0; background: #1a0000; }
    .alert-card-warning { border: 2px solid #ffaa00; border-radius: 8px;
                          padding: 12px; margin: 6px 0; background: #1a1200; }
    div[data-testid="stDataFrame"] { font-size: 13px; }
    .sidebar-title { font-size: 18px; font-weight: 700; color: #1f77b4; }
</style>
""", unsafe_allow_html=True)


# ─── Session State 初始化 ─────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "positions":       [],       # 当前持仓列表
        "options_df":      None,     # 期权链 DataFrame
        "price_path_df":   None,     # 标的价格路径
        "spot":            3.0,      # 当前标的价格
        "backtest_result": None,     # 回测结果
        "data_source":     "手动上传",
        "underlying":      "50ETF (510050)",
        "data_path":       r"D:\auto_tc\data_sync",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-title">📊 Strangle 卖出策略</p>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "页面导航",
        ["回测模拟", "实时监控", "持仓管理", "数据导入"],
        key="page_nav",
    )

    st.markdown("---")
    st.markdown("**全局配置**")
    r_pct = st.number_input("无风险利率 r (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    r = r_pct / 100.0

    st.session_state["data_path"] = st.text_input(
        "数据保存路径", value=st.session_state["data_path"]
    )

    auto_refresh = st.checkbox("自动刷新（30s）", value=False)
    st.markdown("---")
    st.caption("Strangle Strategy v1.0\nPowered by Black-Scholes")


# ─── 自动刷新 ─────────────────────────────────────────────────────────────────
if auto_refresh and page == "实时监控":
    time.sleep(30)
    try:
        st.experimental_rerun()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# 页面 1：回测模拟
# ═══════════════════════════════════════════════════════════════════════════════
def page_backtest():
    st.title("📈 回测模拟")

    # ── 参数设置区 ─────────────────────────────────────────────────────────────
    with st.expander("参数设置", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            underlying = st.selectbox("标的", list(UNDERLYING_MAP.keys()),
                                      key="bt_underlying")
        with col2:
            delta_min = st.slider("Delta 下限", 0.05, 0.40, 0.10, 0.01, key="bt_dmin")
            delta_max = st.slider("Delta 上限", 0.10, 0.50, 0.30, 0.01, key="bt_dmax")
        with col3:
            spot_input = st.number_input("当前标的价格", value=float(st.session_state["spot"]),
                                         step=0.01, key="bt_spot")
            st.session_state["spot"] = spot_input

    spot = float(st.session_state["spot"])

    # ── 期权链展示 ─────────────────────────────────────────────────────────────
    df_options = st.session_state.get("options_df")
    if df_options is None or df_options.empty:
        st.info("请先在「数据导入」页面上传期权列表，或使用自动拉取功能加载数据。")
        return

    # 补充 Greeks
    try:
        df_enriched = enrich_greeks(df_options, spot=spot, r=r)
    except Exception as e:
        st.error("Greeks 计算失败：{}".format(e))
        return

    # 筛选 OTM
    df_otm = filter_otm_options(df_enriched, spot=spot,
                                delta_min=delta_min, delta_max=delta_max)

    st.markdown("#### 期权链（Delta 筛选后的 OTM 合约）")
    display_cols_chain = ["ts_code", "call_put", "strike_price", "exp_date",
                          "open_price", "iv", "delta", "gamma", "vega", "theta", "otm_pct"]
    show_cols = [c for c in display_cols_chain if c in df_otm.columns]

    if df_otm.empty:
        st.warning("未找到符合 Delta [{:.2f}, {:.2f}] 范围的 OTM 合约，请调整参数。".format(
            delta_min, delta_max))
    else:
        st.dataframe(
            df_otm[show_cols].style.format({
                "strike_price": "{:.3f}",
                "open_price":   "{:.4f}",
                "iv":           "{:.2%}",
                "delta":        "{:.4f}",
                "gamma":        "{:.4f}",
                "vega":         "{:.4f}",
                "theta":        "{:.4f}",
                "otm_pct":      "{:+.2f}%",
            }),
            width='stretch',
        )

        # ── Strangle 构建 ──────────────────────────────────────────────────────
        st.markdown("#### Strangle 组合构建")
        pair_mode = st.radio("构建模式", ["最优组合（nearest_delta）", "批量测试（all_pairs）"],
                             horizontal=True)
        mode_key  = "nearest_delta" if "最优" in pair_mode else "all_pairs"

        pairs = build_strangle_pairs(df_otm, mode=mode_key)
        if not pairs:
            st.warning("无法构建 Strangle 组合，请检查 OTM 期权是否同时包含 call 和 put。")
            return

        # 显示摘要
        summaries = [summarize_strangle(p, spot) for p in pairs]
        st.dataframe(pd.DataFrame(summaries), width='stretch')

        # ── 价格路径 ────────────────────────────────────────────────────────────
        df_price = st.session_state.get("price_path_df")
        if df_price is None or df_price.empty:
            st.info("请在「数据导入」页面上传标的价格路径，然后点击运行回测。")
            return

        # 回测参数
        with st.expander("回测参数", expanded=False):
            bt_col1, bt_col2, bt_col3 = st.columns(3)
            initial_capital = bt_col1.number_input(
                "期初本金（元）", value=1_000_000, step=100_000,
                min_value=10_000, key="bt_capital",
            )
            margin_ratio_pct = bt_col2.slider(
                "保证金占用比例 (%)", 10, 50, 25, 5,
                key="bt_margin_ratio",
                help="每手期权保证金 = 标的价 × 合约乘数 × 比例",
            )
            max_margin_cap_pct = bt_col3.slider(
                "保证金上限（净资产 %）", 50, 100, 80, 5,
                key="bt_max_margin_cap",
                help="总保证金占用不超过净资产的此比例（默认80%）",
            )
            bt_col4, bt_col5 = st.columns(2)
            delta_hedge = bt_col4.checkbox("启用 Delta 对冲（每日买卖ETF）", value=True,
                                           key="bt_delta_hedge")
            hedge_threshold = bt_col5.slider(
                "对冲触发阈值（Delta）", 0.01, 0.30, 0.07, 0.01,
                key="bt_hedge_thresh",
                help="净Delta绝对值超过阈值时才执行ETF对冲（默认0.07）",
            )

        # 资金占用预估（实时显示，不需要点按钮）
        if pairs:
            sample_pair  = pairs[0]
            sample_mult  = int(sample_pair["call_leg"].get("multiplier", 10000))
            margin_ratio    = margin_ratio_pct / 100.0
            max_margin_cap  = max_margin_cap_pct / 100.0
            m_per_lot    = calc_margin_per_lot(spot, sample_mult, margin_ratio)
            qty_est      = calc_max_lots(
                float(initial_capital), spot, sample_mult, margin_ratio,
                n_legs=2, max_margin_cap=max_margin_cap)
            total_margin_est = 2 * m_per_lot * qty_est
            margin_pct_est   = total_margin_est / float(initial_capital) * 100
            free_cap_est     = float(initial_capital) - total_margin_est

            st.info(
                "💡 **资金占用预估**：标的价 {:.3f}，1手保证金 **{:,.0f} 元**（名义本金{:,.0f}×{:.0f}%）；"
                "本金 {:,.0f} 元（上限{:.0f}%）→ 每腿最多建仓 **{} 手**（共 {} 手 call + {} 手 put），"
                "保证金占用 **{:,.0f} 元（{:.1f}%）**，可用资金 **{:,.0f} 元**".format(
                    spot,
                    m_per_lot, spot * sample_mult, margin_ratio_pct,
                    float(initial_capital), max_margin_cap_pct,
                    qty_est, qty_est, qty_est,
                    total_margin_est, margin_pct_est, free_cap_est,
                )
            )

        if st.button("运行回测", type="primary"):
            all_results = []
            debug_msgs  = []
            margin_ratio   = margin_ratio_pct / 100.0
            max_margin_cap = max_margin_cap_pct / 100.0
            for pair in pairs:
                positions = strangle_to_positions(pair)
                # 调试：打印各腿 open_price
                for p in positions:
                    debug_msgs.append(
                        "{} K={} open_price={:.4f} iv={:.2%}".format(
                            p["call_put"], p["strike_price"],
                            p["open_price"], p.get("iv", 0)
                        )
                    )
                df_daily, stats, trade_log = run_backtest(
                    positions, df_price, r=r,
                    delta_hedge=delta_hedge,
                    hedge_threshold=hedge_threshold,
                    initial_capital=float(initial_capital),
                    margin_ratio=margin_ratio,
                    max_margin_cap=max_margin_cap,
                )
                summary = summarize_strangle(pair, spot)
                label = "Call{}/Put{}".format(
                    summary["call_strike"], summary["put_strike"])
                all_results.append({
                    "label":     label,
                    "df_daily":  df_daily,
                    "stats":     stats,
                    "summary":   summary,
                    "positions": positions,
                    "trade_log": trade_log,
                })
            st.session_state["backtest_result"] = all_results
            # 显示调试信息
            with st.expander("调试：各腿开仓价格", expanded=False):
                for msg in debug_msgs:
                    st.text(msg)

    # ── 回测结果可视化 ─────────────────────────────────────────────────────────
    results = st.session_state.get("backtest_result")
    if not results:
        return

    st.markdown("---")
    st.markdown("### 回测结果")

    df_price = st.session_state.get("price_path_df")

    def _fmt_pct(val, plus=False):
        """安全格式化百分比，NaN/None 显示 N/A"""
        try:
            v = float(val)
            if not math.isfinite(v):
                return "N/A"
            return ("{:+.2f}%" if plus else "{:.2f}%").format(v)
        except Exception:
            return "N/A"

    for res in results:
        st.markdown("#### {}".format(res["label"]))
        summary   = res["summary"]
        stats     = res["stats"]
        df_daily  = res["df_daily"]
        trade_log = res.get("trade_log", [])

        # ── 概览：本金 + 权利金 + 最终盈亏 ─────────────────────────────────
        cap_col1, cap_col2, cap_col3 = st.columns(3)
        cap_col1.info("💰 期初本金：**{:,.0f} 元**".format(
            stats.get("initial_capital", 0)))
        cap_col2.info("📥 收取权利金：**{:,.2f} 元**".format(
            stats.get("total_premium", 0)))
        cap_col3.info("📤 最终盈亏：**{:+,.2f} 元**（总计）".format(
            stats.get("final_pnl", 0)))

        cap_col4, cap_col5, cap_col6 = st.columns(3)
        cap_col4.info("📈 期权累计盈亏：**{:+,.2f} 元**".format(
            stats.get("final_opt_pnl", 0)))
        cap_col5.info("🔄 ETF对冲盈亏：**{:+,.2f} 元**".format(
            stats.get("final_etf_pnl", 0)))
        cap_col6.info("📊 浮盈亏（期末）：**{:+,.2f} 元** / **{:+.2f}%**（权利金）/ **{:+.2f}%**（本金）".format(
            stats.get("final_float_pnl", 0),
            stats.get("final_float_pct_prem", 0),
            stats.get("final_float_pct_cap", 0),
        ))

        # ── 资金占用面板 ─────────────────────────────────────────────────────
        with st.expander("💼 资金占用详情", expanded=False):
            mc_a, mc_b, mc_c, mc_d, mc_e, mc_f = st.columns(6)
            mc_a.metric("每手保证金", "{:,.0f} 元".format(stats.get("margin_per_lot", 0)))
            mc_b.metric("每腿建仓手数", "{} 手".format(stats.get("qty_per_leg", 1)))
            mc_c.metric("保证金总占用", "{:,.0f} 元".format(stats.get("total_margin", 0)))
            mc_d.metric("保证金占本金", "{:.1f}%".format(stats.get("margin_pct", 0)))
            mc_e.metric("保证金上限", "{:.0f}%净资产".format(
                stats.get("max_margin_cap", 0.80) * 100))
            mc_f.metric("可用余资金", "{:,.0f} 元".format(stats.get("free_capital", 0)))

        # 绩效指标卡片
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("总收益率",   _fmt_pct(stats.get("total_return_pct", 0), plus=True))
        mc2.metric("年化收益",   _fmt_pct(stats.get("annualized_pct",   0), plus=True))
        mc3.metric("最大回撤",   _fmt_pct(stats.get("max_drawdown_pct", 0)))
        mc4.metric("Sharpe",     "{:.3f}".format(stats.get("sharpe", 0) or 0))
        mc5.metric("胜率（日）", "{:.1f}%".format(stats.get("win_rate_pct", 0) or 0))

        # 诊断：若 final_pnl 为 0，给出提示
        if stats.get("final_pnl", 0) == 0 and not df_daily.empty:
            st.warning(
                "⚠️ 盈亏为 0：可能是价格路径的日期范围（{} ~ {}）"
                "与期权到期日不匹配，或所有日期的 BS 价格变化为零。"
                "请检查期权到期日与价格路径日期是否存在重叠。".format(
                    str(df_daily["date"].iloc[0]),
                    str(df_daily["date"].iloc[-1]),
                )
            )
            with st.expander("调试：逐日盈亏明细（前20行）", expanded=True):
                dcols = ["date", "spot"]
                dcols += [c for c in df_daily.columns if c.startswith("pos_")]
                dcols += ["opt_pnl", "etf_pnl", "total_pnl", "cumulative_pnl",
                          "net_delta", "etf_shares"]
                dcols = [c for c in dcols if c in df_daily.columns]
                st.dataframe(df_daily[dcols].head(20), width='stretch')

        if df_daily.empty or df_price is None:
            continue

        # ── 交易明细表 ────────────────────────────────────────────────────
        with st.expander("📋 交易明细（开仓 / Delta对冲 / 到期结算）", expanded=True):
            if trade_log:
                df_log = pd.DataFrame(trade_log)
                df_log["日期"] = df_log["日期"].astype(str)
                for col in ["金额", "累计盈亏"]:
                    if col in df_log.columns:
                        df_log[col] = df_log[col].apply(
                            lambda x: "{:+,.2f}".format(float(x))
                            if x is not None else "—"
                        )
                display_cols = [c for c in ["日期", "类型", "操作", "方向",
                                             "数量", "价格", "金额", "累计盈亏", "说明"]
                                if c in df_log.columns]
                st.dataframe(df_log[display_cols], width='stretch')
            else:
                st.info("暂无交易记录")

        # ── 三图：价格路径 + 累计盈亏 + Delta 走势 ───────────────────────
        has_delta        = "net_delta"          in df_daily.columns
        has_float        = "float_pnl"          in df_daily.columns
        has_cum_opt      = "cumulative_opt_pnl" in df_daily.columns
        has_cum_etf      = "cumulative_etf_pnl" in df_daily.columns
        n_rows           = 3 if has_delta else 2
        row_h            = [0.38, 0.38, 0.24] if has_delta else [0.52, 0.48]
        subtitles        = (
            ["标的价格路径 + 行权价区间",
             "累计盈亏：总 / 期权 / ETF对冲",
             "净Delta走势（对冲触发阈值 {:.2f}）".format(hedge_threshold)]
            if has_delta else
            ["标的价格路径 + 行权价区间",
             "累计盈亏：总 / 期权 / ETF对冲"]
        )

        fig = make_subplots(
            rows=n_rows, cols=1, shared_xaxes=True,
            row_heights=row_h,
            vertical_spacing=0.05,
            subplot_titles=subtitles,
        )

        # 图1：价格路径
        fig.add_trace(
            go.Scatter(
                x=df_price["date"].astype(str),
                y=df_price["close"],
                mode="lines",
                name="标的价格",
                line={"color": "#1f77b4", "width": 2},
            ), row=1, col=1,
        )

        call_k = summary["call_strike"]
        put_k  = summary["put_strike"]
        ub     = summary["upper_breakeven"]
        lb     = summary["lower_breakeven"]

        for val, color, lbl in [
            (call_k, "#ff7f0e", "Call K={:.3f}".format(call_k)),
            (put_k,  "#2ca02c", "Put K={:.3f}".format(put_k)),
            (ub,     "#ff4444", "上方平衡 {:.3f}".format(ub)),
            (lb,     "#ff4444", "下方平衡 {:.3f}".format(lb)),
        ]:
            fig.add_hline(y=val, line_dash="dash", line_color=color,
                          annotation_text=lbl, row=1, col=1)

        # 图2：累计盈亏（总 + 期权分量 + ETF分量）
        # ── 主曲线：累计总盈亏（含填充）
        pnl_color = "#00cc44" if df_daily["cumulative_pnl"].iloc[-1] >= 0 else "#ff4444"
        fig.add_trace(
            go.Scatter(
                x=df_daily["date"].astype(str),
                y=df_daily["cumulative_pnl"],
                mode="lines", fill="tozeroy",
                name="累计总盈亏",
                line={"color": pnl_color, "width": 2.5},
                fillcolor="rgba(0,204,68,0.12)" if pnl_color == "#00cc44"
                           else "rgba(255,68,68,0.12)",
            ), row=2, col=1,
        )
        # ── 期权累计盈亏
        if has_cum_opt:
            fig.add_trace(
                go.Scatter(
                    x=df_daily["date"].astype(str),
                    y=df_daily["cumulative_opt_pnl"],
                    mode="lines", name="期权累计盈亏",
                    line={"color": "#17becf", "width": 1.8, "dash": "dot"},
                ), row=2, col=1,
            )
        elif "opt_pnl" in df_daily.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_daily["date"].astype(str),
                    y=df_daily["opt_pnl"].cumsum(),
                    mode="lines", name="期权累计盈亏",
                    line={"color": "#17becf", "width": 1.8, "dash": "dot"},
                ), row=2, col=1,
            )
        # ── ETF累计盈亏
        if has_cum_etf:
            fig.add_trace(
                go.Scatter(
                    x=df_daily["date"].astype(str),
                    y=df_daily["cumulative_etf_pnl"],
                    mode="lines", name="ETF对冲累计盈亏",
                    line={"color": "#bcbd22", "width": 1.8, "dash": "dash"},
                ), row=2, col=1,
            )
        elif "etf_pnl" in df_daily.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_daily["date"].astype(str),
                    y=df_daily["etf_pnl"].cumsum(),
                    mode="lines", name="ETF对冲累计盈亏",
                    line={"color": "#bcbd22", "width": 1.8, "dash": "dash"},
                ), row=2, col=1,
            )
        # ── 浮盈亏曲线（权利金衰减）
        if has_float:
            fig.add_trace(
                go.Scatter(
                    x=df_daily["date"].astype(str),
                    y=df_daily["float_pnl"],
                    mode="lines", name="浮盈亏（权利金衰减）",
                    line={"color": "#e377c2", "width": 1.2, "dash": "longdash"},
                    opacity=0.75,
                ), row=2, col=1,
            )

        # 零线
        fig.add_hline(y=0, line_color="#555555", line_dash="solid",
                      line_width=1, row=2, col=1)

        # 图3：净 Delta（Bar，超阈值变色）
        if has_delta:
            bar_colors = []
            for v in df_daily["net_delta"]:
                if abs(v) > hedge_threshold:
                    bar_colors.append("#ff4444" if v > 0 else "#4488ff")
                else:
                    bar_colors.append("#ff7f0e" if v > 0 else "#1f77b4")
            fig.add_trace(
                go.Bar(
                    x=df_daily["date"].astype(str),
                    y=df_daily["net_delta"],
                    name="净Delta",
                    marker_color=bar_colors,
                    opacity=0.85,
                ), row=3, col=1,
            )
            # 对冲阈值线
            fig.add_hline(y=0, line_color="#888888", row=3, col=1)
            fig.add_hline(y=hedge_threshold,  line_color="#ff4444",
                          line_dash="dot", line_width=1,
                          annotation_text="+{:.2f}阈值".format(hedge_threshold),
                          row=3, col=1)
            fig.add_hline(y=-hedge_threshold, line_color="#4488ff",
                          line_dash="dot", line_width=1,
                          annotation_text="-{:.2f}阈值".format(hedge_threshold),
                          row=3, col=1)

        fig.update_layout(
            height=750 if has_delta else 600,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1a1d23",
            font={"color": "#fafafa", "size": 12},
            showlegend=True,
            legend={"bgcolor": "#1a1d23", "orientation": "h",
                    "x": 0, "y": -0.05},
            margin={"t": 60, "b": 60},
        )
        fig.update_xaxes(gridcolor="#2a2d33")
        fig.update_yaxes(gridcolor="#2a2d33")

        st.plotly_chart(fig, width='stretch')

        # ── 逐日明细表（可折叠） ──────────────────────────────────────────
        with st.expander("📊 逐日回测明细", expanded=False):
            daily_show = ["date", "spot", "sigma"]
            daily_show += [c for c in df_daily.columns if c.startswith("mkt_")]
            daily_show += [c for c in df_daily.columns if c.startswith("pos_")]
            daily_show += [
                "opt_pnl", "etf_pnl", "total_pnl",
                "cumulative_opt_pnl", "cumulative_etf_pnl", "cumulative_pnl",
                "float_pnl", "float_pnl_pct", "float_pnl_cap_pct",
                "opt_mkt_val", "net_delta", "etf_shares",
            ]
            daily_show = [c for c in daily_show if c in df_daily.columns]
            fmt_dict   = {c: "{:.4f}" for c in daily_show if c not in ("date",)}
            # 百分比列
            for pct_col in ["float_pnl_pct", "float_pnl_cap_pct"]:
                if pct_col in fmt_dict:
                    fmt_dict[pct_col] = "{:.2f}%"
            # 金额类列保留 2 位小数
            for amt_col in ["opt_pnl", "etf_pnl", "total_pnl",
                            "cumulative_opt_pnl", "cumulative_etf_pnl",
                            "cumulative_pnl", "float_pnl", "opt_mkt_val"]:
                if amt_col in fmt_dict:
                    fmt_dict[amt_col] = "{:.2f}"
            # 数值列 NaN 填 0，防止表格显示 None
            df_show = df_daily[daily_show].copy()
            num_cols = [c for c in daily_show if c != "date"]
            df_show[num_cols] = df_show[num_cols].fillna(0.0)
            st.dataframe(
                df_show.style.format(fmt_dict),
                width='stretch',
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 页面 2：实时监控
# ═══════════════════════════════════════════════════════════════════════════════
def page_monitor():
    st.title("🔴 实时监控")

    positions = st.session_state.get("positions", [])
    spot      = st.session_state.get("spot", 3.0)

    # 尝试自动刷新最新价
    col_spot, col_btn = st.columns([3, 1])
    with col_spot:
        spot_input = st.number_input("当前标的价格", value=float(spot), step=0.001, key="mon_spot")
        st.session_state["spot"] = spot_input
    with col_btn:
        underlying_key = st.session_state.get("underlying", "50ETF (510050)")
        etf_sym = UNDERLYING_MAP.get(underlying_key, {}).get("etf", "510050")
        if st.button("刷新最新价"):
            latest, err = get_latest_spot(etf_sym)
            if err:
                st.warning("自动获取失败：{}，使用手动输入价格".format(err))
            else:
                st.session_state["spot"] = latest
                spot_input = latest
                st.success("已更新最新价：{:.4f}".format(latest))

    render_monitor_page(positions, float(spot_input), r=r)


# ═══════════════════════════════════════════════════════════════════════════════
# 页面 3：持仓管理
# ═══════════════════════════════════════════════════════════════════════════════
def page_positions():
    st.title("📋 持仓管理")

    positions = st.session_state.get("positions", [])

    # ── 当前持仓 ──────────────────────────────────────────────────────────────
    if positions:
        st.markdown("#### 当前持仓（{}条）".format(len(positions)))
        df_pos = pd.DataFrame(positions)
        show_cols = [c for c in ["ts_code", "call_put", "strike_price", "exp_date",
                                  "open_price", "iv", "multiplier", "quantity"]
                     if c in df_pos.columns]
        st.dataframe(df_pos[show_cols], width='stretch')

        # 删除选择
        idx_to_del = st.multiselect("选择要删除的行索引", list(range(len(positions))))
        if st.button("删除选中持仓"):
            st.session_state["positions"] = [
                p for i, p in enumerate(positions) if i not in idx_to_del
            ]
            try:
                st.experimental_rerun()
            except Exception:
                pass

    else:
        st.info("当前无持仓。可通过以下方式添加：\n1. 在「数据导入」页上传 Excel\n2. 在「回测模拟」页构建后加入持仓")

    st.markdown("---")

    # ── 手动添加单条持仓 ──────────────────────────────────────────────────────
    st.markdown("#### 手动添加持仓")
    with st.form("add_position_form"):
        fc1, fc2, fc3 = st.columns(3)
        ts_code      = fc1.text_input("合约代码", value="510050C3200")
        call_put     = fc2.selectbox("类型", ["C", "P"])
        strike_price = fc3.number_input("行权价", value=3.20, step=0.01)
        fd1, fd2, fd3 = st.columns(3)
        today = datetime.date.today()
        exp_date     = fd1.date_input("到期日", value=today + datetime.timedelta(days=30))
        open_price   = fd2.number_input("开仓价", value=0.030, step=0.001, format="%.4f")
        iv           = fd3.number_input("IV（小数）", value=0.20, step=0.01, format="%.3f")
        ff1, ff2 = st.columns(2)
        multiplier   = ff1.number_input("合约乘数", value=10000, step=1000)
        quantity     = ff2.number_input("数量（手）", value=1, step=1)
        submitted = st.form_submit_button("添加持仓")

    if submitted:
        new_pos = {
            "ts_code":      ts_code,
            "call_put":     call_put,
            "strike_price": float(strike_price),
            "exp_date":     exp_date,
            "open_price":   float(open_price),
            "iv":           float(iv),
            "multiplier":   int(multiplier),
            "quantity":     int(quantity),
            "direction":    "short",
        }
        st.session_state["positions"].append(new_pos)
        st.success("已添加：{}".format(ts_code))

    st.markdown("---")

    # ── 保存/加载持仓快照 ──────────────────────────────────────────────────────
    st.markdown("#### 持仓快照")
    save_col, load_col = st.columns(2)
    data_path = st.session_state.get("data_path", r"D:\auto_tc\data_sync")

    with save_col:
        if st.button("保存持仓到 Excel"):
            if not positions:
                st.warning("无持仓数据")
            else:
                path = os.path.join(data_path, "positions_snapshot.xlsx")
                err = save_positions(pd.DataFrame(positions), path)
                if err:
                    st.error("保存失败：{}".format(err))
                else:
                    st.success("已保存至 {}".format(path))

    with load_col:
        up_file = st.file_uploader("加载持仓快照 Excel", type=["xlsx"], key="load_pos")
        if up_file:
            df_loaded, err = load_positions(up_file)
            if err:
                st.error("加载失败：{}".format(err))
            else:
                st.session_state["positions"] = df_loaded.to_dict("records")
                st.success("已加载 {} 条持仓".format(len(df_loaded)))


# ═══════════════════════════════════════════════════════════════════════════════
# 页面 4：数据导入
# ═══════════════════════════════════════════════════════════════════════════════
def page_data():
    st.title("📥 数据导入")

    tab_manual, tab_auto = st.tabs(["手动上传", "自动拉取（akshare）"])

    # ── 手动上传 ──────────────────────────────────────────────────────────────
    with tab_manual:
        st.markdown("#### 上传期权列表 Excel")
        tmpl_df  = build_template_df()

        # 下载模板按钮
        import io
        buf = io.BytesIO()
        tmpl_df.to_excel(buf, index=False)
        buf.seek(0)
        st.download_button(
            "下载 Excel 模板",
            data=buf,
            file_name="option_input_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        opt_file = st.file_uploader("上传期权列表（.xlsx）", type=["xlsx"], key="opt_upload")
        if opt_file:
            df, err = load_options_from_excel(opt_file)
            if err:
                st.error(err)
            else:
                st.session_state["options_df"] = df
                st.success("解析成功，共 {} 条期权".format(len(df)))
                st.dataframe(df, width='stretch')

        st.markdown("---")
        st.markdown("#### 上传标的价格路径 Excel")
        st.caption("必须包含 date（日期）和 close（收盘价）两列")
        price_file = st.file_uploader("上传价格路径（.xlsx）", type=["xlsx"], key="price_upload")
        if price_file:
            df_price, err = load_price_path_from_excel(price_file)
            if err:
                st.error(err)
            else:
                st.session_state["price_path_df"] = df_price
                if not df_price.empty:
                    st.session_state["spot"] = float(df_price["close"].iloc[-1])
                st.success("价格路径加载成功，共 {} 天".format(len(df_price)))
                st.dataframe(df_price.tail(10), width='stretch')

        st.markdown("---")
        st.markdown("#### 持仓快照加入回测持仓")
        if st.session_state.get("options_df") is not None:
            df_opt = st.session_state["options_df"]
            if st.button("将期权列表全部设为卖出持仓"):
                positions = df_opt.to_dict("records")
                for p in positions:
                    p["direction"] = "short"
                st.session_state["positions"] = positions
                st.success("已设置 {} 条持仓".format(len(positions)))

    # ── 自动拉取 ──────────────────────────────────────────────────────────────
    with tab_auto:
        st.markdown("#### akshare 自动拉取")
        underlying = st.selectbox("选择标的", list(UNDERLYING_MAP.keys()),
                                  key="data_underlying")
        st.session_state["underlying"] = underlying
        meta = UNDERLYING_MAP[underlying]

        data_path = st.session_state.get("data_path", r"D:\auto_tc\data_sync")
        strategy_name = meta["name"]   # 用标的中文名作为策略目录名

        # ── 期权链拉取 ────────────────────────────────────────────────────────
        if st.button("拉取期权链", type="primary"):
            with st.spinner("正在拉取期权链..."):
                df_chain, err = load_option_chain_from_akshare(meta["opt"], meta["etf"])
            if err:
                st.error("期权链拉取失败：{}".format(err))
            else:
                st.session_state["options_df"] = df_chain
                st.success("期权链拉取成功，共 {} 条".format(len(df_chain)))
                # ── 自动保存 ──────────────────────────────────────────────────
                saved_path, save_err = save_fetched_data(
                    df_chain, data_path, strategy_name, "option_chain"
                )
                if save_err:
                    st.warning("保存失败：{}".format(save_err))
                else:
                    st.caption("✅ 已保存至 {}".format(saved_path))
                st.dataframe(df_chain.head(20), width='stretch')

        # ── 日线行情拉取 ──────────────────────────────────────────────────────
        days_options = {
            "30天": 30, "60天": 60, "90天": 90,
            "180天": 180, "1年": 365, "2年": 730,
        }
        days_label = st.selectbox(
            "拉取日线天数", list(days_options.keys()), index=1,
            key="pull_days_select",
        )
        pull_days = days_options[days_label]

        if st.button("拉取日线行情", type="secondary"):
            with st.spinner("正在拉取 {} 日线行情...".format(days_label)):
                df_spot, err = load_spot_from_akshare(meta["etf"], days=pull_days)
            if err:
                st.error("行情拉取失败：{}".format(err))
            else:
                st.session_state["price_path_df"] = df_spot
                if not df_spot.empty:
                    st.session_state["spot"] = float(df_spot["close"].iloc[-1])
                st.success("拉取 {} 条行情（{}），最新价 {:.4f}".format(
                    len(df_spot), days_label, st.session_state["spot"]))
                # ── 自动保存（含天数参数） ─────────────────────────────────────
                saved_path, save_err = save_fetched_data(
                    df_spot, data_path, strategy_name, "spot",
                    params={"days": pull_days},
                )
                if save_err:
                    st.warning("保存失败：{}".format(save_err))
                else:
                    st.caption("✅ 已保存至 {}".format(saved_path))
                # 价格走势小图
                fig_spot = go.Figure(go.Scatter(
                    x=df_spot["date"].astype(str),
                    y=df_spot["close"],
                    mode="lines",
                    line={"color": "#1f77b4"},
                ))
                fig_spot.update_layout(
                    height=250,
                    title="{} 近期走势".format(meta["name"]),
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#1a1d23",
                    font={"color": "#fafafa"},
                    margin={"t": 40, "b": 20},
                )
                st.plotly_chart(fig_spot, width='stretch')

        # ── 历史保存文件列表 ──────────────────────────────────────────────────
        with st.expander("📂 已保存的历史数据文件（最新10条）", expanded=False):
            saved_files = list_saved_files(data_path, strategy_name)
            if saved_files:
                for f in saved_files[:10]:
                    fname = os.path.basename(f)
                    st.text(fname)
                    # 提供加载按钮
                    load_key = "load_{}".format(fname)
                    if st.button("载入 {}".format(fname), key=load_key):
                        import pandas as _pd
                        try:
                            _df = _pd.read_csv(f, encoding="utf-8-sig")
                            # 判断是 spot 还是 option_chain
                            if "close" in _df.columns:
                                if "date" in _df.columns:
                                    _df["date"] = _pd.to_datetime(
                                        _df["date"], errors="coerce").dt.date
                                st.session_state["price_path_df"] = _df
                                if not _df.empty:
                                    st.session_state["spot"] = float(_df["close"].iloc[-1])
                                st.success("已载入行情数据，共 {} 条".format(len(_df)))
                            elif "strike_price" in _df.columns:
                                if "exp_date" in _df.columns:
                                    _df["exp_date"] = _pd.to_datetime(
                                        _df["exp_date"], errors="coerce").dt.date
                                st.session_state["options_df"] = _df
                                st.success("已载入期权链数据，共 {} 条".format(len(_df)))
                        except Exception as _e:
                            st.error("载入失败：{}".format(_e))
            else:
                st.info("暂无保存的数据文件（数据路径：{}）".format(
                    os.path.join(data_path, strategy_name)))


# ═══════════════════════════════════════════════════════════════════════════════
# 页面路由
# ═══════════════════════════════════════════════════════════════════════════════
if page == "回测模拟":
    page_backtest()
elif page == "实时监控":
    page_monitor()
elif page == "持仓管理":
    page_positions()
elif page == "数据导入":
    page_data()
