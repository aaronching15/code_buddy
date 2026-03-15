# -*- coding: utf-8 -*-
"""
app.py
Strangle 卖出对冲策略研究与监控系统 - Streamlit 主入口
4 页面：回测模拟 / 实时监控 / 持仓管理 / 数据导入
兼容 Python 3.7 + Streamlit 1.23.1
"""

import datetime
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from bs_engine import bs_price_and_greeks, implied_vol
from backtest_engine import run_backtest, batch_backtest
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

        if st.button("运行回测", type="primary"):
            all_results = []
            for pair in pairs:
                positions = strangle_to_positions(pair)
                df_daily, stats = run_backtest(positions, df_price, r=r)
                summary = summarize_strangle(pair, spot)
                label = "Call{}/Put{}".format(
                    summary["call_strike"], summary["put_strike"])
                all_results.append({
                    "label":    label,
                    "df_daily": df_daily,
                    "stats":    stats,
                    "summary":  summary,
                    "positions": positions,
                })
            st.session_state["backtest_result"] = all_results

    # ── 回测结果可视化 ─────────────────────────────────────────────────────────
    results = st.session_state.get("backtest_result")
    if not results:
        return

    st.markdown("---")
    st.markdown("### 回测结果")

    df_price = st.session_state.get("price_path_df")

    for res in results:
        st.markdown("#### {}".format(res["label"]))
        summary  = res["summary"]
        stats    = res["stats"]
        df_daily = res["df_daily"]

        # 绩效指标卡片
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("总收益率",    "{:+.2f}%".format(stats.get("total_return_pct", 0)))
        mc2.metric("年化收益",    "{:+.2f}%".format(stats.get("annualized_pct", 0)))
        mc3.metric("最大回撤",    "{:.2f}%".format(stats.get("max_drawdown_pct", 0)))
        mc4.metric("Sharpe",      "{:.3f}".format(stats.get("sharpe", 0)))
        mc5.metric("胜率（日）",  "{:.1f}%".format(stats.get("win_rate_pct", 0)))

        if df_daily.empty or df_price is None:
            continue

        # ── 双图：价格路径 + 盈亏曲线 ─────────────────────────────────────────
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.55, 0.45],
            vertical_spacing=0.06,
            subplot_titles=("标的价格路径 + 行权价区间", "逐日累计盈亏"),
        )

        # 价格路径
        fig.add_trace(
            go.Scatter(
                x=df_price["date"].astype(str),
                y=df_price["close"],
                mode="lines",
                name="标的价格",
                line={"color": "#1f77b4", "width": 2},
            ), row=1, col=1,
        )
        # 行权价水平线
        call_k = summary["call_strike"]
        put_k  = summary["put_strike"]
        ub     = summary["upper_breakeven"]
        lb     = summary["lower_breakeven"]

        for val, color, label in [
            (call_k, "#ff7f0e", "Call K={:.3f}".format(call_k)),
            (put_k,  "#2ca02c", "Put K={:.3f}".format(put_k)),
            (ub,     "#ff4444", "上方盈亏平衡 {:.3f}".format(ub)),
            (lb,     "#ff4444", "下方盈亏平衡 {:.3f}".format(lb)),
        ]:
            fig.add_hline(y=val, line_dash="dash", line_color=color,
                          annotation_text=label, row=1, col=1)

        # 累计盈亏曲线
        pnl_color = "#00cc44" if df_daily["cumulative_pnl"].iloc[-1] >= 0 else "#ff4444"
        fig.add_trace(
            go.Scatter(
                x=df_daily["date"].astype(str),
                y=df_daily["cumulative_pnl"],
                mode="lines",
                fill="tozeroy",
                name="累计盈亏",
                line={"color": pnl_color, "width": 2},
                fillcolor="rgba(0,204,68,0.15)" if pnl_color == "#00cc44"
                           else "rgba(255,68,68,0.15)",
            ), row=2, col=1,
        )

        fig.update_layout(
            height=600,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1a1d23",
            font={"color": "#fafafa", "size": 12},
            showlegend=True,
            legend={"bgcolor": "#1a1d23"},
            margin={"t": 60, "b": 40},
        )
        fig.update_xaxes(gridcolor="#2a2d33")
        fig.update_yaxes(gridcolor="#2a2d33")

        st.plotly_chart(fig, width='stretch')


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

        if st.button("拉取期权链", type="primary"):
            with st.spinner("正在拉取期权链..."):
                df_chain, err = load_option_chain_from_akshare(meta["opt"], meta["etf"])
            if err:
                st.error("期权链拉取失败：{}".format(err))
            else:
                st.session_state["options_df"] = df_chain
                st.success("期权链拉取成功，共 {} 条".format(len(df_chain)))
                st.dataframe(df_chain.head(20), width='stretch')

        if st.button("拉取日线行情", type="secondary"):
            with st.spinner("正在拉取日线行情..."):
                df_spot, err = load_spot_from_akshare(meta["etf"], days=60)
            if err:
                st.error("行情拉取失败：{}".format(err))
            else:
                st.session_state["price_path_df"] = df_spot
                if not df_spot.empty:
                    st.session_state["spot"] = float(df_spot["close"].iloc[-1])
                st.success("拉取 {} 条行情，最新价 {:.4f}".format(
                    len(df_spot), st.session_state["spot"]))
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
