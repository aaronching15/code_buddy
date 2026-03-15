"""
CTA策略回测 - Streamlit 管理界面
运行方式:
    cd cta_strategy
    streamlit run app.py

兼容 Streamlit >= 1.20
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from datetime import datetime, date

# ---- 兼容层：抹平版本差异 ----
def _rerun():
    """兼容 st.rerun() / st.experimental_rerun()"""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

def _divider():
    """兼容 st.divider()（1.25+ 才有）"""
    if hasattr(st, "divider"):
        st.divider()
    else:
        st.markdown("---")

# ---- 页面配置 ----
st.set_page_config(
    page_title="CTA策略回测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- 引入核心模块 ----
try:
    from cta_strategy import (
        CTAStrategy, load_market_data, validate_data,
        REQUIRED_COLUMNS, COLUMN_MAPPING,
        DualMovingAverageStrategy, TurtleStrategy,
        BollingerBandStrategy, MACDStrategy, RSIStrategy
    )
    ENGINE_LOADED = True
except ImportError as e:
    ENGINE_LOADED = False
    ENGINE_ERROR = str(e)

# ==================== 样式 ====================
st.markdown("""
<style>
.metric-positive { color: #27ae60; font-weight: bold; }
.metric-negative { color: #e74c3c; font-weight: bold; }
.info-card {
    background: #f0f4ff;
    border-left: 4px solid #1f77b4;
    padding: 12px 16px;
    border-radius: 4px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ==================== Session State 初始化 ====================
def init_state():
    defaults = {
        "step": 1,
        "loaded_data": {},
        "backtest_result": None,
        "config": {
            "initial_capital": 1_000_000,
            "commission_rate": 0.0003,
            "slippage": 0.0001,
            "position_method": "fixed_fraction",
            "position_fraction": 0.1,
            "max_positions": 5,
            "strategies": ["dual_ma", "turtle", "bollinger", "macd", "rsi"],
            "start_date": None,
            "end_date": None,
        },
        "data_warnings": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ==================== 侧边栏导航 ====================
with st.sidebar:
    st.title("📈 CTA 回测系统")
    _divider()

    steps = {
        1: ("📂", "数据准备"),
        2: ("⚙️", "参数配置"),
        3: ("▶️", "运行回测"),
        4: ("📊", "结果分析"),
    }

    for num, (icon, label) in steps.items():
        is_active = st.session_state.step == num
        if st.button(
            f"{icon}  步骤 {num}：{label}",
            key=f"nav_{num}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.step = num
            _rerun()

    _divider()
    n_loaded = len(st.session_state.loaded_data)
    if n_loaded:
        st.success(f"已加载 {n_loaded} 个标的")
        for sym in st.session_state.loaded_data:
            st.caption(f"• {sym}")
    else:
        st.warning("尚未加载数据")

    if st.session_state.backtest_result:
        st.success("回测已完成 ✓")

    _divider()
    st.caption("CTA Strategy Backtester v1.0")

# ==================== 引擎检查 ====================
if not ENGINE_LOADED:
    st.error(f"引擎加载失败：{ENGINE_ERROR}")
    st.info("请确保 `cta_strategy.py` 与 `app.py` 在同一目录下，且已安装依赖：`pip install -r requirements.txt`")
    st.stop()

# ==================== 步骤路由 ====================

# ──────────────────────────────────────────────────────
# 步骤 1：数据准备
# ──────────────────────────────────────────────────────
if st.session_state.step == 1:
    st.header("📂 步骤 1：数据准备")
    st.caption("上传行情数据文件，或使用演示数据快速体验系统")

    with st.expander("📋 数据格式要求（点击展开）", expanded=True):
        st.markdown("""
### 必须包含的列

| 列名（中/英均可）| 说明 | 格式要求 |
|---|---|---|
| `日期` / `date` | 交易日期 | `YYYY-MM-DD` 或 `YYYYMMDD` |
| `代码` / `symbol` | 标的代码 | 字符串，如 `IF9999`、`000001` |
| `开盘` / `open` | 开盘价 | 数值，单位：元 |
| `最高` / `high` | 最高价 | 数值，单位：元 |
| `最低` / `low` | 最低价 | 数值，单位：元 |
| `收盘` / `close` | 收盘价 | 数值，单位：元 |
| `成交量` / `volume` | 成交量 | 数值，单位：手或股 |

> **可选列**：`成交额 / amount`、`换手率 / turnover`
> 以上中文列名会被自动识别并映射，**无需手动重命名**。

---
### 支持的文件格式

| 格式 | 说明 |
|---|---|
| `.csv` | UTF-8 或 GBK 编码均可 |
| `.xlsx` / `.xls` | Excel 文件，数据在第一个 Sheet |

---
### 数据示例（CSV 格式）

```
日期,代码,开盘,最高,最低,收盘,成交量
2022-01-04,IF9999,4820.0,4856.0,4798.0,4835.0,35421
2022-01-05,IF9999,4835.0,4870.0,4810.0,4851.0,28934
```

---
### 各类资产获取数据的常见渠道

| 资产类型 | 建议数据来源 |
|---|---|
| 股票指数期货 (IF/IC/IH/IM) | 通达信、文华财经导出；或 akshare `ak.futures_main_sina()` |
| A股股票 | tushare `ts.pro_bar()`；或 baostock |
| ETF | 同花顺、Choice 导出；或 akshare `ak.fund_etf_hist_em()` |
| 商品期货 | 文华财经、CTP 历史数据；或 akshare `ak.futures_main_sina()` |
        """)

    _divider()

    col_upload, col_demo = st.columns([3, 2])

    with col_upload:
        st.subheader("上传数据文件")
        uploaded_files = st.file_uploader(
            "支持 .csv / .xlsx，可同时上传多个标的",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            key="file_uploader"
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                default_symbol = os.path.splitext(uploaded_file.name)[0].upper()
                col_sym, col_btn = st.columns([2, 1])
                with col_sym:
                    symbol = st.text_input(
                        f"标的代码（{uploaded_file.name}）",
                        value=default_symbol,
                        key=f"sym_{uploaded_file.name}"
                    )
                with col_btn:
                    st.write("")
                    if st.button("加载", key=f"load_{uploaded_file.name}", type="primary"):
                        try:
                            content = uploaded_file.read()
                            if uploaded_file.name.endswith(".csv"):
                                try:
                                    raw_df = pd.read_csv(io.BytesIO(content), encoding='utf-8')
                                except UnicodeDecodeError:
                                    raw_df = pd.read_csv(io.BytesIO(content), encoding='gbk')
                            else:
                                raw_df = pd.read_excel(io.BytesIO(content))

                            raw_df = raw_df.rename(columns=COLUMN_MAPPING)
                            raw_df.columns = [c.lower().strip() for c in raw_df.columns]
                            if symbol and 'symbol' not in raw_df.columns:
                                raw_df['symbol'] = symbol
                            if 'date' in raw_df.columns:
                                raw_df['date'] = pd.to_datetime(raw_df['date'], errors='coerce')
                                raw_df = raw_df.dropna(subset=['date'])
                            # 若无 volume 但有 amount_m（成交额百万），用成交额/收盘价近似换算成交量
                            if 'volume' not in raw_df.columns and 'amount_m' in raw_df.columns:
                                for col in ['close', 'amount_m']:
                                    raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
                                raw_df['volume'] = (
                                    raw_df['amount_m'] * 1_000_000 / raw_df['close'].replace(0, np.nan)
                                ).fillna(0)
                                st.info("ℹ️ 检测到「成交额(百万)」列，已自动换算为近似成交量（成交额×100万/收盘价）")
                            for col in ['open', 'high', 'low', 'close', 'volume']:
                                if col in raw_df.columns:
                                    raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
                            raw_df = raw_df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)

                            ok, issues = validate_data(raw_df)
                            if not ok:
                                st.warning(f"数据质量问题：{'; '.join(issues)}")
                            else:
                                st.session_state.loaded_data[symbol] = raw_df
                                st.success(f"✅ {symbol} 加载成功，共 {len(raw_df)} 条记录")
                        except Exception as e:
                            st.error(f"加载失败：{e}")

    with col_demo:
        st.subheader("使用演示数据")
        st.info("没有数据文件？点击下方按钮生成随机演示数据，可直接体验完整回测流程。")

        demo_asset = st.selectbox(
            "演示标的",
            ["IF9999（股指期货）", "IC9999（中证500期货）", "RB9999（螺纹钢期货）"],
            key="demo_asset"
        )
        demo_n = st.slider("模拟交易天数", 200, 1000, 500, 50, key="demo_n")
        demo_seed = st.number_input("随机种子", value=42, key="demo_seed")

        if st.button("生成演示数据", type="primary", use_container_width=True):
            symbol_map = {
                "IF9999（股指期货）": ("IF9999", 4000),
                "IC9999（中证500期货）": ("IC9999", 6000),
                "RB9999（螺纹钢期货）": ("RB9999", 4200),
            }
            sym, base_price = symbol_map[demo_asset]
            np.random.seed(int(demo_seed))
            n = demo_n
            price = base_price + np.cumsum(np.random.randn(n) * base_price * 0.008)
            price = np.maximum(price, base_price * 0.3)
            demo_df = pd.DataFrame({
                'date': pd.date_range('2022-01-01', periods=n, freq='B'),
                'symbol': sym,
                'open': price * (1 + np.random.randn(n) * 0.002),
                'high': price * (1 + np.abs(np.random.randn(n)) * 0.006),
                'low': price * (1 - np.abs(np.random.randn(n)) * 0.006),
                'close': price,
                'volume': np.random.randint(10000, 80000, n).astype(float),
            })
            st.session_state.loaded_data[sym] = demo_df
            st.success(f"✅ 演示数据已生成：{sym}，共 {n} 条")
            _rerun()

    if st.session_state.loaded_data:
        _divider()
        st.subheader("已加载数据预览")

        tabs = st.tabs(list(st.session_state.loaded_data.keys()))
        for tab, (sym, df) in zip(tabs, st.session_state.loaded_data.items()):
            with tab:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("记录条数", f"{len(df):,}")
                col2.metric("开始日期", str(df['date'].min())[:10])
                col3.metric("结束日期", str(df['date'].max())[:10])
                col4.metric("收盘价范围", f"{df['close'].min():.1f} ~ {df['close'].max():.1f}")
                st.dataframe(df.tail(10), use_container_width=True, height=250)

                col_del, _ = st.columns([1, 4])
                with col_del:
                    if st.button(f"删除 {sym}", key=f"del_{sym}"):
                        del st.session_state.loaded_data[sym]
                        _rerun()

        _divider()
        if st.button("下一步：参数配置 →", type="primary"):
            st.session_state.step = 2
            _rerun()


# ──────────────────────────────────────────────────────
# 步骤 2：参数配置
# ──────────────────────────────────────────────────────
elif st.session_state.step == 2:
    st.header("⚙️ 步骤 2：参数配置")
    st.caption("配置回测资金、手续费、仓位管理方式及策略组合")

    if not st.session_state.loaded_data:
        st.warning("请先在「步骤 1」中加载数据")
        if st.button("← 返回数据准备"):
            st.session_state.step = 1
            _rerun()
        st.stop()

    cfg = st.session_state.config

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("💰 资金与成本设置")
        cfg["initial_capital"] = st.number_input(
            "初始资金（元）", min_value=100_000, max_value=100_000_000,
            value=int(cfg["initial_capital"]), step=100_000
        )
        cfg["commission_rate"] = st.number_input(
            "手续费率（双边）", min_value=0.0, max_value=0.01,
            value=float(cfg["commission_rate"]), format="%.4f", step=0.0001,
            help="期货单边约 0.0003，股票约 0.001"
        )
        cfg["slippage"] = st.number_input(
            "滑点（比例）", min_value=0.0, max_value=0.005,
            value=float(cfg["slippage"]), format="%.4f", step=0.0001
        )

        st.subheader("📅 回测时间范围")
        all_dates = []
        for df in st.session_state.loaded_data.values():
            all_dates.extend(df['date'].tolist())
        if all_dates:
            min_date = pd.Timestamp(min(all_dates)).date()
            max_date = pd.Timestamp(max(all_dates)).date()
        else:
            min_date = date(2020, 1, 1)
            max_date = date.today()

        start_d = st.date_input("开始日期", value=min_date, min_value=min_date, max_value=max_date)
        end_d = st.date_input("结束日期", value=max_date, min_value=min_date, max_value=max_date)
        cfg["start_date"] = str(start_d)
        cfg["end_date"] = str(end_d)

    with col_b:
        st.subheader("📐 仓位管理")
        position_methods = {
            "固定比例（Fixed Fraction）": "fixed_fraction",
            "等权重（Equal Weight）": "equal_weight",
            "ATR波动率仓位": "atr_based",
        }
        method_label = st.selectbox(
            "仓位计算方式",
            list(position_methods.keys()),
            index=list(position_methods.values()).index(cfg["position_method"])
        )
        cfg["position_method"] = position_methods[method_label]

        cfg["position_fraction"] = st.slider(
            "单笔资金比例",
            min_value=0.02, max_value=0.5,
            value=float(cfg["position_fraction"]), step=0.01,
            help="每笔开仓占总资金的比例，0.1 = 10%"
        )
        cfg["max_positions"] = st.slider(
            "最大同时持仓标的数",
            min_value=1, max_value=20,
            value=int(cfg["max_positions"]), step=1
        )

        st.subheader("🧠 启用策略")
        strategy_options = {
            "dual_ma": "双均线（Dual MA）",
            "turtle": "海龟交易（Turtle）",
            "bollinger": "布林带（Bollinger Band）",
            "macd": "MACD",
            "rsi": "RSI",
        }
        selected = []
        for key, label in strategy_options.items():
            if st.checkbox(label, value=(key in cfg["strategies"]), key=f"strat_{key}"):
                selected.append(key)
        cfg["strategies"] = selected if selected else list(strategy_options.keys())
        if not selected:
            st.warning("至少选择一个策略，已自动全选")

    _divider()
    st.subheader("📋 配置摘要")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("初始资金", f"{cfg['initial_capital']:,.0f} 元")
    col2.metric("手续费率", f"{cfg['commission_rate']:.4f}")
    col3.metric("单笔仓位", f"{cfg['position_fraction']:.0%}")
    col4.metric("启用策略数", f"{len(cfg['strategies'])} 个")

    st.session_state.config = cfg

    col_prev, col_next = st.columns([1, 1])
    with col_prev:
        if st.button("← 返回数据准备"):
            st.session_state.step = 1
            _rerun()
    with col_next:
        if st.button("下一步：运行回测 →", type="primary"):
            st.session_state.step = 3
            _rerun()


# ──────────────────────────────────────────────────────
# 步骤 3：运行回测
# ──────────────────────────────────────────────────────
elif st.session_state.step == 3:
    st.header("▶️ 步骤 3：运行回测")

    if not st.session_state.loaded_data:
        st.warning("请先完成「步骤 1」数据准备")
        if st.button("← 去数据准备"):
            st.session_state.step = 1
            _rerun()
        st.stop()

    cfg = st.session_state.config

    with st.expander("当前配置", expanded=False):
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("初始资金", f"{cfg['initial_capital']:,.0f}")
        col2.metric("手续费", f"{cfg['commission_rate']:.4f}")
        col3.metric("仓位", f"{cfg['position_fraction']:.0%}")
        col4.metric("最大持仓", f"{cfg['max_positions']}")
        col5.metric("策略数", f"{len(cfg['strategies'])}")

    _divider()

    if st.button("🚀 开始回测", type="primary", use_container_width=True):
        with st.spinner("回测运行中，请稍候..."):
            try:
                cta = CTAStrategy(cfg)
                progress_bar = st.progress(0)
                progress_bar.progress(10)

                result = cta.run(
                    st.session_state.loaded_data,
                    start_date=cfg.get("start_date"),
                    end_date=cfg.get("end_date")
                )

                progress_bar.progress(100)
                st.session_state.backtest_result = result
                st.success("✅ 回测完成！")

            except Exception as e:
                st.error(f"回测出错：{e}")
                import traceback
                st.code(traceback.format_exc())

    if st.session_state.backtest_result:
        r = st.session_state.backtest_result
        st.subheader("快速预览")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("总收益率", f"{r['total_return']:.2%}", delta=f"{r['total_return']:.2%}")
        c2.metric("年化收益率", f"{r['annual_return']:.2%}")
        c3.metric("最大回撤", f"{r['max_drawdown']:.2%}")
        c4.metric("夏普比率", f"{r['sharpe_ratio']:.2f}")
        c5.metric("胜率", f"{r['win_rate']:.2%}")

        if st.button("查看详细结果 →", type="primary"):
            st.session_state.step = 4
            _rerun()

    col_prev, _ = st.columns([1, 3])
    with col_prev:
        if st.button("← 返回参数配置"):
            st.session_state.step = 2
            _rerun()


# ──────────────────────────────────────────────────────
# 步骤 4：结果分析
# ──────────────────────────────────────────────────────
elif st.session_state.step == 4:
    st.header("📊 步骤 4：结果分析")

    if not st.session_state.backtest_result:
        st.warning("尚无回测结果，请先运行回测")
        if st.button("← 去运行回测"):
            st.session_state.step = 3
            _rerun()
        st.stop()

    r = st.session_state.backtest_result

    # ---- 核心指标 ----
    st.subheader("📌 核心绩效指标")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("初始资金", f"¥{r['initial_capital']:,.0f}")
    col2.metric("最终资金", f"¥{r['final_capital']:,.0f}", delta=f"{r['total_return']:+.2%}")
    col3.metric("年化收益率", f"{r['annual_return']:.2%}")
    col4.metric("最大回撤", f"{r['max_drawdown']:.2%}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("夏普比率", f"{r['sharpe_ratio']:.2f}")
    col6.metric("Calmar 比率", f"{r['calmar_ratio']:.2f}")
    col7.metric("总交易次数", f"{r['total_trades']}")
    col8.metric("胜率", f"{r['win_rate']:.2%}")

    col9, col10, _, _ = st.columns(4)
    col9.metric("平均盈利", f"¥{r['avg_win']:,.0f}")
    col10.metric("平均亏损", f"¥{r['avg_loss']:,.0f}")

    _divider()

    # ---- 净值曲线 ----
    st.subheader("📈 净值曲线")
    if r['equity_curve']:
        equity_df = pd.DataFrame(r['equity_curve'], columns=['date', 'equity'])
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df = equity_df.sort_values('date').reset_index(drop=True)
        equity_df['net_value'] = equity_df['equity'] / r['initial_capital']
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']

        tab_equity, tab_dd = st.tabs(["净值曲线", "回撤曲线"])

        with tab_equity:
            nv = equity_df['net_value']
            nv_min = float(nv.min())
            nv_max = float(nv.max())
            y_min = nv_min * (1 - 0.01)   # 最小值再减 1%
            y_max = nv_max * (1 + 0.01)   # 最大值再加 1%

            try:
                import altair as alt
                equity_chart_df = equity_df[['date', 'net_value']].copy()
                equity_chart = (
                    alt.Chart(equity_chart_df)
                    .mark_line(color="#1f77b4", strokeWidth=1.8)
                    .encode(
                        x=alt.X('date:T', title='日期'),
                        y=alt.Y('net_value:Q', title='净值',
                                scale=alt.Scale(domain=[y_min, y_max])),
                        tooltip=[
                            alt.Tooltip('date:T', title='日期', format='%Y-%m-%d'),
                            alt.Tooltip('net_value:Q', title='净值', format='.4f'),
                        ]
                    )
                    .properties(height=350)
                )
                st.altair_chart(equity_chart, use_container_width=True)
            except ImportError:
                # altair 未安装，退回 st.line_chart（无法精确控制 Y 轴范围）
                st.line_chart(
                    equity_df.set_index('date')['net_value'],
                    use_container_width=True,
                    height=350,
                )
            st.caption(
                f"起止日期：{equity_df['date'].min().date()} ～ {equity_df['date'].max().date()}"
                f"　｜　净值区间：[{nv_min:.4f}, {nv_max:.4f}]"
            )

        with tab_dd:
            st.area_chart(
                equity_df.set_index('date')['drawdown'],
                use_container_width=True,
                height=300,
            )
            if r.get('max_dd_start') and r.get('max_dd_end'):
                st.caption(
                    f"最大回撤期间：{str(r['max_dd_start'])[:10]} ～ {str(r['max_dd_end'])[:10]}"
                    f"，最大回撤 {r['max_drawdown']:.2%}"
                )
    else:
        st.info("净值曲线数据为空（可能回测期间无交易）")

    _divider()

    # ---- 交易记录 ----
    st.subheader("📋 交易记录")
    trades = r.get('trades', [])
    if trades:
        trades_data = []
        for t in trades:
            trades_data.append({
                "开仓日期": str(t.open_date)[:10],
                "平仓日期": str(t.close_date)[:10],
                "标的": t.symbol,
                "方向": "多" if t.direction.value == 1 else "空",
                "开仓价": f"{t.entry_price:.2f}",
                "平仓价": f"{t.exit_price:.2f}",
                "数量": f"{t.volume:.2f}",
                "手续费": f"{t.commission:.2f}",
                "净盈亏": f"{t.pnl:+.2f}",
            })
        trades_df = pd.DataFrame(trades_data)

        # 盈亏着色：正数 → 红底黑字；负数 → 黑底红字
        def highlight_pnl(val):
            try:
                v = float(val)
                if v > 0:
                    return "background-color: #e53935; color: #000000; font-weight: bold"
                elif v < 0:
                    return "background-color: #000000; color: #e53935; font-weight: bold"
                else:
                    return ""
            except Exception:
                return ""

        st.dataframe(
            trades_df.style.applymap(highlight_pnl, subset=["净盈亏"]),
            use_container_width=True,
            height=400
        )

        csv_buf = trades_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            "⬇️ 下载交易记录 CSV",
            data=csv_buf,
            file_name=f"cta_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        # ---- 盈亏分布 ----
        st.subheader("📊 盈亏分布")
        pnl_series = pd.Series([t.pnl for t in trades], name="净盈亏")
        col_hist, col_stats = st.columns([3, 2])

        with col_hist:
            try:
                import altair as alt
                hist_df = pd.DataFrame({"pnl": pnl_series})
                chart = (
                    alt.Chart(hist_df)
                    .mark_bar(opacity=0.8)
                    .encode(
                        x=alt.X("pnl:Q", bin=alt.Bin(maxbins=30), title="净盈亏"),
                        y=alt.Y("count()", title="频次"),
                        color=alt.condition(
                            alt.datum.pnl > 0,
                            alt.value("#27ae60"),
                            alt.value("#e74c3c")
                        )
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart, use_container_width=True)
            except ImportError:
                st.bar_chart(pnl_series.value_counts(bins=20).sort_index())

        with col_stats:
            wins = pnl_series[pnl_series > 0]
            losses = pnl_series[pnl_series <= 0]
            ratio = abs(wins.mean() / losses.mean()) if len(losses) > 0 and losses.mean() != 0 else float('inf')
            st.markdown(f"""
| 统计项 | 数值 |
|---|---|
| 盈利笔数 | {len(wins)} |
| 亏损笔数 | {len(losses)} |
| 最大单笔盈利 | `{pnl_series.max():+.2f}` |
| 最大单笔亏损 | `{pnl_series.min():+.2f}` |
| 平均盈利 | `{wins.mean():+.2f}` |
| 平均亏损 | `{losses.mean():+.2f}` |
| 盈亏比 | `{ratio:.2f}` |
            """)
    else:
        st.info("回测期间无任何已平仓交易记录")

    # ---- 月度盈亏 ----
    _divider()
    st.subheader("📅 月度盈亏")
    if r['equity_curve']:
        _equity_df = pd.DataFrame(r['equity_curve'], columns=['date', 'equity'])
        _equity_df['date'] = pd.to_datetime(_equity_df['date'])
        _equity_df = _equity_df.sort_values('date').reset_index(drop=True)

        # 每月取最后一个交易日的权益
        _equity_df['month'] = _equity_df['date'].dt.to_period('M')
        monthly_last = _equity_df.groupby('month')['equity'].last()

        # 月度收益 = 当月末权益 / 上月末权益 - 1
        monthly_ret = monthly_last.pct_change()
        # 第一个月用 初始资金 作为基准
        monthly_ret.iloc[0] = monthly_last.iloc[0] / r['initial_capital'] - 1

        monthly_df = pd.DataFrame({
            '月份': [str(m) for m in monthly_ret.index],
            '月末净值': monthly_last.values,
            '月度收益率': monthly_ret.values,
            '月度盈亏(元)': monthly_last.diff().fillna(monthly_last.iloc[0] - r['initial_capital']).values,
        })

        # 着色：正收益 → 红底黑字（A股风格），负收益 → 绿底黑字，零 → 无色
        def color_monthly(val):
            try:
                v = float(val)
                if v > 0:
                    return "background-color: #e53935; color: #000000; font-weight: bold"
                elif v < 0:
                    return "background-color: #1b5e20; color: #ffffff; font-weight: bold"
                else:
                    return ""
            except Exception:
                return ""

        display_df = monthly_df.copy()
        display_df['月度收益率'] = display_df['月度收益率'].map(lambda x: f"{x:+.2%}")
        display_df['月末净值']   = display_df['月末净值'].map(lambda x: f"{x:,.2f}")
        display_df['月度盈亏(元)'] = display_df['月度盈亏(元)'].map(lambda x: f"{x:+,.2f}")

        st.dataframe(
            display_df.style.applymap(
                color_monthly, subset=["月度收益率", "月度盈亏(元)"]
            ),
            use_container_width=True,
            height=min(40 * (len(display_df) + 2), 500),
        )

        # 柱状图：月度收益率
        try:
            import altair as alt
            bar_df = pd.DataFrame({
                'month': [str(m) for m in monthly_ret.index],
                'return': monthly_ret.values,
            })
            bar_chart = (
                alt.Chart(bar_df)
                .mark_bar()
                .encode(
                    x=alt.X('month:N', title='月份', sort=None,
                             axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y('return:Q', title='月度收益率',
                             axis=alt.Axis(format='.1%')),
                    color=alt.condition(
                        alt.datum['return'] > 0,
                        alt.value('#e53935'),
                        alt.value('#1b5e20')
                    ),
                    tooltip=[
                        alt.Tooltip('month:N', title='月份'),
                        alt.Tooltip('return:Q', title='月度收益率', format='+.2%'),
                    ]
                )
                .properties(height=260)
            )
            st.altair_chart(bar_chart, use_container_width=True)
        except ImportError:
            pass
    else:
        st.info("净值曲线数据为空，无法计算月度盈亏")

    _divider()
    col_prev, col_rerun = st.columns([1, 1])
    with col_prev:
        if st.button("← 重新配置参数"):
            st.session_state.step = 2
            _rerun()
    with col_rerun:
        if st.button("🔄 重新运行回测", type="secondary"):
            st.session_state.backtest_result = None
            st.session_state.step = 3
            _rerun()
