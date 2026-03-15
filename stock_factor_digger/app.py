"""
app.py  —  Stock Factor Digger
app.py — 界面

步骤1：上传数据 / 生成演示数据
步骤2：启动辩论，逐轮气泡展示提案者/批判者/仲裁者的发言，实时查看评分
步骤3：IC序列图、分层收益柱图、因子值分布图
步骤4：因子库汇总对比、多维雷达图、IC序列叠加对比、CSV/JSON 导出

==============================
基于 Multi-Agent Debate (MAD) 框架的股票因子研究平台

运行方式：
    cd stock_factor_digger
    streamlit run app.py

步骤：
  1. 数据准备  —— 上传行情数据 / 生成演示数据
  2. MAD 辩论  —— 配置 LLM，启动因子辩论，实时查看辩论过程
  3. 因子评估  —— 对辩论产出的因子进行 IC / 分层收益 / 多空回测
  4. 因子库    —— 管理所有已评估的因子，对比筛选
"""

import io
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────
# 兼容层
# ──────────────────────────────────────────────────────────────────

def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

def _divider():
    if hasattr(st, "divider"):
        st.divider()
    else:
        st.markdown("---")

# ──────────────────────────────────────────────────────────────────
# 页面配置
# ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Stock Factor Digger",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# 引入核心模块
# ──────────────────────────────────────────────────────────────────

try:
    from factor_engine import FactorEngine, FactorResult
    from mad_agents import FactorMAD, DebateResult, FactorProposal

    ENGINE_LOADED = True
except ImportError as e:
    ENGINE_LOADED = False
    ENGINE_ERROR = str(e)

# ──────────────────────────────────────────────────────────────────
# 样式
# ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* 辩论气泡 */
.bubble-proposer {
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    border-left: 4px solid #1976D2;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.bubble-critic {
    background: linear-gradient(135deg, #fce4ec, #f8bbd0);
    border-left: 4px solid #c62828;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.bubble-mediator {
    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    border-left: 4px solid #2e7d32;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.score-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: bold;
    margin: 2px;
}
.score-good  { background: #c8e6c9; color: #1b5e20; }
.score-mid   { background: #fff9c4; color: #f57f17; }
.score-bad   { background: #ffcdd2; color: #b71c1c; }
.adopted-tag   { background: #c8e6c9; color: #1b5e20; padding: 4px 14px; border-radius: 14px; font-weight:bold; }
.rejected-tag  { background: #ffcdd2; color: #b71c1c; padding: 4px 14px; border-radius: 14px; font-weight:bold; }
.factor-card {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
    background: #fafafa;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "step": 1,
        "price_df": None,
        "engine": None,
        "llm_config": {
            "api_key": "",
            "base_url": "",
            "model": "gpt-4o-mini",
            "max_rounds": 3,
            "temperature": 0.7,
        },
        "debate_results": [],      # List[DebateResult]
        "factor_results": {},      # name -> FactorResult
        "factor_library": [],      # 已采纳因子 name 列表
        "mad": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ──────────────────────────────────────────────────────────────────
# 侧边栏导航
# ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 Factor Digger")
    st.caption("基于 Multi-Agent Debate 的因子挖掘平台")
    _divider()

    steps = {
        1: ("📂", "数据准备"),
        2: ("🤖", "MAD 辩论"),
        3: ("📊", "因子评估"),
        4: ("📚", "因子库"),
    }
    for num, (icon, label) in steps.items():
        is_active = st.session_state.step == num
        if st.button(
            f"{icon}  {num}. {label}",
            key=f"nav_{num}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.step = num
            _rerun()

    _divider()
    # 状态摘要
    if st.session_state.price_df is not None:
        df = st.session_state.price_df
        st.success(f"✅ 数据已加载：{df['symbol'].nunique()} 只股票 × {df['date'].nunique()} 个交易日")
    else:
        st.warning("⚠ 尚未加载行情数据")

    if st.session_state.debate_results:
        adopted = sum(1 for r in st.session_state.debate_results if r.adopted)
        st.info(f"💬 已辩论 {len(st.session_state.debate_results)} 个主题，采纳 {adopted} 个因子")

    if st.session_state.factor_library:
        st.success(f"📚 因子库：{len(st.session_state.factor_library)} 个")

    _divider()
    st.caption("Stock Factor Digger v1.0")

# ──────────────────────────────────────────────────────────────────
# 引擎检查
# ──────────────────────────────────────────────────────────────────

if not ENGINE_LOADED:
    st.error(f"模块加载失败：{ENGINE_ERROR}")
    st.info("请确保 `factor_engine.py` 和 `mad_agents.py` 与 `app.py` 在同一目录")
    st.stop()


# ══════════════════════════════════════════════════════════════════
# 步骤 1：数据准备
# ══════════════════════════════════════════════════════════════════

if st.session_state.step == 1:
    st.header("📂 步骤 1：数据准备")
    st.caption("上传多只股票的日行情数据，或使用内置演示数据")

    with st.expander("📋 数据格式要求", expanded=False):
        st.markdown("""
| 列名（中/英均可） | 说明 |
|---|---|
| `日期` / `date` | 交易日期 |
| `代码` / `symbol` | 股票代码 |
| `开盘(价)` / `open` | 开盘价 |
| `最高(价)` / `high` | 最高价 |
| `最低(价)` / `low` | 最低价 |
| `收盘(价)` / `close` | 收盘价 |
| `成交量` / `volume` | 成交量 |

> 支持 `.csv`（UTF-8/GBK） 和 `.xlsx`；单文件可包含多只股票（long format）。
        """)

    col_upload, col_demo = st.columns([3, 2])

    COLUMN_MAP = {
        "日期": "date", "时间": "date",
        "代码": "symbol", "股票代码": "symbol",
        "开盘": "open",  "开盘价": "open",  "开盘价(元)": "open",
        "最高": "high",  "最高价": "high",  "最高价(元)": "high",
        "最低": "low",   "最低价": "low",   "最低价(元)": "low",
        "收盘": "close", "收盘价": "close", "收盘价(元)": "close",
        "成交量": "volume", "成交量(手)": "volume",
        "成交量(股)": "volume", "成交量(张)": "volume",
        "名称": "name", "股票名称": "name",
        "涨跌幅": "change_pct",
    }

    with col_upload:
        st.subheader("上传文件")
        f = st.file_uploader("支持 .csv / .xlsx", type=["csv", "xlsx", "xls"])
        if f:
            try:
                content = f.read()
                if f.name.endswith(".csv"):
                    try:
                        raw = pd.read_csv(io.BytesIO(content), encoding="utf-8")
                    except UnicodeDecodeError:
                        raw = pd.read_csv(io.BytesIO(content), encoding="gbk")
                else:
                    raw = pd.read_excel(io.BytesIO(content))

                raw = raw.rename(columns=COLUMN_MAP)
                raw.columns = [c.lower().strip() for c in raw.columns]
                raw["date"]   = pd.to_datetime(raw["date"], errors="coerce")
                raw["close"]  = pd.to_numeric(raw.get("close", np.nan), errors="coerce")
                raw["volume"] = pd.to_numeric(raw.get("volume", 0),     errors="coerce")
                for col in ["open", "high", "low"]:
                    if col in raw.columns:
                        raw[col] = pd.to_numeric(raw[col], errors="coerce")
                raw = raw.dropna(subset=["date", "close"]).sort_values(["symbol", "date"]).reset_index(drop=True)

                st.session_state.price_df = raw
                st.session_state.engine   = FactorEngine(raw)
                st.success(f"✅ 加载成功：{raw['symbol'].nunique()} 只股票，{len(raw):,} 条记录")
            except Exception as e:
                st.error(f"加载失败：{e}")

    with col_demo:
        st.subheader("演示数据")
        n_stocks = st.slider("股票数量", 10, 100, 30, 10)
        n_days   = st.slider("交易日数", 200, 1000, 500, 100)
        seed     = st.number_input("随机种子", value=42)
        if st.button("🎲 生成演示数据", type="primary", use_container_width=True):
            with st.spinner("生成中..."):
                demo_df = FactorEngine.generate_demo_data(n_stocks, n_days, int(seed))
            st.session_state.price_df = demo_df
            st.session_state.engine   = FactorEngine(demo_df)
            st.success(f"✅ 演示数据就绪：{n_stocks} 只股票 × {n_days} 个交易日")
            _rerun()

    if st.session_state.price_df is not None:
        _divider()
        df = st.session_state.price_df
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("股票数量", df["symbol"].nunique())
        col2.metric("交易日数", df["date"].nunique())
        col3.metric("起始日期", str(df["date"].min())[:10])
        col4.metric("截止日期", str(df["date"].max())[:10])

        st.dataframe(df.head(20), use_container_width=True, height=280)

        if st.button("下一步：MAD 辩论 →", type="primary"):
            st.session_state.step = 2
            _rerun()


# ══════════════════════════════════════════════════════════════════
# 步骤 2：MAD 辩论
# ══════════════════════════════════════════════════════════════════

elif st.session_state.step == 2:
    st.header("🤖 步骤 2：Multi-Agent Debate 因子辩论")
    st.caption("提案者生成因子假设 → 批判者多维度质疑 → 迭代修订 → 仲裁者裁定")

    if st.session_state.price_df is None:
        st.warning("请先在「步骤 1」中准备数据")
        if st.button("← 去数据准备"):
            st.session_state.step = 1; _rerun()
        st.stop()

    # ---- LLM 配置 ----
    with st.expander("⚙️ LLM 配置（不填则使用 Demo 模式）", expanded=False):
        cfg = st.session_state.llm_config
        col_a, col_b = st.columns(2)
        with col_a:
            cfg["api_key"]   = st.text_input("API Key", value=cfg["api_key"], type="password",
                                              help="留空则进入 Demo 模式（规则引擎模拟辩论，无需 API）")
            cfg["base_url"]  = st.text_input("Base URL（可选）", value=cfg["base_url"],
                                              placeholder="https://api.openai.com/v1")
        with col_b:
            cfg["model"]       = st.text_input("模型名称", value=cfg["model"])
            cfg["max_rounds"]  = st.slider("最大辩论轮数", 1, 5, cfg["max_rounds"])
            cfg["temperature"] = st.slider("Temperature", 0.0, 1.0, cfg["temperature"], 0.05)
        st.session_state.llm_config = cfg

        is_demo = not bool(cfg["api_key"])
        if is_demo:
            st.info("🎮 **Demo 模式**：未配置 API Key，将使用内置规则引擎模拟辩论过程，效果真实，适合体验系统。")
        else:
            st.success(f"✅ LLM 模式：{cfg['model']}")

    _divider()

    # ---- 辩论主题输入 ----
    st.subheader("📌 设置辩论主题")

    tab_preset, tab_custom = st.tabs(["📋 预设主题", "✏️ 自定义主题"])

    with tab_preset:
        demo_topics = FactorMAD.get_demo_topics()
        preset_labels = {
            "量价背离（成交量与价格方向相反）": "量价背离",
            "短期反转（超卖反弹效应）": "短期反转",
            "动量加速（1月动量 > 3月动量）": "动量加速",
            "散户情绪结构（隔夜vs日内收益差异）": "散户情绪结构",
            "成交量异常与反转": "成交量异常反转",
        }
        chosen_label = st.selectbox("选择预设主题", list(preset_labels.keys()))
        chosen_topic = preset_labels[chosen_label]
        preset_context = st.text_area("补充研究背景（可选）", height=80,
                                       placeholder="例如：聚焦 A 股 2020-2024 年，排除 ST 股票...")

    with tab_custom:
        custom_topic   = st.text_input("自定义主题", placeholder="例如：资金流入强度因子")
        custom_context = st.text_area("研究背景描述", height=120,
                                       placeholder="描述你的研究思路、参考文献、预期逻辑等...")

    use_custom = bool(custom_topic)
    final_topic   = custom_topic   if use_custom else chosen_topic
    final_context = custom_context if use_custom else preset_context

    col_run, col_batch = st.columns([2, 1])
    with col_run:
        run_single = st.button(
            f"🚀 启动辩论：{final_topic}", type="primary", use_container_width=True
        )
    with col_batch:
        run_batch = st.button(
            "🔁 批量辩论全部预设主题", use_container_width=True,
            help="依次对所有预设主题发起辩论"
        )

    # ---- 执行辩论 ----
    def _build_mad():
        c = st.session_state.llm_config
        return FactorMAD(
            api_key=c["api_key"],
            base_url=c["base_url"],
            model=c["model"],
            max_rounds=c["max_rounds"],
            temperature=c["temperature"],
        )

    def _run_one_debate(topic, context=""):
        mad = _build_mad()
        engine = st.session_state.engine

        def eval_summary_fn(expr: str) -> str:
            """快速评估因子并返回摘要字符串"""
            try:
                panel = engine.compute_expression(expr, "tmp")
                res   = engine.evaluate(panel, factor_name="tmp")
                return (
                    f"IC均值={res.ic_mean:.3f}, IR={res.ir:.2f}, "
                    f"多空年化={res.long_short_return:.1%}, 覆盖率={res.coverage:.0%}"
                )
            except Exception:
                return ""

        result = mad.run_debate(topic, context, eval_summary_fn)
        st.session_state.debate_results.append(result)
        return result

    if run_single:
        with st.spinner(f"辩论进行中：{final_topic} ..."):
            result = _run_one_debate(final_topic, final_context)
        st.success(f"✅ 辩论完成！最终裁定：{'✅ 采纳' if result.adopted else '❌ 不采纳'}")

    if run_batch:
        progress = st.progress(0)
        all_topics = list(preset_labels.values())
        for i, tp in enumerate(all_topics):
            with st.spinner(f"[{i+1}/{len(all_topics)}] 辩论：{tp}"):
                _run_one_debate(tp)
            progress.progress((i + 1) / len(all_topics))
        st.success(f"✅ 批量辩论完成，共 {len(all_topics)} 个主题")

    # ---- 辩论结果展示 ----
    if st.session_state.debate_results:
        _divider()
        st.subheader(f"📜 辩论记录（共 {len(st.session_state.debate_results)} 场）")

        # 选择查看哪一场
        result_labels = [
            f"[{'✅采纳' if r.adopted else '❌不采纳'}] {r.topic}（{r.total_rounds}轮）"
            for r in st.session_state.debate_results
        ]
        selected_idx = st.selectbox("选择查看的辩论", range(len(result_labels)),
                                     format_func=lambda i: result_labels[i],
                                     index=len(result_labels) - 1)

        dr: DebateResult = st.session_state.debate_results[selected_idx]

        # 顶部摘要
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("辩论主题", dr.topic)
        col2.metric("辩论轮数", dr.total_rounds)
        col3.metric("综合评分", f"{dr.final_opinion.overall_score:.1f}/10")
        col4.metric("最终裁定",
                    "✅ 采纳" if dr.adopted else "❌ 不采纳",
                    delta=None)

        # 逐轮展示
        for rnd in dr.rounds:
            with st.expander(
                f"第 {rnd.round_num} 轮辩论  |  评分 {rnd.critic_opinion.overall_score:.1f}  |  "
                f"{'✅ 批准' if rnd.critic_opinion.approve else '🔁 需修订'}",
                expanded=(rnd.round_num == dr.total_rounds),
            ):
                col_p, col_c = st.columns(2)

                # 提案者
                with col_p:
                    st.markdown(f"""
<div class="bubble-proposer">
<b>💡 提案者（Proposer）</b><br>
<b>因子名：</b> <code>{rnd.proposal.name}</code><br>
<b>经济学逻辑：</b> {rnd.proposal.economic_logic}<br>
<b>计算表达式：</b><br><code>{rnd.proposal.expression}</code><br>
<b>预期方向：</b> {rnd.proposal.expected_direction}<br>
<b>风险提示：</b> {rnd.proposal.risk_notes}
</div>
""", unsafe_allow_html=True)

                # 批判者
                with col_c:
                    op = rnd.critic_opinion
                    def _badge(v, reverse=False):
                        if reverse:
                            cls = "score-good" if v <= 4 else "score-mid" if v <= 6 else "score-bad"
                        else:
                            cls = "score-good" if v >= 7 else "score-mid" if v >= 5 else "score-bad"
                        return f'<span class="score-badge {cls}">{v}</span>'

                    st.markdown(f"""
<div class="bubble-critic">
<b>🔍 批判者（Critic）</b><br>
逻辑有效性 {_badge(op.validity_score)} &nbsp;
可测试性 {_badge(op.testability_score)} &nbsp;
新颖性 {_badge(op.novelty_score)} &nbsp;
挖掘风险 {_badge(op.risk_score, reverse=True)}<br>
<b>综合评分：</b> <b>{op.overall_score:.1f}</b> / 10<br>
<b>优点：</b> {"；".join(op.strengths) if op.strengths else "—"}<br>
<b>弱点：</b> {"；".join(op.weaknesses) if op.weaknesses else "—"}<br>
<b>建议：</b> {"；".join(op.suggestions) if op.suggestions else "—"}<br>
{"<b>建议表达式：</b> <code>" + op.revised_expression + "</code>" if op.revised_expression else ""}
</div>
""", unsafe_allow_html=True)

                if rnd.proposer_response:
                    st.markdown(f"""
<div style="background:#f3e5f5;border-left:4px solid #7b1fa2;border-radius:6px;padding:10px 14px;margin-top:8px">
<b>🗣 提案者回应：</b> {rnd.proposer_response}
</div>
""", unsafe_allow_html=True)

        # 仲裁结论
        _divider()
        st.markdown(f"""
<div class="bubble-mediator">
<b>⚖️ 仲裁者（Mediator）结论</b><br><br>
{dr.mediator_summary.replace(chr(10), '<br>')}
</div>
""", unsafe_allow_html=True)

        # 操作按钮
        col_eval, col_del = st.columns([2, 1])
        with col_eval:
            if dr.adopted and dr.final_proposal.expression:
                if st.button(f"📊 评估因子「{dr.final_proposal.name}」", type="primary"):
                    st.session_state.step = 3
                    st.session_state["pending_eval"] = dr
                    _rerun()
        with col_del:
            if st.button("🗑 清空所有辩论记录", type="secondary"):
                st.session_state.debate_results = []
                _rerun()

        _divider()
        if st.button("下一步：因子评估 →", type="primary"):
            st.session_state.step = 3
            _rerun()


# ══════════════════════════════════════════════════════════════════
# 步骤 3：因子评估
# ══════════════════════════════════════════════════════════════════

elif st.session_state.step == 3:
    st.header("📊 步骤 3：因子评估")
    st.caption("对辩论产出的因子（或任意内置因子）进行完整的量化评估")

    if st.session_state.engine is None:
        st.warning("请先在「步骤 1」中准备数据")
        if st.button("← 去数据准备"):
            st.session_state.step = 1; _rerun()
        st.stop()

    engine: FactorEngine = st.session_state.engine

    # ---- 选择评估源 ----
    st.subheader("🎯 选择评估对象")
    source = st.radio(
        "因子来源",
        ["🤖 MAD 辩论产出", "📚 内置因子库", "✏️ 自定义表达式"],
        horizontal=True,
    )

    eval_name  = ""
    eval_expr  = ""
    eval_logic = ""

    if source == "🤖 MAD 辩论产出":
        adopted_debates = [r for r in st.session_state.debate_results
                           if r.adopted and r.final_proposal.expression]

        # 检查是否有来自步骤2的跳转
        pending = st.session_state.pop("pending_eval", None)
        if pending and pending not in adopted_debates:
            adopted_debates.insert(0, pending)

        if not adopted_debates:
            st.info("暂无已采纳的辩论因子，请先在「步骤 2」完成辩论，或选择其他因子来源")
        else:
            options = [f"{r.final_proposal.name}（{r.topic}）" for r in adopted_debates]
            chosen  = st.selectbox("选择已采纳因子", range(len(options)),
                                    format_func=lambda i: options[i])
            dr = adopted_debates[chosen]
            eval_name  = dr.final_proposal.name
            eval_expr  = dr.final_proposal.expression
            eval_logic = dr.final_proposal.economic_logic

            st.markdown(f"""
<div class="bubble-proposer">
<b>因子名：</b> <code>{eval_name}</code><br>
<b>经济逻辑：</b> {eval_logic}<br>
<b>计算表达式：</b> <code>{eval_expr}</code><br>
<b>辩论轮数：</b> {dr.total_rounds}  &nbsp; <b>综合评分：</b> {dr.final_opinion.overall_score:.1f}/10
</div>
""", unsafe_allow_html=True)

    elif source == "📚 内置因子库":
        builtins = FactorEngine.list_builtins()
        chosen_builtin = st.selectbox("选择内置因子", builtins)
        eval_name = chosen_builtin
        # 内置因子不用表达式
        eval_expr = f"__builtin__{chosen_builtin}"

        builtin_desc = {
            "momentum_1m": "1个月动量（20日收益率），趋势跟踪",
            "momentum_3m": "3个月动量（60日收益率）",
            "momentum_6m": "6个月动量（跳过最近1月）",
            "reversal_1w": "短期反转（5日收益率取负）",
            "volatility_20": "20日波动率",
            "volatility_60": "60日波动率",
            "volume_ma5_ratio": "量比（日量/5日均量）",
            "volume_ma20_ratio": "20日量比",
            "turnover_rate_proxy": "换手率代理（量/60日均量）",
            "price_to_high_52w": "52周价格高点强度",
            "price_to_low_52w": "52周低点距离",
            "rsi_14": "RSI(14)技术指标",
            "macd_signal": "MACD 柱状值（动量加速）",
            "bollinger_position": "布林带相对位置（0-1）",
            "atr_ratio": "ATR比率（相对波动率）",
            "trend_strength": "趋势强度（60日线性回归斜率）",
            "price_acceleration": "价格加速度（5日-20日动量差）",
            "volume_price_corr": "量价相关系数（20日）",
            "high_low_ratio": "振幅（日内博弈强度）",
            "open_gap": "隔夜跳空幅度",
            "intraday_return": "日内收益（收盘/开盘-1）",
            "volume_trend": "成交量趋势（5日/20日均量比）",
        }
        st.info(f"📖 {builtin_desc.get(chosen_builtin, '')}")

    else:  # 自定义
        eval_name = st.text_input("因子名称", value="my_factor")
        eval_expr = st.text_area(
            "因子表达式",
            height=100,
            placeholder="例如：rank(ret(5)) - rank(ma(volume,5)/ma(volume,20))",
            help="可用变量：close / open / high / low / volume\n"
                 "可用函数：ret(n) / ma(x,n) / ema(x,n) / std(x,n) / rank(x) / "
                 "ts_rank(x,n) / corr(x,y,n) / delta(x,n) / delay(x,n) / log(x) / abs(x) / sign(x)",
        )
        eval_logic = st.text_area("经济学逻辑描述（可选）", height=80)

    # ---- 评估参数 ----
    _divider()
    st.subheader("⚙️ 评估参数")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        forward_days = st.selectbox("预测周期", [5, 10, 20, 40, 60], index=2,
                                     help="评估因子对未来 N 日收益的预测能力")
    with col_p2:
        n_layers = st.slider("分层数量", 3, 10, 5)
    with col_p3:
        winsorize_pct = st.slider("去极值（%）", 0, 10, 3,
                                   help="截断因子值两端极值，提升稳定性")

    if st.button("🚀 开始评估", type="primary", use_container_width=True,
                 disabled=not eval_name):
        with st.spinner(f"计算因子 [{eval_name}] 中..."):
            try:
                # 计算因子面板
                if eval_expr.startswith("__builtin__"):
                    bname = eval_expr.replace("__builtin__", "")
                    panel = engine.compute_builtin(bname)
                else:
                    panel = engine.compute_expression(eval_expr, eval_name)

                # 去极值
                if winsorize_pct > 0:
                    lo = panel.stack().quantile(winsorize_pct / 100)
                    hi = panel.stack().quantile(1 - winsorize_pct / 100)
                    panel = panel.clip(lo, hi)

                # 评估
                result = engine.evaluate(
                    panel,
                    forward_days=forward_days,
                    n_layers=n_layers,
                    factor_name=eval_name,
                )
                result.description  = eval_logic or ""
                result.expression   = eval_expr
                result.economic_logic = eval_logic or ""

                st.session_state.factor_results[eval_name] = result
                if eval_name not in st.session_state.factor_library:
                    st.session_state.factor_library.append(eval_name)

                st.success(f"✅ 因子评估完成！IC均值={result.ic_mean:.4f}，IR={result.ir:.2f}")

            except Exception as e:
                import traceback
                st.error(f"评估出错：{e}")
                st.code(traceback.format_exc())

    # ---- 展示评估结果 ----
    available = [n for n in st.session_state.factor_library
                 if n in st.session_state.factor_results]
    if available:
        _divider()
        view_name = st.selectbox("查看评估结果", available, index=len(available)-1)
        res: FactorResult = st.session_state.factor_results[view_name]

        # 核心指标
        st.subheader(f"📌 [{view_name}] 核心绩效")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("IC 均值",      f"{res.ic_mean:.4f}",
                  delta="有效" if abs(res.ic_mean) > 0.02 else "偏弱")
        c2.metric("IR",           f"{res.ir:.2f}",
                  delta="优" if abs(res.ir) > 0.5 else None)
        c3.metric("IC>0 占比",    f"{res.ic_positive_ratio:.1%}")
        c4.metric("多空年化收益", f"{res.long_short_return:.1%}")
        c5.metric("多空夏普",     f"{res.sharpe:.2f}")

        c6, c7, c8, _ = st.columns(4)
        c6.metric("最大回撤",    f"{res.max_drawdown:.1%}")
        c7.metric("月均换手率",  f"{res.turnover:.1%}")
        c8.metric("因子覆盖率",  f"{res.coverage:.1%}")

        # IC 曲线
        if not res.ic_series.empty:
            _divider()
            tab_ic, tab_layer, tab_autocorr = st.tabs(["IC 序列", "分层收益", "因子分布"])

            with tab_ic:
                try:
                    import altair as alt
                    ic_df = res.ic_series.reset_index()
                    ic_df.columns = ["date", "IC"]
                    ic_df["cum_IC"] = ic_df["IC"].cumsum()

                    base = alt.Chart(ic_df)
                    bar = (
                        base.mark_bar(opacity=0.6)
                        .encode(
                            x=alt.X("date:T", title="日期"),
                            y=alt.Y("IC:Q", title="IC"),
                            color=alt.condition(
                                alt.datum.IC > 0,
                                alt.value("#e53935"),
                                alt.value("#1b5e20"),
                            ),
                            tooltip=[
                                alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"),
                                alt.Tooltip("IC:Q", format=".4f"),
                            ]
                        )
                        .properties(height=220)
                    )
                    line = (
                        base.mark_line(color="#1976D2", strokeWidth=2)
                        .encode(
                            x="date:T",
                            y=alt.Y("cum_IC:Q", title="累积IC"),
                        )
                        .properties(height=220)
                    )
                    st.altair_chart(bar, use_container_width=True)
                    st.caption("↑ IC 序列（柱）  |  蓝线=累积IC")
                    st.altair_chart(line, use_container_width=True)
                except ImportError:
                    st.line_chart(res.ic_series, height=250)

            with tab_layer:
                if not res.layer_returns.empty:
                    try:
                        import altair as alt
                        lr = res.layer_returns.reset_index()
                        lr.columns = ["分层", "平均收益", "收益标准差"]
                        lr["分层"] = lr["分层"].astype(str)

                        chart = (
                            alt.Chart(lr)
                            .mark_bar()
                            .encode(
                                x=alt.X("分层:N", title="因子分层（1=最低，5=最高）"),
                                y=alt.Y("平均收益:Q", title="平均收益率",
                                        axis=alt.Axis(format=".2%")),
                                color=alt.condition(
                                    alt.datum["平均收益"] > 0,
                                    alt.value("#e53935"),
                                    alt.value("#1b5e20"),
                                ),
                                tooltip=[
                                    alt.Tooltip("分层:N"),
                                    alt.Tooltip("平均收益:Q", format="+.3%"),
                                    alt.Tooltip("收益标准差:Q", format=".3%"),
                                ]
                            )
                            .properties(height=280)
                        )
                        st.altair_chart(chart, use_container_width=True)
                        st.caption("分层越高表示因子值越大；若单调递增则因子方向正确")
                        st.dataframe(res.layer_returns.style.format("{:.4%}"),
                                     use_container_width=True)
                    except ImportError:
                        st.dataframe(res.layer_returns)
                else:
                    st.info("分层收益数据不足（可能数据量太少）")

            with tab_autocorr:
                col_ac, col_cov = st.columns(2)
                col_ac.metric("因子一阶自相关", f"{res.autocorr:.3f}",
                               help="自相关越高，因子越稳定但换手率越低")
                col_cov.metric("截面覆盖率", f"{res.coverage:.1%}")

                # 因子值截面分布（最新一期）
                latest = res.raw_factor.iloc[-1].dropna()
                if len(latest) > 0:
                    try:
                        import altair as alt
                        hist_df = pd.DataFrame({"value": latest.values})
                        hist = (
                            alt.Chart(hist_df)
                            .mark_bar(opacity=0.75, color="#1976D2")
                            .encode(
                                x=alt.X("value:Q", bin=alt.Bin(maxbins=30), title="因子值"),
                                y=alt.Y("count()", title="股票数"),
                            )
                            .properties(height=220, title="最新期因子值分布")
                        )
                        st.altair_chart(hist, use_container_width=True)
                    except ImportError:
                        st.bar_chart(pd.cut(latest, bins=20).value_counts().sort_index())

        if st.button("→ 进入因子库管理", type="primary"):
            st.session_state.step = 4
            _rerun()


# ══════════════════════════════════════════════════════════════════
# 步骤 4：因子库
# ══════════════════════════════════════════════════════════════════

elif st.session_state.step == 4:
    st.header("📚 步骤 4：因子库管理")
    st.caption("查看、筛选、对比所有已评估的因子，导出研究报告")

    available = [n for n in st.session_state.factor_library
                 if n in st.session_state.factor_results]

    if not available:
        st.info("因子库为空，请先在「步骤 3」评估因子")
        if st.button("← 去因子评估"):
            st.session_state.step = 3; _rerun()
        st.stop()

    # ---- 汇总表 ----
    st.subheader(f"📋 因子汇总（共 {len(available)} 个）")

    summary_rows = []
    for name in available:
        r: FactorResult = st.session_state.factor_results[name]
        summary_rows.append({
            "因子名称":    name,
            "经济逻辑":    r.economic_logic[:40] + "..." if len(r.economic_logic) > 40 else r.economic_logic,
            "IC均值":     round(r.ic_mean, 4),
            "IR":         round(r.ir, 2),
            "IC>0占比":   f"{r.ic_positive_ratio:.0%}",
            "多空年化":   f"{r.long_short_return:.1%}",
            "多空夏普":   round(r.sharpe, 2),
            "最大回撤":   f"{r.max_drawdown:.1%}",
            "换手率":     f"{r.turnover:.1%}",
            "覆盖率":     f"{r.coverage:.0%}",
            "自相关":     round(r.autocorr, 3),
        })

    summary_df = pd.DataFrame(summary_rows)

    def _color_ic(val):
        try:
            v = float(val)
            if abs(v) > 0.04:  return "background-color:#c8e6c9;color:#1b5e20;font-weight:bold"
            if abs(v) > 0.02:  return "background-color:#fff9c4;color:#f57f17"
            return "background-color:#ffcdd2;color:#b71c1c"
        except Exception:
            return ""

    def _color_ir(val):
        try:
            v = float(val)
            if v > 0.8:   return "background-color:#c8e6c9;color:#1b5e20;font-weight:bold"
            if v > 0.4:   return "background-color:#fff9c4;color:#f57f17"
            return ""
        except Exception:
            return ""

    styled = summary_df.style\
        .applymap(_color_ic,  subset=["IC均值"])\
        .applymap(_color_ir,  subset=["IR", "多空夏普"])

    st.dataframe(styled, use_container_width=True, height=min(50 * (len(summary_df) + 2), 500))

    # ---- 多因子对比雷达图 ----
    if len(available) >= 2:
        _divider()
        st.subheader("🎯 多因子对比")
        compare_factors = st.multiselect("选择对比因子（2-6个）", available,
                                          default=available[:min(4, len(available))])

        if len(compare_factors) >= 2:
            try:
                import altair as alt
                dims = ["IC均值(×100)", "IR", "多空夏普", "覆盖率(×10)", "IC>0占比(×10)"]
                comp_rows = []
                for name in compare_factors:
                    r = st.session_state.factor_results[name]
                    comp_rows.append({
                        "因子": name,
                        "IC均值(×100)": abs(r.ic_mean) * 100,
                        "IR":            abs(r.ir),
                        "多空夏普":      max(r.sharpe, 0),
                        "覆盖率(×10)":   r.coverage * 10,
                        "IC>0占比(×10)": r.ic_positive_ratio * 10,
                    })
                comp_df = pd.DataFrame(comp_rows)
                long_df = comp_df.melt(id_vars="因子", var_name="维度", value_name="得分")

                radar = (
                    alt.Chart(long_df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("维度:N"),
                        y=alt.Y("得分:Q"),
                        color=alt.Color("因子:N"),
                        detail="因子:N",
                        tooltip=["因子:N", "维度:N",
                                 alt.Tooltip("得分:Q", format=".3f")],
                    )
                    .properties(height=300)
                )
                st.altair_chart(radar, use_container_width=True)
            except ImportError:
                st.dataframe(pd.DataFrame([{
                    "因子": name,
                    "IC均值": st.session_state.factor_results[name].ic_mean,
                    "IR":     st.session_state.factor_results[name].ir,
                } for name in compare_factors]))

    # ---- IC 对比折线图 ----
    if len(available) >= 2:
        _divider()
        st.subheader("📈 IC 序列对比")
        ic_compare = st.multiselect(
            "选择对比因子（IC序列）", available,
            default=available[:min(3, len(available))],
            key="ic_compare",
        )
        if ic_compare:
            ic_all = pd.DataFrame({
                name: st.session_state.factor_results[name].ic_series
                for name in ic_compare
                if not st.session_state.factor_results[name].ic_series.empty
            })
            if not ic_all.empty:
                st.line_chart(ic_all, height=260)

    # ---- 导出 ----
    _divider()
    st.subheader("⬇️ 导出报告")
    col_csv, col_json = st.columns(2)

    with col_csv:
        csv_data = summary_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            "📥 下载因子汇总 CSV",
            data=csv_data,
            file_name=f"factor_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_json:
        export_data = []
        for name in available:
            r = st.session_state.factor_results[name]
            export_data.append({
                "name":           r.name,
                "expression":     r.expression,
                "economic_logic": r.economic_logic,
                "ic_mean":        r.ic_mean,
                "ir":             r.ir,
                "sharpe":         r.sharpe,
                "long_short_return": r.long_short_return,
                "max_drawdown":   r.max_drawdown,
                "turnover":       r.turnover,
                "coverage":       r.coverage,
                "autocorr":       r.autocorr,
            })
        st.download_button(
            "📥 下载因子详情 JSON",
            data=json.dumps(export_data, ensure_ascii=False, indent=2),
            file_name=f"factor_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

    # ---- 删除 ----
    _divider()
    col_del1, col_del2 = st.columns([2, 1])
    with col_del1:
        to_delete = st.selectbox("删除因子", ["（不删除）"] + available)
    with col_del2:
        st.write("")
        st.write("")
        if to_delete != "（不删除）":
            if st.button(f"🗑 确认删除 {to_delete}", type="secondary"):
                st.session_state.factor_library.remove(to_delete)
                del st.session_state.factor_results[to_delete]
                _rerun()
