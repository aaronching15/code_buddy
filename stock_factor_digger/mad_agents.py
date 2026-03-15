"""
mad_agents.py
— MAD 辩论引擎

ProposerAgent：生成含经济学逻辑的因子假设 + Python 表达式
CriticAgent：从逻辑有效性、可测试性、新颖性、数据挖掘风险 4 维度评分
MediatorAgent：综合多轮辩论给出最终裁定
Demo 模式：未配置 API Key 时，内置 5 个精心设计的因子辩论样本（量价背离、短期反转、动量加速等），
    完整呈现辩论流程，无需 LLM 也能体验

=============
Multi-Agent Debate (MAD) 框架 —— FactorMAD 实现
  - ProposerAgent  : 提案者，生成因子假设与表达式
  - CriticAgent    : 批判者，从多维度质疑并提出改进
  - MediatorAgent  : 仲裁者，综合辩论结论，给出最终采纳意见
  - DebateSession  : 管理单轮辩论（多轮迭代 + 早停）
  - FactorMAD      : 整体流程编排，对接因子引擎

依赖 OpenAI-compatible API（支持 openai / azure / 兼容本地模型）。
若未配置 API Key，则回退到内置规则引擎模拟辩论（Demo 模式）。
"""

from __future__ import annotations

import json
import os
import re
import time
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────
# LLM 接入层（支持 openai / 兼容接口）
# ──────────────────────────────────────────────────────────────────

def _try_import_openai():
    try:
        from openai import OpenAI
        return OpenAI
    except ImportError:
        return None


def _llm_call(
    messages: List[Dict],
    model: str = "gpt-4o-mini",
    api_key: str = "",
    base_url: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1500,
) -> str:
    """统一 LLM 调用入口，失败时返回空字符串"""
    OpenAI = _try_import_openai()
    if OpenAI is None or not api_key:
        return ""

    try:
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"[LLM ERROR] {e}"


# ──────────────────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────────────────

@dataclass
class FactorProposal:
    """因子提案"""
    name: str
    economic_logic: str          # 经济学直觉/逻辑
    hypothesis: str              # 可验证假设
    expression: str              # 计算表达式（供因子引擎 eval）
    expected_direction: str      # "正向" 或 "负向"
    applicable_market: str       # 适用市场环境
    risk_notes: str              # 潜在风险/局限


@dataclass
class CriticOpinion:
    """批判意见"""
    validity_score: int          # 经济逻辑有效性 1-10
    testability_score: int       # 可测试性 1-10
    novelty_score: int           # 新颖性 1-10
    risk_score: int              # 数据挖掘风险 1-10（越低越好）
    overall_score: float         # 综合评分
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    revised_expression: str      # 建议修改后的表达式（可为空）
    approve: bool                # 是否建议采纳


@dataclass
class DebateRound:
    """单轮辩论记录"""
    round_num: int
    proposal: FactorProposal
    critic_opinion: CriticOpinion
    proposer_response: str       # 提案者对批判的回应


@dataclass
class DebateResult:
    """辩论最终结果"""
    topic: str
    rounds: List[DebateRound]
    final_proposal: FactorProposal
    final_opinion: CriticOpinion
    mediator_summary: str
    adopted: bool
    total_rounds: int
    is_demo_mode: bool = False   # True = 规则引擎模拟


# ──────────────────────────────────────────────────────────────────
# 提案者 Agent
# ──────────────────────────────────────────────────────────────────

class ProposerAgent:
    """
    提案者：
    - 接受研究主题/方向
    - 输出经济逻辑清晰的因子假设 + 计算表达式
    - 面对批评时修订提案
    """

    SYSTEM_PROMPT = textwrap.dedent("""
        你是一位顶级量化研究员，专注 A 股市场因子挖掘。
        你的任务是提出具有清晰经济学逻辑的股票选因子。

        **输出格式要求（严格 JSON）**：
        {
          "name": "因子名称（英文下划线，如 momentum_reversal_combo）",
          "economic_logic": "经济学直觉，50-150字",
          "hypothesis": "可验证假设，如「过去1个月收益较高的股票，下个月收益会回落」",
          "expression": "Python 表达式，仅可使用：close/open/high/low/volume 五个变量，以及 ret/ma/ema/std/rank/ts_rank/corr/delta/delay/log/abs/sign 函数",
          "expected_direction": "正向 或 负向",
          "applicable_market": "适用市场环境描述",
          "risk_notes": "潜在风险或局限"
        }
    """).strip()

    def __init__(self, llm_config: Dict):
        self.llm_config = llm_config

    def propose(self, topic: str, context: str = "") -> FactorProposal:
        """初始提案"""
        user_msg = f"研究主题：{topic}\n\n背景参考：{context}" if context else f"研究主题：{topic}"
        raw = _llm_call(
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            **self.llm_config,
        )
        return self._parse(raw, topic)

    def revise(
        self,
        original: FactorProposal,
        critic: CriticOpinion,
        round_num: int,
    ) -> Tuple[FactorProposal, str]:
        """根据批判修订提案，同时返回辩护/回应文字"""
        critique_summary = "\n".join([
            f"- 弱点：{w}" for w in critic.weaknesses
        ] + [
            f"- 建议：{s}" for s in critic.suggestions
        ])
        if critic.revised_expression:
            critique_summary += f"\n- 批评者建议改为：{critic.revised_expression}"

        user_msg = textwrap.dedent(f"""
            这是第 {round_num} 轮辩论修订。
            你的原始提案：
              名称: {original.name}
              表达式: {original.expression}
              逻辑: {original.economic_logic}

            批判意见（综合评分 {critic.overall_score:.1f}/10）：
            {critique_summary}

            请：
            1. 先用 1-2 句话回应批评（说明你认同或不认同的理由）
            2. 然后输出修订后的 JSON 提案

            格式：
            [RESPONSE]
            你的回应文字
            [/RESPONSE]
            [PROPOSAL]
            {{修订后的 JSON}}
            [/PROPOSAL]
        """).strip()

        raw = _llm_call(
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            **self.llm_config,
        )

        response_text = ""
        proposal_json = raw
        if "[RESPONSE]" in raw and "[/RESPONSE]" in raw:
            m = re.search(r"\[RESPONSE\](.*?)\[/RESPONSE\]", raw, re.DOTALL)
            if m:
                response_text = m.group(1).strip()
        if "[PROPOSAL]" in raw and "[/PROPOSAL]" in raw:
            m = re.search(r"\[PROPOSAL\](.*?)\[/PROPOSAL\]", raw, re.DOTALL)
            if m:
                proposal_json = m.group(1).strip()

        revised = self._parse(proposal_json, original.name)
        return revised, response_text

    def _parse(self, raw: str, fallback_name: str) -> FactorProposal:
        """解析 LLM 返回的 JSON"""
        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                return FactorProposal(
                    name=data.get("name", fallback_name),
                    economic_logic=data.get("economic_logic", ""),
                    hypothesis=data.get("hypothesis", ""),
                    expression=data.get("expression", ""),
                    expected_direction=data.get("expected_direction", "正向"),
                    applicable_market=data.get("applicable_market", ""),
                    risk_notes=data.get("risk_notes", ""),
                )
        except Exception:
            pass
        return FactorProposal(
            name=fallback_name,
            economic_logic=raw[:200] if raw else "解析失败",
            hypothesis="",
            expression="",
            expected_direction="正向",
            applicable_market="",
            risk_notes="LLM 返回格式异常",
        )


# ──────────────────────────────────────────────────────────────────
# 批判者 Agent
# ──────────────────────────────────────────────────────────────────

class CriticAgent:
    """
    批判者：
    - 从经济逻辑、过拟合风险、市场适用性、表达式合理性等多维度批判
    - 给出评分及改进建议
    """

    SYSTEM_PROMPT = textwrap.dedent("""
        你是一位严格的量化策略评审专家，负责批判同事提出的股票因子。
        你需要从以下维度评估：
        1. 经济学逻辑有效性（有无扎实的行为金融/微观结构依据）
        2. 数据挖掘风险（是否过于依赖历史拟合）
        3. 可测试性（假设是否清晰可证伪）
        4. 新颖性（是否有别于已有经典因子）
        5. 表达式健壮性（是否有除零、量纲问题等）

        **输出格式要求（严格 JSON）**：
        {
          "validity_score": 整数1-10,
          "testability_score": 整数1-10,
          "novelty_score": 整数1-10,
          "risk_score": 整数1-10（越高=数据挖掘风险越高，越低越好）,
          "overall_score": 浮点数（综合评分，取前三项均值减去risk_score×0.3）,
          "strengths": ["优点1", "优点2"],
          "weaknesses": ["弱点1", "弱点2"],
          "suggestions": ["改进建议1", "改进建议2"],
          "revised_expression": "若你有更好的表达式写在这里，否则为空字符串",
          "approve": true 或 false（综合评分>=6.5 且 risk_score<=6 时建议 true）
        }
    """).strip()

    def __init__(self, llm_config: Dict):
        self.llm_config = llm_config

    def critique(
        self,
        proposal: FactorProposal,
        eval_result_summary: str = "",
    ) -> CriticOpinion:
        """批判一个因子提案"""
        user_msg = textwrap.dedent(f"""
            请评审以下因子提案：

            **因子名称**: {proposal.name}
            **经济学逻辑**: {proposal.economic_logic}
            **可验证假设**: {proposal.hypothesis}
            **计算表达式**: {proposal.expression}
            **预期方向**: {proposal.expected_direction}
            **适用市场**: {proposal.applicable_market}
            **风险提示**: {proposal.risk_notes}
            {f"**初步回测摘要**: {eval_result_summary}" if eval_result_summary else ""}
        """).strip()

        raw = _llm_call(
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            **self.llm_config,
        )
        return self._parse(raw)

    def _parse(self, raw: str) -> CriticOpinion:
        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                d = json.loads(m.group())
                vs = int(d.get("validity_score", 5))
                ts = int(d.get("testability_score", 5))
                ns = int(d.get("novelty_score", 5))
                rs = int(d.get("risk_score", 5))
                ov = d.get("overall_score", (vs + ts + ns) / 3 - rs * 0.3)
                return CriticOpinion(
                    validity_score=vs,
                    testability_score=ts,
                    novelty_score=ns,
                    risk_score=rs,
                    overall_score=round(float(ov), 2),
                    strengths=d.get("strengths", []),
                    weaknesses=d.get("weaknesses", []),
                    suggestions=d.get("suggestions", []),
                    revised_expression=d.get("revised_expression", ""),
                    approve=bool(d.get("approve", ov >= 6.5)),
                )
        except Exception:
            pass
        return CriticOpinion(
            validity_score=5, testability_score=5, novelty_score=5,
            risk_score=5, overall_score=5.0,
            strengths=[], weaknesses=["LLM 返回解析失败"],
            suggestions=[], revised_expression="", approve=False,
        )


# ──────────────────────────────────────────────────────────────────
# 仲裁者 Agent
# ──────────────────────────────────────────────────────────────────

class MediatorAgent:
    """
    仲裁者：
    - 综合 N 轮辩论，给出最终裁定
    - 提炼最终因子的改进建议
    """

    SYSTEM_PROMPT = textwrap.dedent("""
        你是辩论仲裁专家。你需要综合阅读量化因子辩论的多个回合，给出：
        1. 辩论核心分歧的总结
        2. 最终因子的综合评价
        3. 是否采纳（adopted: true/false）
        4. 最终改进建议

        用中文输出，200-400字，段落清晰。
        最后一行单独写：【最终裁定：采纳】 或 【最终裁定：不采纳】
    """).strip()

    def __init__(self, llm_config: Dict):
        self.llm_config = llm_config

    def summarize(self, rounds: List[DebateRound], topic: str) -> Tuple[str, bool]:
        """生成仲裁摘要，返回 (summary, adopted)"""
        debate_text = "\n\n".join([
            f"=== 第{r.round_num}轮 ===\n"
            f"提案表达式: {r.proposal.expression}\n"
            f"提案逻辑: {r.proposal.economic_logic}\n"
            f"批评综合评分: {r.critic_opinion.overall_score}/10\n"
            f"批评要点: {'; '.join(r.critic_opinion.weaknesses[:2])}\n"
            f"提案者回应: {r.proposer_response[:100]}"
            for r in rounds
        ])

        user_msg = f"研究主题：{topic}\n\n辩论过程摘要：\n{debate_text}"
        raw = _llm_call(
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.4,
            **{k: v for k, v in self.llm_config.items() if k != "temperature"},
        )
        if not raw:
            raw = "（Demo模式：无 LLM 输出）"

        adopted = "采纳" in raw.split("最终裁定")[-1] and "不采纳" not in raw.split("最终裁定")[-1]
        return raw, adopted


# ──────────────────────────────────────────────────────────────────
# Demo 模式：规则引擎（无 LLM 时的回退）
# ──────────────────────────────────────────────────────────────────

_DEMO_PROPOSALS: List[Dict] = [
    {
        "name": "volume_price_divergence",
        "economic_logic": "量价背离因子：当价格上涨而成交量萎缩时，往往意味着上涨动能不足，机构资金并未真正参与，后续可能出现回调。反之，价格下跌伴随缩量则意味着抛压减轻。",
        "hypothesis": "过去5日价格涨幅排名靠前但成交量排名靠后的股票，未来20日超额收益为负。",
        "expression": "rank(ret(5)) - rank(ma(volume, 5) / ma(volume, 20))",
        "expected_direction": "负向",
        "applicable_market": "震荡市与下行市更有效，趋势市可能失效",
        "risk_notes": "在 FOMO 行情中（全市场量价齐升）可能失效；北向资金流入期间逻辑受扰",
    },
    {
        "name": "short_term_reversal_vol_adj",
        "economic_logic": "短期反转+波动率调整：短期超卖后的反转效应在 A 股显著，但高波动股票反转更剧烈也更不稳定，用波动率标准化后可提升因子稳健性。",
        "hypothesis": "过去5日跌幅（用20日波动率标准化）越大的股票，未来20日收益越高。",
        "expression": "-(ret(5) / (std(ret(1), 20) + 1e-6))",
        "expected_direction": "正向",
        "applicable_market": "适合交投活跃的环境，流动性差的小票效果更强",
        "risk_notes": "连续下跌趋势市场中可能持续亏损（价值陷阱）",
    },
    {
        "name": "momentum_acceleration",
        "economic_logic": "动量加速因子：近期动量（1月）超过中期动量（3月）意味着趋势在加速，反映资金持续追捧，符合趋势追踪的行为金融学依据。",
        "hypothesis": "1月动量高于3月动量（即动量在加速）的股票，未来20日收益更高。",
        "expression": "ret(20) - ret(60)",
        "expected_direction": "正向",
        "applicable_market": "趋势型市场效果好，横盘市信噪比低",
        "risk_notes": "容易与市场整体趋势混淆，需做市场中性处理；换手率较高",
    },
    {
        "name": "intraday_overnight_ratio",
        "economic_logic": "日内/隔夜收益结构因子：散户倾向在开盘竞价形成跳空（情绪驱动），机构则更多通过日内交易获利。若隔夜收益长期高于日内，说明该股票散户情绪溢价高，后续可能回归。",
        "hypothesis": "近20日隔夜平均收益远高于日内平均收益的股票，散户情绪溢价较高，未来回归空间大。",
        "expression": "ma(delay(close, 0) / delay(close, 1) - 1, 20) - ma(close / open - 1, 20)",
        "expected_direction": "负向",
        "applicable_market": "散户主导、情绪波动大的时期",
        "risk_notes": "涨停板机制可能导致隔夜跳空偏大，需排除涨跌停样本",
    },
    {
        "name": "volume_surge_reversal",
        "economic_logic": "成交量突破反转：异常高成交量通常伴随市场情绪极端化（恐慌或FOMO），随后价格往往反转回归均值。用当日成交量与历史均量的比值衡量情绪极端程度。",
        "hypothesis": "成交量显著高于60日均量（量比>2）的股票，短期内出现反转的概率更高。",
        "expression": "-sign(ret(5)) * (volume / ma(volume, 60))",
        "expected_direction": "正向（反转方向）",
        "applicable_market": "成交量充裕的活跃市场",
        "risk_notes": "需区分放量上涨（趋势延续）与放量下跌（止损抛售）；在趋势明确期可能失效",
    },
]

_DEMO_CRITIQUES: List[Dict] = [
    {
        "scores": (7, 8, 7, 4, 7.4),
        "strengths": ["量价分析是经典研究框架", "表达式使用了截面相对排名，控制了市场整体影响"],
        "weaknesses": ["量价背离在流动性低的标的上信号噪音较大", "未考虑行业效应，可能被行业轮动混淆"],
        "suggestions": ["建议在行业内进行中性化处理", "可加入成交量的趋势项以过滤噪音"],
        "revised": "rank(ret(5)) - rank(delta(ma(volume, 5), 5) / ma(volume, 20))",
        "approve": True,
    },
    {
        "scores": (8, 8, 6, 3, 8.1),
        "strengths": ["波动率标准化是专业处理，提升可比性", "反转效应在 A 股有充分学术证据"],
        "weaknesses": ["1e-6 的截断虽然处理了除零，但数值稳定性仍需关注", "短期反转与市值因子相关性较高"],
        "suggestions": ["建议对市值进行中性化", "可扩展为5日和10日反转的组合，提升稳定性"],
        "revised": "",
        "approve": True,
    },
    {
        "scores": (7, 8, 5, 5, 6.5),
        "strengths": ["动量加速思路直观", "计算简洁，易于实现"],
        "weaknesses": ["与经典动量因子高度相关，新颖性不足", "在 A 股动量效应整体较弱的背景下有效性存疑"],
        "suggestions": ["考虑加入交叉资产信号（如行业动量）", "建议与成交量结合，筛选有资金参与的动量"],
        "revised": "ret(20) - ret(60) + rank(ma(volume, 5) / ma(volume, 60))",
        "approve": False,
    },
    {
        "scores": (8, 7, 9, 4, 7.8),
        "strengths": ["结构因子设计有独特视角", "行为金融逻辑清晰，有学术支撑"],
        "weaknesses": ["A 股涨跌停机制使隔夜收益计算存在系统性偏差", "需要更严格的样本过滤"],
        "suggestions": ["建议剔除涨跌停次日样本", "可考虑用近5日均值替代单日，降低极端值影响"],
        "revised": "ma(delay(close, 0) / delay(close, 1) - 1, 10) - ma(close / open - 1, 10)",
        "approve": True,
    },
    {
        "scores": (7, 7, 7, 5, 6.7),
        "strengths": ["成交量异常与反转的关系有微观结构理论支撑", "sign 函数的使用考虑了方向性"],
        "weaknesses": ["sign 函数信息损失较大，幅度信息被丢弃", "量比的绝对阈值需要时变校准"],
        "suggestions": ["考虑用连续量比代替 sign 二值化", "建议加入价格波动率约束，过滤极端振幅样本"],
        "revised": "-(ret(5)) * log(volume / ma(volume, 60))",
        "approve": True,
    },
]

_DEMO_TOPICS: List[str] = [
    "量价背离", "短期反转", "动量加速", "散户情绪结构", "成交量异常反转",
]


def _make_demo_proposal(idx: int) -> FactorProposal:
    d = _DEMO_PROPOSALS[idx % len(_DEMO_PROPOSALS)]
    return FactorProposal(**d)


def _make_demo_critique(idx: int, round_num: int) -> CriticOpinion:
    base = _DEMO_CRITIQUES[idx % len(_DEMO_CRITIQUES)]
    vs, ts, ns, rs, ov = base["scores"]
    # 第2轮后评分略微上升（体现迭代改进）
    boost = 0.3 * (round_num - 1)
    return CriticOpinion(
        validity_score=min(10, vs),
        testability_score=min(10, ts),
        novelty_score=min(10, ns),
        risk_score=max(1, rs),
        overall_score=round(min(10.0, ov + boost), 2),
        strengths=base["strengths"],
        weaknesses=base["weaknesses"] if round_num == 1 else base["weaknesses"][:1],
        suggestions=base["suggestions"],
        revised_expression=base["revised"] if round_num == 1 else "",
        approve=(ov + boost >= 6.5),
    )


# ──────────────────────────────────────────────────────────────────
# 辩论会话
# ──────────────────────────────────────────────────────────────────

class DebateSession:
    """
    管理一次完整辩论：
    - 最多 max_rounds 轮
    - 批判者连续两轮 approve=True 则提前结束
    - 支持实时 yield 每轮结果（供 Streamlit 流式展示）
    """

    def __init__(
        self,
        proposer: ProposerAgent,
        critic: CriticAgent,
        mediator: MediatorAgent,
        max_rounds: int = 3,
        approval_threshold: float = 6.5,
    ):
        self.proposer   = proposer
        self.critic     = critic
        self.mediator   = mediator
        self.max_rounds = max_rounds
        self.approval_threshold = approval_threshold

    def run(
        self,
        topic: str,
        initial_context: str = "",
        eval_summary_fn=None,   # 可选：每轮后调用因子引擎拿评估摘要
        is_demo: bool = False,
        demo_idx: int = 0,
    ) -> DebateResult:
        """同步运行，返回最终结果"""
        rounds: List[DebateRound] = []

        # 初始提案
        if is_demo:
            proposal = _make_demo_proposal(demo_idx)
        else:
            proposal = self.proposer.propose(topic, initial_context)

        consecutive_approvals = 0

        for rn in range(1, self.max_rounds + 1):
            # 获取评估摘要（可选）
            eval_summary = ""
            if eval_summary_fn and not is_demo:
                try:
                    eval_summary = eval_summary_fn(proposal.expression)
                except Exception:
                    eval_summary = ""

            # 批判
            if is_demo:
                opinion = _make_demo_critique(demo_idx, rn)
            else:
                opinion = self.critic.critique(proposal, eval_summary)

            # 提案者回应（仅非最后一轮）
            response_text = ""
            if rn < self.max_rounds:
                if is_demo:
                    response_text = _DEMO_RESPONSES[rn % len(_DEMO_RESPONSES)]
                else:
                    proposal, response_text = self.proposer.revise(proposal, opinion, rn)

            rounds.append(DebateRound(
                round_num=rn,
                proposal=proposal,
                critic_opinion=opinion,
                proposer_response=response_text,
            ))

            # 早停：连续 2 轮被批准
            if opinion.approve:
                consecutive_approvals += 1
            else:
                consecutive_approvals = 0

            if consecutive_approvals >= 2:
                break

            # 非最后轮：用修订后提案继续
            if rn < self.max_rounds and not is_demo:
                pass  # proposal 已在 revise 中更新

        # 仲裁
        final_opinion = rounds[-1].critic_opinion
        final_proposal = rounds[-1].proposal

        if is_demo:
            med_text = _make_demo_mediator(final_opinion, topic)
            adopted  = final_opinion.approve
        else:
            med_text, adopted = self.mediator.summarize(rounds, topic)

        return DebateResult(
            topic=topic,
            rounds=rounds,
            final_proposal=final_proposal,
            final_opinion=final_opinion,
            mediator_summary=med_text,
            adopted=adopted,
            total_rounds=len(rounds),
            is_demo_mode=is_demo,
        )


_DEMO_RESPONSES: List[str] = [
    "感谢批评者的意见。我认同关于行业中性化的建议，同时坚持表达式的基本框架不变，因为截面排名已部分控制行业影响。",
    "接受波动率相关性的质疑，已在修订版本中加入市值中性化处理，并扩展了时间窗口以提升稳定性。",
    "认同信息损失的问题，已改用连续形式替代 sign 函数，同时保留原始逻辑的核心方向。",
]


def _make_demo_mediator(opinion: CriticOpinion, topic: str) -> str:
    verdict = "采纳" if opinion.approve else "不采纳"
    return textwrap.dedent(f"""
        本轮辩论围绕「{topic}」因子展开，共经历多轮提案与批判迭代。

        **核心分歧**：批判者主要质疑行业效应混淆及数据挖掘风险，提案者则坚持截面排名能够部分控制系统性因素。

        **最终因子评估**：
        - 经济逻辑有效性：{opinion.validity_score}/10 — 具备一定理论依据
        - 可测试性：{opinion.testability_score}/10 — 假设清晰可验证
        - 新颖性：{opinion.novelty_score}/10
        - 数据挖掘风险：{opinion.risk_score}/10（越低越好）
        - 综合评分：{opinion.overall_score:.1f}/10

        **结论**：经过辩论迭代，因子表达式已充分考虑批判建议并完成修订，综合评分满足采纳门槛。

        【最终裁定：{verdict}】
    """).strip()


# ──────────────────────────────────────────────────────────────────
# FactorMAD 主类
# ──────────────────────────────────────────────────────────────────

class FactorMAD:
    """
    因子多智能体辩论 (Multi-Agent Debate) 主类
    整合 ProposerAgent + CriticAgent + MediatorAgent + DebateSession
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "gpt-4o-mini",
        max_rounds: int = 3,
        temperature: float = 0.7,
    ):
        self.is_demo = not bool(api_key)
        llm_cfg: Dict = {
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "temperature": temperature,
        }
        self.proposer  = ProposerAgent(llm_cfg)
        self.critic    = CriticAgent(llm_cfg)
        self.mediator  = MediatorAgent(llm_cfg)
        self.max_rounds = max_rounds
        self._demo_counter = 0

    def run_debate(
        self,
        topic: str,
        context: str = "",
        eval_summary_fn=None,
    ) -> DebateResult:
        """
        运行一次完整的因子辩论

        Args:
            topic           : 研究主题，如"动量反转组合"
            context         : 背景信息（行业、市场环境等）
            eval_summary_fn : 可选回调，接受 expression str，返回评估摘要 str

        Returns:
            DebateResult
        """
        session = DebateSession(
            proposer=self.proposer,
            critic=self.critic,
            mediator=self.mediator,
            max_rounds=self.max_rounds,
        )
        result = session.run(
            topic=topic,
            initial_context=context,
            eval_summary_fn=eval_summary_fn,
            is_demo=self.is_demo,
            demo_idx=self._demo_counter % len(_DEMO_PROPOSALS),
        )
        self._demo_counter += 1
        return result

    def batch_debate(
        self,
        topics: List[str],
        eval_summary_fn=None,
    ) -> List[DebateResult]:
        """批量辩论多个主题"""
        results = []
        for topic in topics:
            r = self.run_debate(topic, eval_summary_fn=eval_summary_fn)
            results.append(r)
        return results

    @staticmethod
    def get_demo_topics() -> List[str]:
        return _DEMO_TOPICS.copy()
