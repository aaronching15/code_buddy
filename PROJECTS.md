# Code Buddy 开发项目清单

## 项目列表

### 1. option_risk_monitor - 场外期权实时风险监控系统
- **开发时间**: 2025-03-13
- **项目路径**: `c:/rc/ciss_web/CISS_rc/apps/agent/code_buddy/option_risk_monitor/`
- **功能描述**: 使用Streamlit构建的场外期权对冲股票持仓的实时风险监控系统
- **核心功能**:
  - 自动下载A股行情数据（每30秒/1分钟）
  - 计算期权Delta、Gamma、Vega、Theta希腊字母
  - 计算市值风险、浮盈亏、期权费剩余、对冲比例
  - 预警逻辑：浮盈亏<-5%、已对冲比例>120%、Delta>0.60、Delta<0.41
  - 每30秒自动刷新
- **技术栈**: Streamlit, Pandas, NumPy, SciPy
- **数据文件**:
  - 输入: 全部持仓.xlsx, df_stocks.xlsx, quote_now.csv, para_option_pricing.xlsx, 固定参数.xlsx
  - 输出: df_delta.xlsx, df_warm.xlsx

---

### 2. cta_strategy - 多资产CTA策略回测框架
- **开发时间**: 2025-03-13
- **项目路径**: `c:/rc/ciss_web/CISS_rc/apps/agent/code_buddy/cta_strategy/`
- **功能描述**: 多资产CTA策略回测框架，支持股票指数期货、ETF、股票、商品期货
- **核心策略**:
  - 双均线策略 (Dual Moving Average)
  - 海龟交易法则 (Turtle Strategy - 唐奇安通道突破)
  - 布林带突破策略 (Bollinger Band)
  - MACD策略
  - RSI策略
- **技术栈**: Python, Pandas, NumPy
- **投资标的**:
  - 股票指数期货 (IF, IC, IH, T)
  - ETF (股票ETF, 商品ETF, 债券ETF)
  - 股票 (A股)
  - 商品期货 (螺纹钢, 焦煤, 原油等)

---

### 3. stock_factor_digger - 基于 MAD 框架的股票因子研究平台
- **开发时间**: 2026-03-14
- **项目路径**: `c:/rc/ciss_web/CISS_rc/apps/agent/code_buddy/stock_factor_digger/`
- **功能描述**: 基于 Multi-Agent Debate (MAD) 框架的 A 股因子挖掘研究平台，实现从"提案-批判-迭代-仲裁"的全流程因子生成与评估
- **核心架构**:
  - `ProposerAgent`：提案者，生成具备经济学逻辑的因子假设与计算表达式
  - `CriticAgent`：批判者，从逻辑有效性、过拟合风险、新颖性等维度评审
  - `MediatorAgent`：仲裁者，综合多轮辩论给出最终采纳裁定
  - `DebateSession`：辩论会话，支持多轮迭代与早停
  - `FactorEngine`：因子计算引擎（20+ 内置因子 + 自定义表达式）与全套评估指标
- **因子评估指标**:
  - IC 序列（Spearman 秩相关）、IR（Information Ratio）
  - 分层收益（N层组合超额收益单调性）
  - 多空组合年化收益、最大回撤、夏普比率
  - 因子自相关、换手率、截面覆盖率
- **LLM 支持**: 兼容 OpenAI / Azure / 本地兼容接口；未配置 API Key 时自动进入 Demo 模式（规则引擎模拟辩论）
- **技术栈**: Streamlit, Pandas, NumPy, SciPy, Altair, OpenAI SDK（可选）
- **文件结构**:
  - `app.py`：Streamlit 主界面（4步骤：数据准备 → MAD辩论 → 因子评估 → 因子库）
  - `factor_engine.py`：因子计算与评估引擎
  - `mad_agents.py`：Multi-Agent Debate 框架实现
  - `requirements.txt`：依赖列表

---

## 项目结构

```
c:/rc/ciss_web/CISS_rc/apps/agent/code_buddy/
├── option_risk_monitor/          # 场外期权风险监控系统
│   ├── option_risk_monitor.py     # 主程序
│   ├── requirements.txt           # 依赖
│   └── README.md                   # 说明文档
├── cta_strategy/                  # 多资产CTA策略回测框架
│   ├── cta_strategy.py            # 主程序
│   └── requirements.txt           # 依赖
└── PROJECTS.md                    # 项目清单
```

## 运行方式

```bash
cd c:/rc/ciss_web/CISS_rc/apps/agent/code_buddy/option_risk_monitor
pip install -r requirements.txt
streamlit run option_risk_monitor.py
```
