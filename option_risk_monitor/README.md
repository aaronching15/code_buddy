# 场外期权实时风险监控系统

## 项目概述
使用Streamlit构建的场外期权对冲股票持仓的实时风险监控系统。

## 生成的文件

### 1. option_risk_monitor.py
完整的Streamlit应用，包含以下功能：

#### 核心功能模块：
- **任务0**: 自动下载A股行情数据
  - 保存到 `D:\CISS_db\quote_ashares\YYYYMMDD-HHMMSS.xlsx`
  - 同时保存到 `D:\auto_tc\data_sync\quote_now.xlsx` 和 `quote_now.csv`
  - 支持多种数据源（CISS数据库、akshare、efinance）

- **数据导入**:
  - 持仓期权: `全部持仓.xlsx` (sheet=全部)
  - 持仓股票: `df_stocks.xlsx`
  - 股票行情: `quote_now.csv` (RT_LAST=最新价, RT_PCT_CHG=涨跌幅)
  - 波动率预测: `para_option_pricing.xlsx` (vol_esti)
  - 模型参数: `固定参数.xlsx` (sheet=rKS)

- **期权定价计算**:
  - Black-Scholes模型
  - 计算Delta, Gamma, Vega, Theta

- **风险指标计算**:
  - 市值风险 = Delta × 名义本金
  - 浮盈亏% = (现价 - 成本价) / 成本价
  - 期权费剩余 = (成本价 × 浮盈亏% × 持仓股票数量 + 期权费) / 期权费
  - 需买+/卖-的市值 = 市值风险 - 持仓股票市值
  - 已对冲比例 = 持仓股票市值 / 市值风险

- **预警逻辑**:
  - 浮盈亏 < -5%: 卖出20%本金的持仓
  - 已对冲比例 > 120%: 卖出20%市值风险的持仓
  - Delta > 0.60: 买入20%本金的持仓
  - Delta < 0.41: 卖出持仓到40%本金或以下

- **输出文件**:
  - `df_delta.xlsx` - 完整的Delta监控表
  - `df_warm.xlsx` - 预警信息表

#### Streamlit界面:
- 实时监控页面：展示df_delta和df_warm表格
- 新增期权记录页面：可手动添加期权
- 每30秒自动刷新

### 2. requirements.txt
Python依赖文件

## 运行方式

```bash
# 安装依赖
pip install -r requirements.txt

# 运行应用
streamlit run option_risk_monitor.py
```

## 数据文件位置
- 输入目录: `D:\auto_tc\data_sync\`
- 输出目录: `D:\auto_tc\data_sync\` 和 `D:\CISS_db\quote_ashares\`

## 核心计算公式

### Black-Scholes期权定价
```
d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)
d2 = d1 - σ√T

Delta = N(d1)  (看涨期权)
Gamma = N'(d1) / (Sσ√T)
Vega = S·N'(d1)√T / 100
Theta = (-S·N'(d1)σ / 2√T - rK·e^(-rT)·N(d2)) / 365
```

### 市值风险
```
市值风险 = Delta × 名义本金
```
