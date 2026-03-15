# Strangle 卖出对冲策略系统

基于卖出宽跨式 Strangle（同时卖出 OTM 认购 + 认沽期权）的策略研究与实时监控平台。

## 功能概览

| 模块 | 功能 |
|------|------|
| 回测模拟 | 历史价格路径 + 逐日 BS 重定价 + 绩效统计 |
| 实时监控 | Net Greeks、浮动盈亏、预警（Delta/滚仓/亏损） |
| 持仓管理 | 新增/删除/保存/加载持仓快照 |
| 数据导入 | 手动上传 Excel / akshare 自动拉取 |

## 快速启动

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 Streamlit
cd option_sell_hedge
streamlit run app.py
```

浏览器访问 http://localhost:8501

## 文件结构

```
option_sell_hedge/
├── app.py                  # Streamlit 主入口（4 页路由）
├── bs_engine.py            # BS 定价引擎（买/卖双视角 Greeks）
├── strangle_builder.py     # OTM Delta 筛选 + Strangle 构建
├── backtest_engine.py      # 逐日回测 + 绩效统计
├── data_loader.py          # Excel 上传 + akshare 自动拉取
├── monitor.py              # 实时监控页面逻辑
├── templates/
│   └── option_input_template.xlsx  # 手动上传模板
├── requirements.txt
└── README.md
```

## 数据格式

### 期权列表 Excel（必填列）

| 列名 | 说明 | 示例 |
|------|------|------|
| ts_code | 合约代码 | 510050 |
| strike_price | 行权价 | 3.2 |
| call_put | 类型：C/P | C |
| exp_date | 到期日（YYYY-MM-DD）| 2024-01-26 |
| open_price | 期权开仓价 | 0.0300 |
| iv | 隐含波动率（小数）| 0.20 |
| multiplier | 合约乘数（可选，默认 10000）| 10000 |
| quantity | 手数（可选，默认 1）| 1 |

### 价格路径 Excel

| 列名 | 说明 |
|------|------|
| date | 日期（YYYY-MM-DD）|
| close | 收盘价 |

## 支持标的

| 标的 | ETF 代码 | akshare symbol |
|------|---------|---------------|
| 上证 50 ETF | 510050 | 50ETF |
| 沪深 300 ETF | 510300 | 300ETF |
| 中证 1000 ETF | 159922 | 1000ETF |
| 科创 50 ETF | 588000 | 科创50ETF |

## 预警规则

| 预警类型 | 触发条件 | 等级 |
|---------|---------|------|
| Delta 风险 | \|Net Delta\| > 0.30 | 红色 |
| 滚仓预警 | 剩余天数 ≤ 5 | 橙色 |
| 亏损预警 | 亏损 > 初始权利金 × 1.5 | 红色 |

## 策略逻辑

1. **选仓**：筛选 |Delta| ∈ [0.10, 0.30] 的 OTM 认购 + 认沽
2. **建仓**：同时卖出，收取权利金
3. **持仓**：标的在两行权价之间横盘时，Theta 衰减带来正收益
4. **风险**：标的大幅突破行权价（黑天鹅）导致被行权损失

## 注意事项

- 本系统仅供研究和学习使用，不构成任何投资建议
- akshare 接口可能因版本更新失效，建议保留手动上传模式作为备用
- Python 3.7+ 兼容，建议使用 Python 3.8+
