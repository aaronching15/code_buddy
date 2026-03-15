# -*- encoding:"utf-8"-*-
__author__=" ruoyu.Cheng"

################################################################################################
### 
import streamlit as st
# ----------------- 页面配置 & 自动刷新 -----------------
st.set_page_config(page_title="场外期权对冲风险监控", layout="wide")


#################################################################################
### Initialization 
import os 
import sys
sys.path.append(os.getcwd()[:2] + "\\rc\\ciss_web\\CISS_rc\\db\\db_assets\\" ) 

import pandas as pd
import numpy as np
import math
from scipy.stats import norm
import time 
import datetime as dt
import shutil

# ----------------- 打印脚本所在目录 -----------------
print(f"脚本所在目录: {os.path.dirname(os.path.abspath(__file__))}")
print(f"工作目录: {os.getcwd()}")


# ----------------- 配置路径 -----------------
PATH_DATA = "D:\\auto_tc\\data_sync\\"
PATH_SYNC = "D:\\auto_tc\\data_sync\\"
PATH_QUOTE = "D:\\CISS_db\\quote_ashares\\"

# 确保目录存在
for path in [PATH_DATA, PATH_SYNC, PATH_QUOTE]:
    if not os.path.exists(path):
        os.makedirs(path)

# 刷新频率为30秒
REFRESH_RATE = 30 


# ----------------- 数值格式化函数 -----------------
def format_value(val, col_name):
    """
    数值格式化函数
    对于"今日涨跌幅", "浮盈亏%|-5%", "已对冲比例"这三列：乘以100后保留1位小数
    其他数值列：保留1位小数
    """
    if pd.isna(val):
        return val
    
    # 对于需要乘以100的列
    multiply_100_cols = ["今日涨跌幅", "浮盈亏%|-5%", "已对冲比例"]
    if col_name in multiply_100_cols:
        # 乘以100并保留1位小数
        try:
            return f"{(val * 100):.1f}"
        except:
            return val
    
    # 其他数值列保留1位小数
    if isinstance(val, (int, float, np.integer, np.floating)):
        try:
            return f"{val:.1f}"
        except:
            return val
    
    return val


def safe_format_number(x, multiply_by_100=False):
    """安全格式化数值，处理各种数据类型"""
    if pd.isna(x):
        return ""
    
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            if multiply_by_100:
                return f"{(x * 100):.1f}"
            else:
                return f"{x:.1f}"
        except:
            return str(x)
    else:
        # 对于非数值类型，直接返回字符串
        return str(x)


################################################################################################
### 任务0: 自动下载A股行情数据
def download_quote_data():
    """
    下载A股最新行情数据
    保存到:
    - D:\CISS_db\quote_ashares\YYYYMMDD-HHMMSS.xlsx
    - D:\auto_tc\data_sync\quote_now.xlsx
    """
    try:
        # 检查是否在A股交易时间（9:30-15:00）
        now = dt.datetime.now()
        current_time = now.time()
        
        # A股交易时间：9:30-11:30, 13:00-15:00
        morning_start = dt.time(9, 30)
        morning_end = dt.time(11, 30)
        afternoon_start = dt.time(13, 0)
        afternoon_end = dt.time(15, 0)
        
        is_trading_time = (morning_start <= current_time <= morning_end) or \
                         (afternoon_start <= current_time <= afternoon_end)
        
        # 周末不交易
        is_weekday = now.weekday() < 5
        
        if not (is_trading_time and is_weekday):
            print(f"当前不在A股交易时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
            # 仍然尝试读取现有数据
            return False
        
        print(f"正在下载A股行情数据... {now.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 方法1: 尝试从CISS数据库获取行情数据
        try:
            # 尝试导入CISS数据库连接模块
            sys.path.append("D:\\rc\\ciss_web\\CISS_rc\\db\\")
            from data_io import cs_assets_001
            
            # 获取A股实时行情
            df_quote = cs_assets_001().get_quote_ashares()
            
            if df_quote is not None and not df_quote.empty:
                # 格式化列名
                if 'RT_LAST' not in df_quote.columns:
                    # 尝试常见的列名映射
                    column_mapping = {
                        'last': 'RT_LAST',
                        'close': 'RT_LAST',
                        'price': 'RT_LAST',
                        'pct_chg': 'RT_PCT_CHG',
                        'change_pct': 'RT_PCT_CHG',
                        'low': 'RT_LOW',
                        'code': 'code',
                        'time': 'time'
                    }
                    for old_col, new_col in column_mapping.items():
                        if old_col in df_quote.columns:
                            df_quote = df_quote.rename(columns={old_col: new_col})
                
                # 确保必要列存在
                required_cols = ['RT_LAST', 'code']
                missing_cols = [c for c in required_cols if c not in df_quote.columns]
                if missing_cols:
                    print(f"警告: 缺少列 {missing_cols}")
                    raise Exception("缺少必要列")
                
                # 添加时间列
                df_quote['time'] = now.strftime("%H:%M:%S")
                
                # 生成文件名
                time_str = now.strftime("%Y%m%d-%H%M%S")
                
                # 保存到 quote_ashares 目录
                quote_file_ashares = os.path.join(PATH_QUOTE, f"{time_str}.xlsx")
                df_quote.to_excel(quote_file_ashares, index=False)
                print(f"行情数据已保存到: {quote_file_ashares}")
                
                # 保存到 data_sync 目录
                quote_file_now = os.path.join(PATH_DATA, "quote_now.xlsx")
                df_quote.to_excel(quote_file_now, index=False)
                print(f"行情数据已保存到: {quote_file_now}")
                
                # 同时保存CSV版本
                quote_csv = os.path.join(PATH_DATA, "quote_now.csv")
                df_quote.to_csv(quote_csv, index=False, encoding='utf-8')
                print(f"行情数据已保存到CSV: {quote_csv}")
                
                return True
                
        except Exception as e:
            print(f"从CISS数据库获取行情失败: {e}")
        
        # 方法2: 尝试从其他数据源获取
        try:
            # 尝试使用tushare或其他数据源
            import akshare as ak
            
            # 获取A股实时行情
            df_quote = ak.stock_zh_a_spot_em()
            
            if df_quote is not None and not df_quote.empty:
                # 标准化列名
                column_mapping = {
                    '代码': 'code',
                    '最新价': 'RT_LAST',
                    '涨跌幅': 'RT_PCT_CHG',
                    '最低': 'RT_LOW'
                }
                df_quote = df_quote.rename(columns=column_mapping)
                
                # 添加时间列
                df_quote['time'] = now.strftime("%H:%M:%S")
                
                # 生成文件名
                time_str = now.strftime("%Y%m%d-%H%M%S")
                
                # 保存文件
                quote_file_ashares = os.path.join(PATH_QUOTE, f"{time_str}.xlsx")
                df_quote.to_excel(quote_file_ashares, index=False)
                
                quote_file_now = os.path.join(PATH_DATA, "quote_now.xlsx")
                df_quote.to_excel(quote_file_now, index=False)
                
                quote_csv = os.path.join(PATH_DATA, "quote_now.csv")
                df_quote.to_csv(quote_csv, index=False, encoding='utf-8')
                
                print(f"通过akshare获取行情成功，共{len(df_quote)}条记录")
                return True
                
        except ImportError:
            print("akshare未安装，跳过")
        except Exception as e:
            print(f"通过akshare获取行情失败: {e}")
        
        # 方法3: 尝试使用efinance
        try:
            import efinance as ef
            
            # 获取A股实时行情
            df_quote = ef.stock.get_quote_history()
            
            if df_quote is not None and not df_quote.empty:
                # 标准化列名
                df_quote = df_quote.rename(columns={
                    '代码': 'code',
                    '最新价': 'RT_LAST',
                    '涨跌幅': 'RT_PCT_CHG',
                    '最低': 'RT_LOW'
                })
                df_quote['time'] = now.strftime("%H:%M:%S")
                
                time_str = now.strftime("%Y%m%d-%H%M%S")
                
                quote_file_ashares = os.path.join(PATH_QUOTE, f"{time_str}.xlsx")
                df_quote.to_excel(quote_file_ashares, index=False)
                
                quote_file_now = os.path.join(PATH_DATA, "quote_now.xlsx")
                df_quote.to_excel(quote_file_now, index=False)
                
                quote_csv = os.path.join(PATH_DATA, "quote_now.csv")
                df_quote.to_csv(quote_csv, index=False, encoding='utf-8')
                
                print(f"通过efinance获取行情成功，共{len(df_quote)}条记录")
                return True
                
        except ImportError:
            print("efinance未安装，跳过")
        except Exception as e:
            print(f"通过efinance获取行情失败: {e}")
        
        print("无法获取实时行情数据，将使用现有数据")
        return False
        
    except Exception as e:
        print(f"下载行情数据时出错: {e}")
        return False


# ----------------- 读取固定参数 -----------------
def load_para():
    """读取模型参数"""
    path_pricing = "D:\\auto_tc\\data_sync\\" 
    file_para = "固定参数.xlsx"  
    sheet1 = "rKS"
    
    df_temp = pd.read_excel(path_pricing + file_para, sheet_name=sheet1)
    r = df_temp.loc[0, "r"]
    date = df_temp.loc[0, "date"]
    K = df_temp.loc[0, "K"]
    S = df_temp.loc[0, "S"]
    
    return r, date, K, S


# ----------------- 核心定价公式 -----------------
def calculate_delta_values(S, K, T_days, r, sigma, option_type='call'):
    """
    计算期权的Delta, Gamma, Vega, Theta
    S: 现价, K: 行权价, T_days: 剩余天数, r: 利率, sigma: 波动率
    """
    if T_days <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0, 0.0
    
    T = T_days / 365.0
    
    # 避免除零错误
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0, 0.0, 0.0
    
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
            
        # Gamma
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Vega (结果通常除以100表示波动率变动1%的影响)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Theta (日损耗)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        
        return delta, gamma, vega, theta
    except:
        return 0.0, 0.0, 0.0, 0.0


# ----------------- 新增期权记录功能 -----------------
def add_option_record(new_option_data: dict) -> bool:
    """
    新增期权记录到Excel文件
    """
    try:
        file_path = os.path.join(PATH_DATA, "全部持仓.xlsx")
        
        if os.path.exists(file_path):
            df_existing = pd.read_excel(file_path, sheet_name="全部")
        else:
            df_existing = pd.DataFrame()
        
        # 标记为人工新增
        new_option_data['是否人工新增'] = '是'
        
        # 创建新的DataFrame
        df_new = pd.DataFrame([new_option_data])
        
        # 合并到现有数据
        if df_existing.empty:
            df_combined = df_new
        else:
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        
        # 保存到Excel文件
        df_combined.to_excel(file_path, sheet_name="全部", index=False)
        
        return True
        
    except Exception as e:
        st.error(f"新增期权记录失败: {e}")
        return False


def display_add_option_form():
    """显示新增期权记录的表单"""
    with st.expander("➕ 新增期权记录", expanded=False):
        st.markdown("请填写期权信息，提交后自动纳入监控")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            structure = st.selectbox("期权结构*", [" 100call", " 105call", " 80call"])
            stock_code = st.text_input("股票代码*", placeholder="例如: 000001.SZ")
            stock_name = st.text_input("股票名称*", placeholder="例如: 平安银行")
            
        with col2:
            remaining_days = st.number_input("剩余期限(天)*", min_value=1, max_value=3650, value=30)
            strike_price = st.number_input("行权价(期初价)*", min_value=0.0, value=10.0, step=0.01)
            nominal_amount = st.number_input("名义本金*", min_value=0.0, value=1000000.0, step=10000.0)
            
        with col3:
            premium = st.number_input("期权费*", min_value=0.0, value=0.0, step=1000.0)
            stock_qty = st.number_input("持仓股票数量*", min_value=0, value=0, step=100)
            cost_price = st.number_input("成本价", min_value=0.0, value=0.0, step=0.01, help="可选，如果不填则使用行权价")
        
        col4, col5 = st.columns(2)
        with col4:
            vol_adjusted = st.number_input("波动率(vol-调整后)", min_value=0.0, value=0.65, step=0.01, help="可选，默认0.65")
            client_name = st.text_input("客户姓名", placeholder="可选")
        
        submitted = st.button("提交新增", type="primary", use_container_width=True)
        
        if submitted:
            required_fields = {
                '结构': structure,
                '股票代码': stock_code,
                '股票名称': stock_name,
                '剩余日期(天)': remaining_days,
                '期初价': strike_price,
                '名义本金': nominal_amount,
                '期权费': premium,
                '持仓股票数量': stock_qty
            }
            
            missing_fields = [k for k, v in required_fields.items() if not v and v != 0]
            
            if missing_fields:
                st.error(f"请填写必填字段: {', '.join(missing_fields)}")
            else:
                new_option_data = {
                    '结构': structure,
                    '股票代码': stock_code.upper(),
                    '股票名称': stock_name,
                    '剩余日期(天)': remaining_days,
                    '期初价': strike_price,
                    '名义本金': nominal_amount,
                    '期权费': premium,
                    '持仓股票数量': stock_qty,
                    '成本价': cost_price if cost_price > 0 else strike_price,
                    'vol-调整后波动率': vol_adjusted if vol_adjusted > 0 else 0.65,
                    '中文客户姓名': client_name if client_name else '',
                    '是否人工新增': '是'
                }
                
                if add_option_record(new_option_data):
                    st.success(f"✅ 期权记录新增成功！股票: {stock_name} ({stock_code})")
                    st.experimental_rerun()
                else:
                    st.error("❌ 新增失败，请检查输入数据")


# ----------------- 预警卡片展示 -----------------
def display_warning_cards(df_warm):
    """以卡片形式展示预警股票"""
    if df_warm.empty:
        return
    
    st.markdown("### 🎯 预警股票快速查看")
    
    unique_stocks = df_warm[['股票代码', '股票名称', '需买+/卖-的市值']].drop_duplicates(subset=['股票代码'])
    stocks_list = unique_stocks.to_dict('records')
    
    cards_per_row = 5
    
    for i in range(0, len(stocks_list), cards_per_row):
        cols = st.columns(cards_per_row)
        for j in range(cards_per_row):
            if i + j < len(stocks_list):
                stock = stocks_list[i + j]
                with cols[j]:
                    hedge_value = stock.get('需买+/卖-的市值', 0)
                    if pd.notna(hedge_value):
                        if hedge_value > 0:
                            border_color = "#ff4444"
                            action_text = "需买入"
                        else:
                            border_color = "#ffaa00"
                            action_text = "需卖出"
                        
                        st.markdown(f"""
                        <div style="
                            border: 3px solid {border_color};
                            border-radius: 10px;
                            padding: 15px;
                            margin: 5px;
                            background-color: #f8f9fa;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        ">
                            <div style="font-size: 16px; font-weight: bold; color: #333; margin-bottom: 8px;">
                                {stock.get('股票名称', 'N/A')}
                            </div>
                            <div style="font-size: 14px; color: #666; margin-bottom: 5px;">
                                代码: {stock.get('股票代码', 'N/A')}
                            </div>
                            <div style="font-size: 16px; font-weight: bold; color: {border_color};">
                                {action_text}: {abs(hedge_value):,.0f}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)


# ----------------- 样式化info文本 -----------------
def style_info_text(info_str):
    """为info文本添加样式"""
    if pd.isna(info_str):
        return ""
    
    parts = info_str.split(" | ")
    styled_parts = []
    
    for part in parts:
        if "卖出" in part:
            styled = f'<span style="background-color: #ffeb3b; color: black; padding: 4px 8px; border-radius: 4px; font-weight: bold;">{part}</span>'
        elif "买入" in part:
            styled = f'<span style="background-color: black; color: #ff4444; padding: 4px 8px; border-radius: 4px; font-weight: bold;">{part}</span>'
        else:
            styled = part
        styled_parts.append(styled)
    
    return " | ".join(styled_parts)


# ----------------- 调整列顺序 -----------------
def reorder_columns(df):
    """将指定列移到最前面"""
    priority_cols = [
        "股票名称", "股票代码", "今日涨跌幅", "成本价", "浮盈亏%|-5%",
        "持仓股票数量", "期权费剩余", "需买+/卖-的市值", "已对冲比例",
        "持仓股票市值", "市值风险", "剩余日期(天)", "中文客户姓名"
    ]
    
    existing_priority_cols = [col for col in priority_cols if col in df.columns]
    other_cols = [col for col in df.columns if col not in existing_priority_cols]
    new_order = existing_priority_cols + other_cols
    
    return df[new_order]


# ----------------- 核心数据处理流 -----------------
def run_calculation():
    """
    核心计算流程:
    1. 导入数据文件
    2. 计算期权Delta等希腊字母
    3. 计算市值风险
    4. 从df_stocks读取并分配股票持仓
    5. 计算浮盈亏、期权费剩余、需买+/卖-的市值、已对冲比例
    6. 生成预警信息
    7. 保存结果
    """
    try:
        # 1. 导入股票行情数据
        print("=" * 50)
        print("步骤1: 导入数据文件")
        print("=" * 50)
        
        # 优先尝试读取quote_now.xlsx，如果没有则尝试CSV
        quote_files = [
            os.path.join(PATH_DATA, "quote_now.xlsx"),
            os.path.join(PATH_DATA, "quote_now.csv")
        ]
        
        df_quote = None
        for f in quote_files:
            if os.path.exists(f):
                try:
                    if f.endswith('.csv'):
                        df_quote = pd.read_csv(f, encoding='utf-8')
                    else:
                        df_quote = pd.read_excel(f)
                    print(f"成功读取行情文件: {f}, 共{len(df_quote)}条记录")
                    break
                except Exception as e:
                    print(f"读取{f}失败: {e}")
                    continue
        
        if df_quote is None or df_quote.empty:
            st.error("无法读取股票行情数据，请检查数据文件")
            return None, None
        
        # 创建行情映射
        quote_map_price = df_quote.set_index('code')['RT_LAST'].to_dict()
        quote_map_pct = df_quote.set_index('code')['RT_PCT_CHG'].to_dict() if 'RT_PCT_CHG' in df_quote.columns else {}
        
        # 2. 导入持仓期权文件
        print("读取持仓期权文件...")
        df_option = pd.read_excel(os.path.join(PATH_DATA, "全部持仓.xlsx"), sheet_name="全部")
        
        # 筛选期权结构
        df_option = df_option[df_option["结构"].isin([" 100call", " 105call", " 80call"])]
        print(f"筛选后期权记录数: {len(df_option)}")
        
        # 3. 导入波动率预测文件
        print("读取波动率预测文件...")
        df_vol_est = pd.read_excel(os.path.join(PATH_DATA, "para_option_pricing.xlsx"))
        vol_map = df_vol_est.set_index('code')['vol_esti'].to_dict()
        print(f"波动率数据记录数: {len(df_vol_est)}")
        
        # 4. 模型参数
        print("读取模型参数...")
        df_para = pd.read_excel(os.path.join(PATH_DATA, "固定参数.xlsx"), sheet_name="rKS")
        r = df_para.loc[0, "r"]
        print(f"利率 r = {r}")
        
        # -----------------------------------------------------------------
        # 步骤2: 计算每张期权的Delta值
        print("=" * 50)
        print("步骤2: 计算期权Delta等希腊字母")
        print("=" * 50)
        
        df_delta = df_option.copy()
        
        deltas, gammas, vegas, thetas = [], [], [], []
        
        for idx, row in df_delta.iterrows():
            code = row['股票代码']
            S = quote_map_price.get(code, row.get('现价', row.get('期初价', 0)))
            K = row.get('期初价', S)
            
            # 获取波动率
            sigma = vol_map.get(code, row.get('vol-调整后波动率', 0.65))
            T_days = row.get('剩余日期(天)', 0)
            
            # 计算希腊字母
            d, g, v, t = calculate_delta_values(S, K, T_days, r, sigma)
            deltas.append(d)
            gammas.append(g)
            vegas.append(v)
            thetas.append(t)

        df_delta['Delta'] = deltas
        df_delta['Gamma'] = gammas
        df_delta['Vega-波动率敏感'] = vegas
        df_delta['Theta(Θ)'] = thetas
        df_delta['最新价'] = df_delta['股票代码'].map(quote_map_price)
        df_delta['今日涨跌幅'] = df_delta['股票代码'].map(quote_map_pct)
        
        # 3. 计算市值风险：op_market_risk = delta * 名义本金
        print("计算市值风险...")
        df_delta['市值风险'] = df_delta['Delta'] * df_delta.get('名义本金', 0)
        
        # -----------------------------------------------------------------
        # 步骤4: 从df_stocks.xlsx读取股票持仓数据
        print("=" * 50)
        print("步骤4: 读取并分配股票持仓")
        print("=" * 50)
        
        try:
            stocks_file_path = os.path.join(PATH_DATA, "df_stocks.xlsx")
            if os.path.exists(stocks_file_path):
                df_stocks = pd.read_excel(stocks_file_path)
                print(f"成功读取股票持仓数据，共{len(df_stocks)}条记录")
                
                # 筛选有效记录：成本价 > 0 且持仓股票数量 > 0
                df_stocks_filtered = df_stocks[
                    (df_stocks['成本价'] > 0) & 
                    (df_stocks['持仓股票数量'] > 0)
                ].copy()
                
                print(f"筛选后有效记录: {len(df_stocks_filtered)}条")
                
                if not df_stocks_filtered.empty:
                    # 创建股票代码到持仓数据的映射
                    stock_positions = {}
                    for idx, row in df_stocks_filtered.iterrows():
                        stock_code = str(row.get('股票代码', '')).strip()
                        if stock_code:
                            stock_positions[stock_code] = {
                                '成本价': row['成本价'],
                                '持仓股票数量': row['持仓股票数量']
                            }
                    
                    print(f"股票持仓映射: {len(stock_positions)}只股票")
                    
                    # 按名义本金占比分配股票数量和成本价
                    for stock_code, position_data in stock_positions.items():
                        stock_options = df_delta[df_delta['股票代码'] == stock_code]
                        
                        if not stock_options.empty:
                            total_nominal = stock_options['名义本金'].sum()
                            if total_nominal > 0:
                                for idx in stock_options.index:
                                    nominal_ratio = df_delta.at[idx, '名义本金'] / total_nominal
                                    allocated_qty = position_data['持仓股票数量'] * nominal_ratio
                                    
                                    df_delta.at[idx, '持仓股票数量'] = allocated_qty
                                    df_delta.at[idx, '成本价'] = position_data['成本价']
                                    
                                print(f"股票 {stock_code}: 总名义本金 {total_nominal:.0f}，持仓 {position_data['持仓股票数量']}股")
            else:
                print(f"警告: 股票持仓文件不存在: {stocks_file_path}")
        except Exception as e:
            print(f"读取股票持仓数据时出错: {e}")
            st.warning(f"读取股票持仓数据时出错: {e}")
        
        # -----------------------------------------------------------------
        # 步骤5: 计算衍生指标
        print("=" * 50)
        print("步骤5: 计算衍生指标")
        print("=" * 50)
        
        # 初始化需要计算的列
        cols_to_init = ['浮盈亏%|-5%', '期权费剩余', '持仓股票市值', '需买+/卖-的市值', '已对冲比例']
        for col in cols_to_init:
            if col not in df_delta.columns:
                df_delta[col] = np.nan
        
        # 计算每个股票的市值风险总和（用于计算已对冲比例）
        market_risk_by_code = df_delta.groupby('股票代码')['市值风险'].sum().to_dict()
        
        # 计算所有期权记录的衍生指标
        for idx in df_delta.index:
            row = df_delta.loc[idx]
            code = row['股票代码']
            S = df_delta.at[idx, '最新价']
            cost = row.get('成本价', row.get('期初价', S))
            qty = row.get('持仓股票数量', 0)
            premium = row.get('期权费', 0)
            
            # 获取该股票的总市值风险
            total_market_risk = market_risk_by_code.get(code, 0)
            
            # 计算浮盈亏: (现价 - 成本价) / 成本价
            if pd.notna(cost) and cost > 0 and pd.notna(S):
                floating_pl = (S - cost) / cost
                df_delta.at[idx, '浮盈亏%|-5%'] = floating_pl
            else:
                df_delta.at[idx, '浮盈亏%|-5%'] = np.nan
            
            # 计算持仓股票市值: 持仓股票数量 * 现价
            if pd.notna(qty) and qty > 0 and pd.notna(S) and S > 0:
                stock_mv = qty * S
                df_delta.at[idx, '持仓股票市值'] = stock_mv
            else:
                df_delta.at[idx, '持仓股票市值'] = 0
            
            # 计算期权费剩余: (成本价 * 浮盈亏% * 持仓股票数量 + 期权费) / 期权费
            if pd.notna(premium) and premium > 0 and pd.notna(cost) and cost > 0:
                floating_pl_val = df_delta.at[idx, '浮盈亏%|-5%']
                if pd.notna(floating_pl_val):
                    # 期权费剩余 = (成本价 * 浮盈亏% * 持仓股票数量 + 期权费) / 期权费
                    df_delta.at[idx, '期权费剩余'] = (cost * floating_pl_val * qty + premium) / premium
                else:
                    df_delta.at[idx, '期权费剩余'] = np.nan
            else:
                df_delta.at[idx, '期权费剩余'] = np.nan
            
            # 计算需买+/卖-的市值: 同一股票的"市值风险"之和 - "持仓股票市值"
            stock_mv_val = df_delta.at[idx, '持仓股票市值']
            df_delta.at[idx, '需买+/卖-的市值'] = total_market_risk - stock_mv_val if pd.notna(stock_mv_val) else total_market_risk
            
            # 计算已对冲比例: 持仓股票市值 / 市值风险
            if total_market_risk > 0 and pd.notna(stock_mv_val) and stock_mv_val > 0:
                df_delta.at[idx, '已对冲比例'] = stock_mv_val / total_market_risk
            else:
                df_delta.at[idx, '已对冲比例'] = 0
        
        # 优化计算"需买+/卖-的市值" - 按名义本金比例分配
        nominal_sum_by_code = df_delta.groupby('股票代码')['名义本金'].sum()
        hedge_max_by_code = df_delta.groupby('股票代码')['需买+/卖-的市值'].apply(lambda x: x.abs().max())
        
        for idx in df_delta.index:
            code = df_delta.at[idx, '股票代码']
            nominal = df_delta.at[idx, '名义本金']
            nominal_sum = nominal_sum_by_code.get(code, 1)
            hedge_max = hedge_max_by_code.get(code, 0)
            
            if nominal_sum != 0:
                df_delta.at[idx, '需买+/卖-的市值_调整'] = (nominal / nominal_sum) * hedge_max
            else:
                df_delta.at[idx, '需买+/卖-的市值_调整'] = 0
        
        # 用调整后的值替换原值
        df_delta['需买+/卖-的市值'] = df_delta['需买+/卖-的市值_调整']
        df_delta.drop(columns=['需买+/卖-的市值_调整'], inplace=True)
        
        # -----------------------------------------------------------------
        # 步骤6: 筛选和提示逻辑
        print("=" * 50)
        print("步骤6: 生成预警信息")
        print("=" * 50)
        
        warm_rows = []
        for idx, row in df_delta.iterrows():
            info_list = []
            
            floating_pl = row['浮盈亏%|-5%']
            hedge_ratio = row['已对冲比例']
            delta_val = row['Delta']
            
            # 条件1: 浮盈亏小于 -5%
            if pd.notnull(floating_pl) and floating_pl < -0.05:
                info_list.append("卖出20%本金的持仓")
            
            # 条件2: 已对冲比例大于120%
            if pd.notnull(hedge_ratio) and hedge_ratio > 1.2:
                info_list.append("卖出20%市值风险的持仓")
            
            # 条件3: Delta大于0.60
            if pd.notnull(delta_val) and delta_val > 0.60:
                info_list.append("买入20%本金的持仓")
            
            # 条件4: Delta小于0.41
            if pd.notnull(delta_val) and delta_val < 0.41:
                info_list.append("卖出持仓到40%本金或以下")
                
            if info_list:
                new_row = row.copy()
                new_row['info'] = " | ".join(info_list)
                warm_rows.append(new_row)

        df_warm = pd.DataFrame(warm_rows)
        if not df_warm.empty:
            cols = ['info'] + [c for c in df_warm.columns if c != 'info']
            df_warm = df_warm[cols]
        else:
            df_warm = pd.DataFrame(columns=['info'] + list(df_delta.columns))
        
        print(f"触发预警的记录数: {len(df_warm)}")
        
        # -----------------------------------------------------------------
        # 步骤7: 调整列顺序
        df_delta = reorder_columns(df_delta)
        df_warm = reorder_columns(df_warm)
        
        # -----------------------------------------------------------------
        # 步骤8: 数据保存
        print("=" * 50)
        print("步骤8: 保存数据")
        print("=" * 50)
        
        df_warm.to_excel(os.path.join(PATH_SYNC, "df_warm.xlsx"), index=False)
        df_delta.to_excel(os.path.join(PATH_SYNC, "df_delta.xlsx"), index=False)
        
        print(f"df_delta已保存: {os.path.join(PATH_SYNC, 'df_delta.xlsx')}")
        print(f"df_warm已保存: {os.path.join(PATH_SYNC, 'df_warm.xlsx')}")
        
        return df_delta, df_warm

    except Exception as e:
        import traceback
        print(f"数据读取或计算过程中发生错误: {e}")
        print(traceback.format_exc())
        st.error(f"数据读取或计算过程中发生错误: {e}")
        return None, None


# ----------------- Streamlit 页面渲染 -----------------
st.sidebar.title("📊 期权风险管理")

page = st.sidebar.radio("选择功能", ["实时监控", "新增期权记录"], index=0)

if page == "实时监控":
    st.title("📈 场外期权及对冲股票持仓 - 实时风险监控")
    
    # 任务0: 自动下载行情数据
    print("检查是否需要更新行情数据...")
    download_quote_data()
    
    # 显示新增期权表单（折叠式）
    display_add_option_form()
    
    # 记录刷新时间
    current_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"**最后更新时间:** `{current_time}` *(系统设定每 {REFRESH_RATE} 秒自动刷新)*")
    
    with st.spinner('正在获取最新数据与计算期权指标...'):
        df_delta, df_warm = run_calculation()

    if df_delta is not None:
        # 添加页面布局CSS
        st.markdown("""
        <style>
        .main-container {
            display: flex;
            height: calc(100vh - 200px);
            overflow: hidden;
        }
        .left-directory {
            width: 25%;
            min-width: 250px;
            background-color: #f8f9fa;
            border-right: 2px solid #e9ecef;
            padding: 15px;
            overflow-y: auto;
        }
        .right-content {
            width: 75%;
            padding: 15px;
            overflow-y: auto;
            overflow-x: auto;
            background-color: #ffffff;
        }
        .table-container {
            overflow-x: auto;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            margin-bottom: 20px;
            background-color: white;
        }
        </style>
        """, unsafe_allow_html=True)
        
        ######################################################################
        # 预警表格
        st.subheader("🚨 触发预警与建议调仓列表 (df_warm)")
        if not df_warm.empty:
            df_warm_display = df_warm.copy()
            
            for idx in df_warm_display.index:
                info_value = df_warm_display.at[idx, 'info']
                styled_info = style_info_text(info_value)
                df_warm_display.at[idx, 'info_styled'] = styled_info
            
            display_cols_warm = [c for c in df_warm.columns if c != 'info']
            display_warm_df = df_warm_display[display_cols_warm].copy()
            
            format_dict = {}
            for col in display_warm_df.columns:
                if col in ["今日涨跌幅", "浮盈亏%|-5%", "已对冲比例"]:
                    format_dict[col] = lambda x: safe_format_number(x, multiply_by_100=True)
                elif col not in ["股票代码", "股票名称", "中文客户姓名", "结构", "是否人工新增", "info", "info_styled"]:
                    format_dict[col] = lambda x: safe_format_number(x)
            
            st.dataframe(
                display_warm_df.style.format(format_dict),
                use_container_width=True
            )
            
            # 单独显示带样式的info列
            st.markdown("**预警建议:**")
            for idx, row in df_warm_display.iterrows():
                stock_name = row.get('股票名称', 'N/A')
                stock_code = row.get('股票代码', 'N/A')
                info_styled = row.get('info_styled', '')
                
                nominal_amount = row.get('名义本金', 0)
                if pd.notna(nominal_amount):
                    nominal_in_wan = nominal_amount / 10000
                    nominal_text = f"{nominal_in_wan:,.0f}万"
                else:
                    nominal_text = "N/A"
                
                seq_num = idx + 1
                st.markdown(f"**{seq_num}. {stock_name} ({stock_code}) - 名义本金: {nominal_text}:** {info_styled}", unsafe_allow_html=True)
        else:
            st.success("目前没有触发预警的期权持仓。")

        ######################################################################
        # 全局监控表格
        st.subheader("📊 持仓Delta监控与分析看板 (df_delta)")
        
        display_df = df_delta.copy()
        
        styled_display = display_df.style.format({
            "今日涨跌幅": lambda x: safe_format_number(x, multiply_by_100=True),
            "浮盈亏%|-5%": lambda x: safe_format_number(x, multiply_by_100=True),
            "已对冲比例": lambda x: safe_format_number(x, multiply_by_100=True),
            "成本价": lambda x: safe_format_number(x),
            "持仓股票数量": lambda x: safe_format_number(x),
            "期权费剩余": lambda x: safe_format_number(x),
            "需买+/卖-的市值": lambda x: safe_format_number(x),
            "持仓股票市值": lambda x: safe_format_number(x),
            "市值风险": lambda x: safe_format_number(x),
            "Delta": lambda x: safe_format_number(x),
            "Gamma": lambda x: safe_format_number(x),
            "Vega-波动率敏感": lambda x: safe_format_number(x),
            "Theta(Θ)": lambda x: safe_format_number(x),
            "名义本金": lambda x: safe_format_number(x),
            "期权费": lambda x: safe_format_number(x),
            "期初价": lambda x: safe_format_number(x),
            "最新价": lambda x: safe_format_number(x),
            "剩余日期(天)": lambda x: safe_format_number(x),
            "vol-调整后波动率": lambda x: safe_format_number(x)
        }).background_gradient(cmap="RdYlGn", subset=['Delta'])
        
        st.dataframe(styled_display)
    
    # 自动刷新
    time.sleep(REFRESH_RATE)
    st.experimental_rerun()

elif page == "新增期权记录":
    st.title("➕ 新增期权记录")
    st.markdown("在此页面新增期权记录，新增后会自动纳入实时监控")
    
    display_add_option_form()
    
    st.divider()
    st.subheader("📋 当前人工新增的期权记录")
    
    try:
        file_path = os.path.join(PATH_DATA, "全部持仓.xlsx")
        if os.path.exists(file_path):
            df_all = pd.read_excel(file_path, sheet_name="全部")
            df_manual = df_all[df_all['是否人工新增'] == '是'] if '是否人工新增' in df_all.columns else pd.DataFrame()
            
            if not df_manual.empty:
                st.dataframe(df_manual, use_container_width=True)
            else:
                st.info("暂无人工新增的期权记录")
    except Exception as e:
        st.warning(f"无法读取现有记录: {e}")
