# -*- coding: utf-8 -*-
"""
data_loader.py
数据加载器：手动 Excel 解析 + akshare 多标的期权链/日线自动拉取
兼容 Python 3.7
"""

import io
import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─── 标的映射表 ─────────────────────────────────────────────────────────────
# key: 显示名称  value: (akshare ETF symbol, akshare option symbol)
UNDERLYING_MAP = {
    "50ETF (510050)":       {"etf": "510050", "opt": "50ETF",    "name": "上证50ETF"},
    "沪深300ETF (510300)":  {"etf": "510300", "opt": "300ETF",   "name": "沪深300ETF"},
    "中证1000ETF (159922)": {"etf": "159922", "opt": "1000ETF",  "name": "中证1000ETF"},
    "科创50 (588000)":      {"etf": "588000", "opt": "科创50ETF","name": "科创50ETF"},
}

# 必填列及类型映射
REQUIRED_COLS = ["ts_code", "strike_price", "call_put", "exp_date", "open_price", "iv"]
OPTIONAL_COLS = {"multiplier": 10000, "quantity": 1}

# call_put 标准化映射
CP_MAP = {
    "c": "C", "call": "C", "认购": "C", "C": "C",
    "p": "P", "put":  "P", "认沽": "P", "P": "P",
}


# ─── 手动 Excel 加载 ────────────────────────────────────────────────────────

def load_options_from_excel(file_obj) -> Tuple[Optional[pd.DataFrame], str]:
    """
    从上传的 Excel / BytesIO 对象解析期权列表。
    返回 (DataFrame, error_msg)，成功时 error_msg=""
    """
    try:
        df = pd.read_excel(file_obj)
    except Exception as e:
        return None, "读取 Excel 失败：{}".format(e)

    # 列名归一化（去空格、转小写后匹配）
    df.columns = [c.strip() for c in df.columns]
    col_lower = {c.lower(): c for c in df.columns}
    rename = {}
    for req in REQUIRED_COLS + list(OPTIONAL_COLS.keys()):
        if req.lower() in col_lower and col_lower[req.lower()] != req:
            rename[col_lower[req.lower()]] = req
    if rename:
        df = df.rename(columns=rename)

    # 检查必填列
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return None, "缺少必填列：{}".format(missing)

    # 补充可选列默认值
    for col, default in OPTIONAL_COLS.items():
        if col not in df.columns:
            df[col] = default

    # 数据清洗
    df = df.dropna(subset=["strike_price", "open_price"])
    df["call_put"] = df["call_put"].astype(str).str.strip().map(
        lambda x: CP_MAP.get(x, CP_MAP.get(x.lower(), x))
    )
    df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce")
    df["open_price"]   = pd.to_numeric(df["open_price"],   errors="coerce")
    df["iv"]           = pd.to_numeric(df["iv"],           errors="coerce")

    # exp_date 统一为 date 类型
    df["exp_date"] = pd.to_datetime(df["exp_date"], errors="coerce").dt.date

    invalid = df["exp_date"].isna().sum()
    if invalid > 0:
        return None, "exp_date 列含 {} 个无效日期，请检查格式（YYYY-MM-DD）".format(invalid)

    df = df.dropna(subset=["strike_price", "open_price", "iv"])
    df = df.reset_index(drop=True)
    return df, ""


def load_price_path_from_excel(file_obj) -> Tuple[Optional[pd.DataFrame], str]:
    """
    从上传的 Excel 解析标的价格路径。
    期望列：date（日期），close（收盘价）。
    返回 (DataFrame[date, close], error_msg)
    """
    try:
        df = pd.read_excel(file_obj)
    except Exception as e:
        return None, "读取价格 Excel 失败：{}".format(e)

    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "close" not in df.columns:
        # 尝试用第一列作为 date，第二列作为 close
        if df.shape[1] >= 2:
            df.columns = ["date", "close"] + list(df.columns[2:])
        else:
            return None, "价格文件需包含 date、close 两列"

    df["date"]  = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return df[["date", "close"]], ""


# ─── akshare 自动拉取 ───────────────────────────────────────────────────────

def _check_akshare() -> bool:
    """检查 akshare 是否可用"""
    try:
        import akshare as ak  # noqa
        return True
    except ImportError:
        return False


def load_spot_from_akshare(etf_symbol: str, days: int = 60) -> Tuple[Optional[pd.DataFrame], str]:
    """
    通过 akshare 拉取 ETF 日线行情，返回近 days 天数据。
    返回 DataFrame[date, open, high, low, close, volume] 或 (None, err)
    """
    if not _check_akshare():
        return None, "akshare 未安装，请 pip install akshare"
    try:
        import akshare as ak
        end_date   = datetime.date.today().strftime("%Y%m%d")
        start_date = (datetime.date.today() - datetime.timedelta(days=days)).strftime("%Y%m%d")
        df = ak.fund_etf_hist_em(
            symbol=etf_symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
        if df is None or df.empty:
            return None, "akshare 返回空数据，symbol={}".format(etf_symbol)
        # 列名归一化
        col_map = {}
        for c in df.columns:
            cl = c.strip().lower()
            if cl in ("日期", "date", "交易日期"):
                col_map[c] = "date"
            elif cl in ("收盘", "close", "收盘价"):
                col_map[c] = "close"
            elif cl in ("开盘", "open", "开盘价"):
                col_map[c] = "open"
        df = df.rename(columns=col_map)
        if "date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "date"})
        if "close" not in df.columns:
            # 尝试找数值列
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                df["close"] = df[num_cols[0]]
            else:
                return None, "无法识别 close 列"
        df["date"]  = pd.to_datetime(df["date"], errors="coerce").dt.date
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        return df, ""
    except Exception as e:
        return None, "akshare 拉取日线失败：{}".format(e)


def _parse_spot_price_df(df_kv: pd.DataFrame, code: str, call_put: str) -> Optional[dict]:
    """
    将 option_sse_spot_price_sina 返回的竖表（字段/值）解析为一行期权记录。
    df_kv: 43行 x 2列（字段, 值）
    """
    try:
        kv = dict(zip(df_kv["字段"].astype(str).str.strip(),
                      df_kv["值"].astype(str).str.strip()))
        strike = pd.to_numeric(kv.get("行权价", ""), errors="coerce")
        price  = pd.to_numeric(kv.get("最新价", ""), errors="coerce")
        name   = kv.get("期权合约简称", "")

        # 从合约简称解析到期日，格式如 "50ETF购3月2630A" → 当年3月
        exp_date = None
        import re
        m = re.search(r"(\d{1,2})月", name)
        if m:
            month = int(m.group(1))
            year  = datetime.date.today().year
            if month < datetime.date.today().month:
                year += 1
            # 上交所期权到期日为当月第四个星期三，近似用28日
            try:
                exp_date = datetime.date(year, month, 28)
            except Exception:
                exp_date = datetime.date(year, month, 1)

        if exp_date is None:
            exp_date = datetime.date.today().replace(day=28)

        return {
            "ts_code":      code,
            "strike_price": strike,
            "call_put":     call_put,
            "exp_date":     exp_date,
            "open_price":   price,
            "iv":           np.nan,
            "multiplier":   10000,
            "name":         name,
        }
    except Exception:
        return None


def load_option_chain_from_akshare(opt_symbol: str, etf_symbol: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    通过 akshare 拉取上交所 ETF 期权链（最近到期月全部合约）。
    返回标准化 DataFrame 或 (None, err)。
    标准化列：ts_code, strike_price, call_put, exp_date, open_price, iv, multiplier

    API 链路（akshare 1.10+）：
      1. option_sse_list_sina(symbol, exchange)        → 到期月列表
      2. option_sse_codes_sina(symbol, trade_date, underlying) → 合约代码列表
      3. option_sse_spot_price_sina(symbol=code)       → 单合约竖表报价
    """
    if not _check_akshare():
        return None, "akshare 未安装，请 pip install akshare"

    try:
        import akshare as ak

        # ── step1: 获取最近到期月 ──────────────────────────────────────────
        months = ak.option_sse_list_sina(symbol=opt_symbol, exchange="null")
        if not months:
            return None, "option_sse_list_sina 返回空，symbol={}".format(opt_symbol)
        nearest_month = sorted(months)[0]

        # ── step2: 获取认购/认沽合约代码 ──────────────────────────────────
        rows = []
        errors = []
        for cp_name, cp_val in [("看涨期权", "C"), ("看跌期权", "P")]:
            try:
                df_codes = ak.option_sse_codes_sina(
                    symbol=cp_name,
                    trade_date=nearest_month,
                    underlying=etf_symbol,
                )
                if df_codes is None or df_codes.empty:
                    errors.append("{} 合约列表为空".format(cp_name))
                    continue
                codes = df_codes["期权代码"].astype(str).tolist()

                # ── step3: 逐合约拉取报价 ──────────────────────────────────
                for code in codes:
                    try:
                        df_kv = ak.option_sse_spot_price_sina(symbol=code)
                        if df_kv is None or df_kv.empty:
                            continue
                        row = _parse_spot_price_df(df_kv, code, cp_val)
                        if row is not None:
                            rows.append(row)
                    except Exception as e:
                        errors.append("code {}: {}".format(code, e))
            except Exception as e:
                errors.append("{}: {}".format(cp_name, e))

        if not rows:
            err_msg = " | ".join(errors) if errors else "未获取到任何合约数据"
            return None, err_msg

        df = pd.DataFrame(rows)

        # ── 数值清洗 ────────────────────────────────────────────────────────
        df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce")
        df["open_price"]   = pd.to_numeric(df["open_price"],   errors="coerce")

        keep = [c for c in REQUIRED_COLS + ["multiplier", "name"] if c in df.columns]
        df = df[keep].dropna(subset=["strike_price", "open_price"]).reset_index(drop=True)

        if df.empty:
            return None, "期权链解析后无有效数据（行权价或最新价为空）"

        return df, ""

    except Exception as e:
        return None, "akshare 拉取期权链失败：{}".format(e)


def get_latest_spot(etf_symbol: str) -> Tuple[Optional[float], str]:
    """获取 ETF 最新价（用于实时监控）"""
    if not _check_akshare():
        return None, "akshare 未安装"
    try:
        import akshare as ak
        df, err = load_spot_from_akshare(etf_symbol, days=5)
        if df is None or df.empty:
            return None, err
        return float(df["close"].iloc[-1]), ""
    except Exception as e:
        return None, str(e)


# ─── Excel 模板生成 ─────────────────────────────────────────────────────────

def build_template_df() -> pd.DataFrame:
    """返回期权列表模板 DataFrame（含说明行 + 示例数据）"""
    today = datetime.date.today()
    exp_date = today.replace(day=28)  # 示例到期日
    rows = [
        {
            "ts_code":      "510050",
            "strike_price": 3.0,
            "call_put":     "C",
            "exp_date":     exp_date.strftime("%Y-%m-%d"),
            "open_price":   0.0300,
            "iv":           0.20,
            "multiplier":   10000,
            "quantity":     1,
            "note":         "示例：卖出50ETF认购",
        },
        {
            "ts_code":      "510050",
            "strike_price": 2.7,
            "call_put":     "P",
            "exp_date":     exp_date.strftime("%Y-%m-%d"),
            "open_price":   0.0250,
            "iv":           0.22,
            "multiplier":   10000,
            "quantity":     1,
            "note":         "示例：卖出50ETF认沽",
        },
    ]
    return pd.DataFrame(rows)


# ─── 持仓快照保存/读取 ──────────────────────────────────────────────────────

def save_positions(df: pd.DataFrame, path: str) -> str:
    """保存持仓快照到 Excel，返回错误信息（空字符串=成功）"""
    try:
        df.to_excel(path, index=False)
        return ""
    except Exception as e:
        return str(e)


def load_positions(path: str) -> Tuple[Optional[pd.DataFrame], str]:
    """从 Excel 读取持仓快照"""
    try:
        df = pd.read_excel(path)
        return df, ""
    except Exception as e:
        return None, str(e)


# ─── 自测 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== data_loader 自测 ===")

    # 测试模板生成
    tmpl = build_template_df()
    print("模板 DataFrame:")
    print(tmpl.to_string())

    # 测试 Excel 解析（写到 BytesIO）
    buf = io.BytesIO()
    tmpl.to_excel(buf, index=False)
    buf.seek(0)
    df_loaded, err = load_options_from_excel(buf)
    if err:
        print("Excel 解析失败:", err)
    else:
        print("\n解析结果 call_put:", df_loaded["call_put"].tolist())
        assert list(df_loaded["call_put"]) == ["C", "P"], "call_put 映射错误"
        print("[OK] Excel 手动加载测试通过")

    # akshare 可用性测试
    if _check_akshare():
        print("\nakshare 可用，尝试拉取 50ETF 最新价...")
        spot, err = get_latest_spot("510050")
        if err:
            print("拉取失败:", err)
        else:
            print("50ETF 最新价: {:.4f}".format(spot))
    else:
        print("\nakshare 未安装，跳过自动拉取测试")

    print("\n[OK] data_loader 自测完成")
