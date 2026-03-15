# -*- coding: utf-8 -*-
"""
bs_engine.py — Black-Scholes 定价引擎
=====================================
基于 option_risk_monitor 的 calculate_delta_values 扩展而来，新增：
  1. bs_price_and_greeks()   — 同时返回期权理论价格 + 全量 Greeks
  2. bs_price()              — 仅返回期权理论价格（便于回测逐日重定价）
  3. implied_vol()           — 用二分法反解隐含波动率
  4. 卖方视角 Greeks 符号翻转 — short_pos=True 时 Delta/Gamma/Vega 取反，Theta 取反(卖方 Theta > 0)
  5. 批量计算 batch_bs()     — 对 DataFrame 一次性计算全部 Greeks

Python 3.7 兼容（不使用 walrus operator、不使用内置泛型类型注解）
"""
import math
import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple, Optional

__author__ = "ruoyu.Cheng"

# ------------------------------------------------------------------
# 常量
# ------------------------------------------------------------------
DAYS_PER_YEAR = 365.0
MIN_T = 1e-6        # 最小剩余时间（年），防止除零
MIN_SIGMA = 1e-6    # 最小波动率，防止除零
MIN_PRICE = 1e-8    # 最小价格，防止 log 域错误

# ------------------------------------------------------------------
# 基础工具函数（复用 option_risk_monitor 保护逻辑）
# ------------------------------------------------------------------

def _safe_T(T_days):
    # type: (float) -> float
    """将剩余天数转为年，并保证正数"""
    T = T_days / DAYS_PER_YEAR
    return max(T, MIN_T)


def _safe_sigma(sigma):
    # type: (float) -> float
    return max(float(sigma), MIN_SIGMA)


def _d1_d2(S, K, T, r, sigma):
    # type: (float, float, float, float, float) -> Tuple[float, float]
    """计算 d1 和 d2（内部使用，已假设参数安全）"""
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


# ------------------------------------------------------------------
# 核心函数 1：bs_price_and_greeks
# ------------------------------------------------------------------

def bs_price_and_greeks(S, K, T_days, r, sigma, option_type='call', short_pos=False):
    # type: (float, float, float, float, float, str, bool) -> Dict[str, float]
    """
    Black-Scholes 定价 + 全量 Greeks（买方视角，可翻转至卖方）

    Parameters
    ----------
    S          : 标的现价
    K          : 行权价
    T_days     : 剩余天数
    r          : 无风险利率（年化，如 0.02）
    sigma      : 波动率（年化，如 0.20）
    option_type: 'call' 或 'put'
    short_pos  : True = 卖方持仓视角（Greeks 符号取反，Theta 变为正）

    Returns
    -------
    dict with keys: price, delta, gamma, vega, theta, rho
    出错时返回全零字典。
    """
    zero = {'price': 0.0, 'delta': 0.0, 'gamma': 0.0,
            'vega': 0.0, 'theta': 0.0, 'rho': 0.0}

    # 参数保护
    if S <= MIN_PRICE or K <= MIN_PRICE:
        return zero

    T = _safe_T(T_days)
    sig = _safe_sigma(sigma)
    opt = option_type.lower().strip()

    # 到期处理：直接返回内在价值，Greeks 为 0
    if T_days <= 0:
        if opt == 'call':
            price = max(S - K, 0.0)
        else:
            price = max(K - S, 0.0)
        res = dict(zero)
        res['price'] = price
        return res

    try:
        d1, d2 = _d1_d2(S, K, T, r, sig)

        # --- 期权价格 ---
        discount = math.exp(-r * T)
        if opt == 'call':
            price = S * norm.cdf(d1) - K * discount * norm.cdf(d2)
        else:
            price = K * discount * norm.cdf(-d2) - S * norm.cdf(-d1)

        # --- Delta ---
        if opt == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1.0          # 负数，put delta ∈ [-1, 0]

        # --- Gamma（call == put）---
        gamma = norm.pdf(d1) / (S * sig * math.sqrt(T))

        # --- Vega（对 sigma 的一阶偏导，除以 100 表示 vol 变动 1%）---
        vega = S * norm.pdf(d1) * math.sqrt(T) / 100.0

        # --- Theta（日衰减，年化值除以 365）---
        common_theta = -(S * norm.pdf(d1) * sig) / (2.0 * math.sqrt(T))
        if opt == 'call':
            theta = (common_theta - r * K * discount * norm.cdf(d2)) / DAYS_PER_YEAR
        else:
            theta = (common_theta + r * K * discount * norm.cdf(-d2)) / DAYS_PER_YEAR

        # --- Rho ---
        if opt == 'call':
            rho = K * T * discount * norm.cdf(d2) / 100.0
        else:
            rho = -K * T * discount * norm.cdf(-d2) / 100.0

        result = {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'vega':  vega,
            'theta': theta,
            'rho':   rho,
        }

        # --- 卖方视角符号翻转 ---
        # 卖出期权：Delta/Gamma/Vega/Rho 变负，Theta 变正（时间流逝收益）
        if short_pos:
            result['delta'] = -delta
            result['gamma'] = -gamma
            result['vega']  = -vega
            result['theta'] = -theta   # -(-|theta|) > 0，卖方 Theta 为正
            result['rho']   = -rho

        return result

    except Exception as e:
        print("bs_price_and_greeks error: {}".format(e))
        return zero


# ------------------------------------------------------------------
# 核心函数 2：bs_price（轻量版，仅价格，适合高频回测循环）
# ------------------------------------------------------------------

def bs_price(S, K, T_days, r, sigma, option_type='call'):
    # type: (float, float, float, float, float, str) -> float
    """
    仅返回 BS 理论价格，速度更快，供回测逐日重定价使用。
    到期时（T_days <= 0）返回内在价值。
    """
    if S <= MIN_PRICE or K <= MIN_PRICE:
        return 0.0

    if T_days <= 0:
        opt = option_type.lower().strip()
        return max(S - K, 0.0) if opt == 'call' else max(K - S, 0.0)

    T = _safe_T(T_days)
    sig = _safe_sigma(sigma)
    opt = option_type.lower().strip()

    try:
        d1, d2 = _d1_d2(S, K, T, r, sig)
        discount = math.exp(-r * T)
        if opt == 'call':
            return S * norm.cdf(d1) - K * discount * norm.cdf(d2)
        else:
            return K * discount * norm.cdf(-d2) - S * norm.cdf(-d1)
    except Exception as e:
        print("bs_price error: {}".format(e))
        return 0.0


# ------------------------------------------------------------------
# 核心函数 3：calculate_delta_values（向后兼容 option_risk_monitor）
# ------------------------------------------------------------------

def calculate_delta_values(S, K, T_days, r, sigma, option_type='call'):
    # type: (float, float, float, float, float, str) -> Tuple[float, float, float, float]
    """
    兼容 option_risk_monitor 的原始接口。
    返回 (delta, gamma, vega, theta)，与原始实现完全一致。
    """
    if T_days <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0, 0.0

    T = T_days / DAYS_PER_YEAR
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0, 0.0, 0.0

    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1

        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega  = S * norm.pdf(d1) * np.sqrt(T) / 100
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / DAYS_PER_YEAR

        return delta, gamma, vega, theta
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


# ------------------------------------------------------------------
# 核心函数 4：implied_vol（二分法反解 IV）
# ------------------------------------------------------------------

def implied_vol(market_price, S, K, T_days, r, option_type='call',
                sigma_low=0.001, sigma_high=5.0, tol=1e-6, max_iter=200):
    # type: (float, float, float, float, float, str, float, float, float, int) -> Optional[float]
    """
    用二分法从市场价格反解隐含波动率。

    Parameters
    ----------
    market_price : 期权市场价格
    sigma_low    : 搜索下界（默认 0.001）
    sigma_high   : 搜索上界（默认 5.0 = 500%）
    tol          : 收敛精度（默认 1e-6）
    max_iter     : 最大迭代次数（默认 200）

    Returns
    -------
    float  : 隐含波动率（年化小数），收敛失败返回 None
    """
    if T_days <= 0 or market_price <= 0:
        return None

    # 快速检查：市场价是否在理论范围内
    intrinsic = max(S - K, 0.0) if option_type.lower() == 'call' else max(K - S, 0.0)
    if market_price < intrinsic * 0.999:   # 允许 0.1% 误差
        return None

    f_low  = bs_price(S, K, T_days, r, sigma_low,  option_type) - market_price
    f_high = bs_price(S, K, T_days, r, sigma_high, option_type) - market_price

    # 确认根存在
    if f_low * f_high > 0:
        return None

    for _ in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2.0
        f_mid = bs_price(S, K, T_days, r, sigma_mid, option_type) - market_price
        if abs(f_mid) < tol or (sigma_high - sigma_low) / 2.0 < tol:
            return sigma_mid
        if f_low * f_mid < 0:
            sigma_high = sigma_mid
            f_high = f_mid
        else:
            sigma_low = sigma_mid
            f_low = f_mid

    return (sigma_low + sigma_high) / 2.0   # 返回最终中值


# ------------------------------------------------------------------
# 核心函数 5：batch_bs（DataFrame 批量计算，供回测/监控调用）
# ------------------------------------------------------------------

def batch_bs(df, S_col='S', K_col='strike_price', T_col='T_days',
             r=0.02, sigma_col='iv', type_col='call_put',
             short_pos=True, multiplier=10000):
    # type: (object, str, str, str, float, str, str, bool, int) -> object
    """
    对 DataFrame 批量计算 BS 价格和 Greeks，原地添加列。

    新增列（前缀可自定义）：
      bs_price, delta, gamma, vega, theta, rho

    Parameters
    ----------
    df         : pandas DataFrame，含期权列表
    S_col      : 标的现价列名
    K_col      : 行权价列名
    T_col      : 剩余天数列名
    r          : 无风险利率（标量）
    sigma_col  : 隐含波动率列名
    type_col   : call/put 类型列名（值应为 'C'/'P' 或 'call'/'put'）
    short_pos  : True = 卖方视角 Greeks 翻转
    multiplier : 合约乘数（用于计算名义 Greeks 敞口，默认 1 万份/张）

    Returns
    -------
    df with added columns（原 df 不被修改，返回副本）
    """
    import pandas as pd

    result = df.copy()
    prices, deltas, gammas, vegas, thetas, rhos = [], [], [], [], [], []

    for _, row in result.iterrows():
        S = float(row.get(S_col, 0) or 0)
        K = float(row.get(K_col, 0) or 0)
        T = float(row.get(T_col, 0) or 0)
        sig = float(row.get(sigma_col, 0.3) or 0.3)

        # 规范化 call/put 标识
        raw_type = str(row.get(type_col, 'call')).strip().upper()
        if raw_type in ('C', 'CALL', '认购'):
            opt_type = 'call'
        else:
            opt_type = 'put'

        g = bs_price_and_greeks(S, K, T, r, sig, opt_type, short_pos=short_pos)
        prices.append(g['price'])
        deltas.append(g['delta'])
        gammas.append(g['gamma'])
        vegas.append(g['vega'])
        thetas.append(g['theta'])
        rhos.append(g['rho'])

    result['bs_price'] = prices
    result['delta']    = deltas
    result['gamma']    = gammas
    result['vega']     = vegas
    result['theta']    = thetas
    result['rho']      = rhos

    return result


# ------------------------------------------------------------------
# 工具函数：组合 Greeks 汇总（用于实时监控页面）
# ------------------------------------------------------------------

def portfolio_greeks(df, qty_col='contracts', multiplier=10000):
    # type: (object, str, int) -> Dict[str, float]
    """
    汇总持仓 DataFrame 的 Net Greeks（已假设 batch_bs 已运行）。

    Parameters
    ----------
    df         : 含 delta/gamma/vega/theta 列的 DataFrame
    qty_col    : 合约张数列名（默认 'contracts'）
    multiplier : 合约乘数

    Returns
    -------
    dict: net_delta, net_gamma, net_vega, net_theta, net_rho
    """
    result = {}
    import pandas as pd

    for greek in ('delta', 'gamma', 'vega', 'theta', 'rho'):
        if greek in df.columns:
            if qty_col in df.columns:
                net = (df[greek] * df[qty_col] * multiplier).sum()
            else:
                net = df[greek].sum()
            result['net_' + greek] = float(net)
        else:
            result['net_' + greek] = 0.0

    return result


# ------------------------------------------------------------------
# 工具函数：到期结算损益
# ------------------------------------------------------------------

def settlement_pnl(S_expiry, K, open_price, option_type='call',
                   short_pos=True, contracts=1, multiplier=10000):
    # type: (float, float, float, str, bool, int, int) -> float
    """
    计算到期结算损益（针对卖方）。

    卖方到期 P&L = (开仓权利金 - 到期内在价值) * contracts * multiplier

    Parameters
    ----------
    S_expiry   : 到期日标的收盘价
    K          : 行权价
    open_price : 开仓时收取的权利金（每单位）
    short_pos  : True 表示卖方（默认）
    contracts  : 合约张数
    multiplier : 合约乘数（默认 1 万份/张）

    Returns
    -------
    float : 损益金额（正数为盈利）
    """
    opt = option_type.lower().strip()
    intrinsic = max(S_expiry - K, 0.0) if opt == 'call' else max(K - S_expiry, 0.0)

    if short_pos:
        pnl_per_unit = open_price - intrinsic   # 卖出：收取权利金 - 被行权损失
    else:
        pnl_per_unit = intrinsic - open_price   # 买入：到期价值 - 支付权利金

    return pnl_per_unit * contracts * multiplier


# ------------------------------------------------------------------
# __main__ 自测
# ------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("bs_engine.py 自测")
    print("=" * 60)

    S, K, T_days, r, sigma = 5.0, 5.0, 30, 0.02, 0.20

    # 1. 买方 Call
    g_call = bs_price_and_greeks(S, K, T_days, r, sigma, 'call', short_pos=False)
    print("\n[买方 Call ATM, 30天, sigma=0.20]")
    for k, v in g_call.items():
        print("  {:8s}: {:.6f}".format(k, v))

    # 2. 卖方 Call（符号翻转）
    g_call_short = bs_price_and_greeks(S, K, T_days, r, sigma, 'call', short_pos=True)
    print("\n[卖方 Call ATM（short_pos=True）]")
    for k, v in g_call_short.items():
        print("  {:8s}: {:.6f}".format(k, v))

    # 3. 卖方 Put
    g_put_short = bs_price_and_greeks(S, K * 0.95, T_days, r, sigma, 'put', short_pos=True)
    print("\n[卖方 Put OTM K=4.75]")
    for k, v in g_put_short.items():
        print("  {:8s}: {:.6f}".format(k, v))

    # 4. 隐含波动率反解
    market_px = g_call['price']
    iv = implied_vol(market_px, S, K, T_days, r, 'call')
    print("\n[IV 反解] market_price={:.6f} => IV={:.4%}".format(market_px, iv if iv else 0))

    # 5. 向后兼容接口
    d, g, v, t = calculate_delta_values(S, K, T_days, r, sigma, 'call')
    print("\n[兼容接口] delta={:.4f} gamma={:.4f} vega={:.4f} theta={:.6f}".format(d, g, v, t))

    # 6. 到期结算
    pnl = settlement_pnl(5.3, K, g_call['price'], 'call', short_pos=True, contracts=1, multiplier=10000)
    print("\n[到期结算 call, S_expiry=5.3] P&L = {:.2f}".format(pnl))

    print("\n[OK] 自测完成")
