"""
CTA策略回测框架
支持: 股票指数期货、ETF、股票、商品期货
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class Direction(Enum):
    """交易方向"""
    LONG = 1      # 做多
    SHORT = -1    # 做空
    CLOSE = 0     # 平仓


class OrderType(Enum):
    """订单类型"""
    MARKET = 'market'      # 市价单
    LIMIT = 'limit'        # 限价单


@dataclass
class Signal:
    """交易信号"""
    symbol: str
    direction: Direction
    order_type: OrderType
    price: float = 0.0
    volume: float = 0.0
    strength: float = 1.0  # 信号强度 0-1


@dataclass
class Position:
    """持仓"""
    symbol: str
    direction: Direction
    volume: float
    entry_price: float
    entry_date: datetime
    stop_loss: float = 0.0
    take_profit: float = 0.0


@dataclass
class Trade:
    """成交记录（完整配对，含盈亏）"""
    open_date: datetime
    close_date: datetime
    symbol: str
    direction: Direction
    entry_price: float
    exit_price: float
    volume: float
    pnl: float          # 净盈亏（已扣手续费）
    commission: float


class BaseStrategy:
    """策略基类"""

    def __init__(self, name: str, params: Dict = None):
        self.name = name
        self.params = params or {}
        self.positions: Dict[str, Position] = {}
        self.signals: List[Signal] = []

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """生成交易信号，子类实现"""
        raise NotImplementedError

    def on_bar(self, df: pd.DataFrame, current_idx: int) -> List[Signal]:
        """逐K线调用"""
        return self.generate_signals(df.iloc[:current_idx + 1])


# ==================== 趋势跟踪策略 ====================

class DualMovingAverageStrategy(BaseStrategy):
    """双均线策略"""

    def __init__(self, short_window: int = 5, long_window: int = 20, params: Dict = None):
        super().__init__("双均线", params)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        if len(df) < self.long_window:
            return []

        close = df['close'].values
        ma_short = pd.Series(close).rolling(self.short_window).mean().values
        ma_long = pd.Series(close).rolling(self.long_window).mean().values

        signals = []
        if len(ma_short) >= 2 and len(ma_long) >= 2:
            if ma_short[-1] > ma_long[-1] and ma_short[-2] <= ma_long[-2]:
                signals.append(Signal(
                    symbol=df['symbol'].iloc[-1],
                    direction=Direction.LONG,
                    order_type=OrderType.MARKET,
                    strength=1.0
                ))
            elif ma_short[-1] < ma_long[-1] and ma_short[-2] >= ma_long[-2]:
                signals.append(Signal(
                    symbol=df['symbol'].iloc[-1],
                    direction=Direction.SHORT,
                    order_type=OrderType.MARKET,
                    strength=1.0
                ))
        return signals


class TurtleStrategy(BaseStrategy):
    """海龟交易法则 - 唐奇安通道突破"""

    def __init__(self,
                 entry_period: int = 20,
                 exit_period: int = 10,
                 atr_period: int = 14,
                 atr_multiplier: float = 2.0,
                 params: Dict = None):
        super().__init__("海龟交易", params)
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def _calculate_atr(self, df: pd.DataFrame) -> np.ndarray:
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))

        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = 0
        atr = pd.Series(tr).rolling(self.atr_period).mean().values
        return atr

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        if len(df) < self.entry_period:
            return []

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        signals = []
        entry_high = pd.Series(high).rolling(self.entry_period).max().values
        entry_low = pd.Series(low).rolling(self.entry_period).min().values
        exit_high = pd.Series(high).rolling(self.exit_period).max().values
        exit_low = pd.Series(low).rolling(self.exit_period).min().values

        if len(close) >= 2:
            if close[-1] > entry_high[-2] and close[-2] <= entry_high[-2]:
                signals.append(Signal(
                    symbol=df['symbol'].iloc[-1],
                    direction=Direction.LONG,
                    order_type=OrderType.MARKET,
                    strength=1.0
                ))
            elif close[-1] < exit_low[-2] and close[-2] >= exit_low[-2]:
                signals.append(Signal(
                    symbol=df['symbol'].iloc[-1],
                    direction=Direction.CLOSE,
                    order_type=OrderType.MARKET,
                    strength=1.0
                ))
            elif close[-1] < entry_low[-2] and close[-2] >= entry_low[-2]:
                signals.append(Signal(
                    symbol=df['symbol'].iloc[-1],
                    direction=Direction.SHORT,
                    order_type=OrderType.MARKET,
                    strength=1.0
                ))
            elif close[-1] > exit_high[-2] and close[-2] <= exit_high[-2]:
                signals.append(Signal(
                    symbol=df['symbol'].iloc[-1],
                    direction=Direction.CLOSE,
                    order_type=OrderType.MARKET,
                    strength=1.0
                ))
        return signals


class BollingerBandStrategy(BaseStrategy):
    """布林带突破策略"""

    def __init__(self, window: int = 20, num_std: float = 2.0, params: Dict = None):
        super().__init__("布林带", params)
        self.window = window
        self.num_std = num_std

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        if len(df) < self.window:
            return []

        close = df['close'].values
        ma = pd.Series(close).rolling(self.window).mean().values
        std = pd.Series(close).rolling(self.window).std().values

        upper_band = ma + std * self.num_std
        lower_band = ma - std * self.num_std

        signals = []
        if len(close) >= 2:
            if close[-1] > upper_band[-1] and close[-2] <= upper_band[-2]:
                signals.append(Signal(
                    symbol=df['symbol'].iloc[-1],
                    direction=Direction.LONG,
                    order_type=OrderType.MARKET,
                    strength=1.0
                ))
            elif close[-1] < lower_band[-1] and close[-2] >= lower_band[-2]:
                signals.append(Signal(
                    symbol=df['symbol'].iloc[-1],
                    direction=Direction.SHORT,
                    order_type=OrderType.MARKET,
                    strength=1.0
                ))
        return signals


class MACDStrategy(BaseStrategy):
    """MACD策略"""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, params: Dict = None):
        super().__init__("MACD", params)
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        if len(df) < self.slow:
            return []

        close = df['close'].values
        ema_fast = pd.Series(close).ewm(span=self.fast, adjust=False).mean().values
        ema_slow = pd.Series(close).ewm(span=self.slow, adjust=False).mean().values

        macd = ema_fast - ema_slow
        signal_line = pd.Series(macd).ewm(span=self.signal, adjust=False).mean().values

        signals = []
        if len(macd) >= 2:
            if macd[-1] > signal_line[-1] and macd[-2] <= signal_line[-2]:
                signals.append(Signal(
                    symbol=df['symbol'].iloc[-1],
                    direction=Direction.LONG,
                    order_type=OrderType.MARKET,
                    strength=1.0
                ))
            elif macd[-1] < signal_line[-1] and macd[-2] >= signal_line[-2]:
                signals.append(Signal(
                    symbol=df['symbol'].iloc[-1],
                    direction=Direction.SHORT,
                    order_type=OrderType.MARKET,
                    strength=1.0
                ))
        return signals


class RSIStrategy(BaseStrategy):
    """RSI策略"""

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70, params: Dict = None):
        super().__init__("RSI", params)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        if len(df) < self.period + 1:
            return []

        close = df['close'].values
        delta = np.diff(close)

        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain).rolling(self.period).mean().values
        avg_loss = pd.Series(loss).rolling(self.period).mean().values

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        signals = []
        if len(rsi) >= 2:
            if rsi[-1] > self.oversold and rsi[-2] <= self.oversold:
                signals.append(Signal(
                    symbol=df['symbol'].iloc[-1],
                    direction=Direction.LONG,
                    order_type=OrderType.MARKET,
                    strength=1.0
                ))
            elif rsi[-1] < self.overbought and rsi[-2] >= self.overbought:
                signals.append(Signal(
                    symbol=df['symbol'].iloc[-1],
                    direction=Direction.SHORT,
                    order_type=OrderType.MARKET,
                    strength=1.0
                ))
        return signals


# ==================== 仓位管理 ====================

class PositionSizer:
    """仓位管理器 - 按资金比例分配每笔交易头寸"""

    def __init__(self,
                 method: str = 'fixed_fraction',
                 fraction: float = 0.1,
                 max_positions: int = 10,
                 atr_multiplier: float = 2.0):
        """
        Args:
            method: 仓位方法
                'fixed_fraction'  - 固定比例（每笔占总资金fraction）
                'equal_weight'    - 等权重（总资金 / max_positions）
                'atr_based'       - ATR波动率仓位
            fraction: 每笔交易占用资金比例（fixed_fraction模式）
            max_positions: 最大同时持仓数
            atr_multiplier: ATR倍数（atr_based模式）
        """
        self.method = method
        self.fraction = fraction
        self.max_positions = max_positions
        self.atr_multiplier = atr_multiplier

    def calculate_volume(self,
                         capital: float,
                         price: float,
                         current_positions: int,
                         atr: float = None) -> float:
        """计算开仓手数（以价值/股数为单位）"""
        if price <= 0:
            return 0.0

        if self.method == 'fixed_fraction':
            position_value = capital * self.fraction
        elif self.method == 'equal_weight':
            position_value = capital / max(self.max_positions, 1)
        elif self.method == 'atr_based' and atr and atr > 0:
            risk_amount = capital * self.fraction
            position_value = risk_amount / (atr * self.atr_multiplier) * price
        else:
            position_value = capital * self.fraction

        volume = position_value / price
        return max(round(volume, 2), 0.0)


# ==================== 策略组合管理器 ====================

class StrategyPortfolio:
    """多策略组合管理器"""

    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.strategies: List[BaseStrategy] = []
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []  # (date, equity)
        self.open_orders: Dict[str, Position] = {}  # 未平仓持仓（按symbol）

    def add_strategy(self, strategy: BaseStrategy):
        self.strategies.append(strategy)

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        all_signals = []
        for strategy in self.strategies:
            signals = strategy.generate_signals(df)
            all_signals.extend(signals)
        return all_signals

    def get_total_equity(self, current_prices: Dict[str, float] = None) -> float:
        """计算当前总权益（资金 + 持仓市值）"""
        equity = self.capital
        if current_prices:
            for symbol, pos in self.positions.items():
                price = current_prices.get(symbol, pos.entry_price)
                if pos.direction == Direction.LONG:
                    equity += (price - pos.entry_price) * pos.volume
                else:
                    equity += (pos.entry_price - price) * pos.volume
        return equity


# ==================== 回测引擎 ====================

class BacktestEngine:
    """回测引擎"""

    def __init__(self,
                 initial_capital: float = 1_000_000,
                 commission_rate: float = 0.0003,
                 slippage: float = 0.0001,
                 position_method: str = 'fixed_fraction',
                 position_fraction: float = 0.1,
                 max_positions: int = 10):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.portfolio = StrategyPortfolio(initial_capital)
        self.sizer = PositionSizer(
            method=position_method,
            fraction=position_fraction,
            max_positions=max_positions
        )
        self._pending_open: Dict[str, Position] = {}  # 未平仓，用于配对

    def add_strategy(self, strategy: BaseStrategy):
        self.portfolio.add_strategy(strategy)

    def run(self, data: Dict[str, pd.DataFrame],
            start_date=None,
            end_date=None) -> Dict:
        """
        运行回测
        Args:
            data: {symbol: DataFrame}，DataFrame 必须含列 date/open/high/low/close/volume/symbol
        """
        # 收集所有交易日期
        all_dates = set()
        for df in data.values():
            if 'date' in df.columns:
                all_dates.update(pd.to_datetime(df['date']).values)

        dates = sorted(list(all_dates))
        if start_date:
            dates = [d for d in dates if d >= pd.Timestamp(start_date)]
        if end_date:
            dates = [d for d in dates if d <= pd.Timestamp(end_date)]

        for date in dates:
            current_prices: Dict[str, float] = {}

            for symbol, df in data.items():
                df = df.copy()
                df['date'] = pd.to_datetime(df['date'])
                df_slice = df[df['date'] <= date].copy()
                if df_slice.empty:
                    continue

                current_bar = df_slice.iloc[-1]
                current_prices[symbol] = current_bar['close']

                signals = self.portfolio.generate_signals(df_slice)
                self._execute_signals(signals, current_bar, date)

            # 记录每日净值
            equity = self.portfolio.get_total_equity(current_prices)
            self.portfolio.equity_curve.append((date, equity))

        # 回测结束，强制平掉剩余持仓（用最后一根Bar价格）
        for symbol in list(self.portfolio.positions.keys()):
            last_prices = {}
            for sym, df in data.items():
                df = df.copy()
                df['date'] = pd.to_datetime(df['date'])
                if not df.empty:
                    last_prices[sym] = df.iloc[-1]['close']
            if symbol in last_prices:
                self._close_position(symbol, last_prices[symbol], dates[-1] if dates else datetime.now())

        return self._generate_report()

    def _execute_signals(self, signals: List[Signal], bar: pd.Series, date):
        """执行信号（去重：同标的同方向不重复开仓）"""
        symbol = bar['symbol'] if 'symbol' in bar.index else bar.name
        # 同一K线多个策略可能产生多个信号，取第一个有效信号
        seen_symbols = set()
        for signal in signals:
            if signal.symbol in seen_symbols:
                continue
            seen_symbols.add(signal.symbol)

            if signal.direction == Direction.CLOSE:
                self._close_position(signal.symbol, bar['close'], date)
            elif signal.symbol not in self.portfolio.positions:
                # 未持仓才开仓
                self._open_position(signal, bar['close'], date)
            else:
                # 已持仓且方向相反，先平后开
                existing = self.portfolio.positions[signal.symbol]
                if existing.direction != signal.direction:
                    self._close_position(signal.symbol, bar['close'], date)
                    self._open_position(signal, bar['close'], date)

    def _open_position(self, signal: Signal, price: float, date):
        """开仓（含仓位计算）"""
        if price <= 0:
            return

        # 滑点
        exec_price = price * (1 + self.slippage) if signal.direction == Direction.LONG else price * (1 - self.slippage)

        # 当前持仓数
        n_positions = len(self.portfolio.positions)
        if n_positions >= self.sizer.max_positions:
            return

        # 计算开仓量
        volume = self.sizer.calculate_volume(
            capital=self.portfolio.capital,
            price=exec_price,
            current_positions=n_positions
        )
        if volume <= 0:
            return

        commission = exec_price * volume * self.commission_rate
        cost = exec_price * volume * 0.1 + commission  # 10% 保证金 + 手续费

        if self.portfolio.capital < cost:
            return

        self.portfolio.capital -= commission  # 手续费即时扣除，保证金仅占用不扣除

        self.portfolio.positions[signal.symbol] = Position(
            symbol=signal.symbol,
            direction=signal.direction,
            volume=volume,
            entry_price=exec_price,
            entry_date=date
        )
        # 记录待配对
        self._pending_open[signal.symbol] = self.portfolio.positions[signal.symbol]

    def _close_position(self, symbol: str, price: float, date):
        """平仓并记录配对交易"""
        if symbol not in self.portfolio.positions:
            return

        pos = self.portfolio.positions[symbol]
        exec_price = price * (1 - self.slippage) if pos.direction == Direction.LONG else price * (1 + self.slippage)
        commission = exec_price * pos.volume * self.commission_rate

        # 盈亏计算
        if pos.direction == Direction.LONG:
            gross_pnl = (exec_price - pos.entry_price) * pos.volume
        else:
            gross_pnl = (pos.entry_price - exec_price) * pos.volume

        net_pnl = gross_pnl - commission

        # 归还保证金 + 净盈亏
        margin_return = pos.entry_price * pos.volume * 0.1
        self.portfolio.capital += margin_return + net_pnl

        # 记录完整配对交易
        self.portfolio.trades.append(Trade(
            open_date=pos.entry_date,
            close_date=date,
            symbol=symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exec_price,
            volume=pos.volume,
            pnl=net_pnl,
            commission=commission
        ))

        del self.portfolio.positions[symbol]
        if symbol in self._pending_open:
            del self._pending_open[symbol]

    def _generate_report(self) -> Dict:
        """生成完整回测报告"""
        trades = self.portfolio.trades
        equity_curve = self.portfolio.equity_curve

        # 净值序列
        if equity_curve:
            equity_values = np.array([v for _, v in equity_curve])
            equity_dates = [d for d, _ in equity_curve]
        else:
            equity_values = np.array([self.initial_capital])
            equity_dates = []

        # 日收益率
        daily_returns = np.diff(equity_values) / equity_values[:-1] if len(equity_values) > 1 else np.array([])

        # 最大回撤
        max_dd, max_dd_start, max_dd_end = self._calculate_max_drawdown_detail(equity_values, equity_dates)

        # 年化收益
        n_days = len(equity_values)
        total_return = (equity_values[-1] - self.initial_capital) / self.initial_capital if len(equity_values) > 0 else 0
        annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        # 夏普比率
        sharpe = self._calculate_sharpe(daily_returns)

        # Calmar 比率
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0

        # 胜率 & 盈亏比
        win_rate, profit_factor, avg_win, avg_loss = self._calculate_trade_stats(trades)

        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.portfolio.capital,
            'total_return': total_return,
            'annual_return': annual_return,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'max_dd_start': max_dd_start,
            'max_dd_end': max_dd_end,
            'calmar_ratio': calmar,
            'equity_curve': equity_curve,   # List[(date, value)]
            'trades': trades,
        }

    def _calculate_trade_stats(self, trades: List[Trade]):
        """计算真实胜率、盈亏比"""
        if not trades:
            return 0.0, 0.0, 0.0, 0.0

        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl <= 0]

        win_rate = len(wins) / len(trades)
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float('inf')

        return win_rate, profit_factor, avg_win, avg_loss

    def _calculate_sharpe(self, returns: np.ndarray, risk_free: float = 0.03) -> float:
        if len(returns) < 2:
            return 0.0
        excess = returns - risk_free / 252
        return float(np.mean(excess) / np.std(excess) * np.sqrt(252)) if np.std(excess) > 0 else 0.0

    def _calculate_max_drawdown_detail(self, equity: np.ndarray, dates: list):
        """最大回撤及起止时间"""
        if len(equity) < 2:
            return 0.0, None, None

        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd = float(np.min(drawdown))
        min_idx = int(np.argmin(drawdown))

        # 找峰值时间
        peak_idx = int(np.argmax(equity[:min_idx + 1]))

        dd_start = dates[peak_idx] if dates else None
        dd_end = dates[min_idx] if dates else None

        return max_dd, dd_start, dd_end


# ==================== 主策略类 ====================

class CTAStrategy:
    """CTA策略主类 - 多资产组合"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.engine = BacktestEngine(
            initial_capital=self.config.get('initial_capital', 1_000_000),
            commission_rate=self.config.get('commission_rate', 0.0003),
            slippage=self.config.get('slippage', 0.0001),
            position_method=self.config.get('position_method', 'fixed_fraction'),
            position_fraction=self.config.get('position_fraction', 0.1),
            max_positions=self.config.get('max_positions', 10),
        )
        self._init_strategies()

    def _init_strategies(self):
        """初始化策略组合（可按 config 选择启用哪些）"""
        enabled = self.config.get('strategies', ['dual_ma', 'turtle', 'bollinger', 'macd', 'rsi'])

        strategy_map = {
            'dual_ma': DualMovingAverageStrategy(short_window=5, long_window=20),
            'turtle': TurtleStrategy(entry_period=20, exit_period=10),
            'bollinger': BollingerBandStrategy(window=20, num_std=2.0),
            'macd': MACDStrategy(fast=12, slow=26, signal=9),
            'rsi': RSIStrategy(period=14, oversold=30, overbought=70),
        }

        for key in enabled:
            if key in strategy_map:
                self.engine.add_strategy(strategy_map[key])

    def run(self, data: Dict[str, pd.DataFrame],
            start_date=None, end_date=None) -> Dict:
        return self.engine.run(data, start_date=start_date, end_date=end_date)

    def get_available_strategies(self) -> List[str]:
        return [s.name for s in self.engine.portfolio.strategies]


# ==================== 数据加载工具 ====================

# 标准列名映射（中文 -> 英文）
COLUMN_MAPPING = {
    # 日期
    '日期': 'date', '时间': 'date',
    # 标的代码 / 名称
    '代码': 'symbol', '股票代码': 'symbol',
    '名称': 'name', '股票名称': 'name',
    # 开盘价
    '开盘': 'open', '开盘价': 'open',
    '开盘价(元)': 'open',
    # 最高价
    '最高': 'high', '最高价': 'high',
    '最高价(元)': 'high',
    # 最低价
    '最低': 'low', '最低价': 'low',
    '最低价(元)': 'low',
    # 收盘价
    '收盘': 'close', '收盘价': 'close',
    '收盘价(元)': 'close',
    # 成交量
    '成交量': 'volume', '成交量(手)': 'volume',
    '成交量(股)': 'volume', '成交量(张)': 'volume',
    # 成交额
    '成交额': 'amount', '成交额(元)': 'amount',
    '成交额(百万)': 'amount_m',   # 单位百万，保留为独立字段，避免与元混淆
    # 涨跌幅
    '涨跌幅': 'change_pct', '涨跌幅(%)': 'change_pct',
    # 换手率
    '换手率': 'turnover', '换手率(%)': 'turnover',
}

REQUIRED_COLUMNS = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']


def load_market_data(file_path: str,
                     symbol: str = None,
                     asset_type: str = 'stock') -> Tuple[pd.DataFrame, List[str]]:
    """
    加载市场数据并标准化列名

    Returns:
        (df, warnings): df 为标准化 DataFrame，warnings 为警告信息列表
    """
    warnings_list = []

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_path}，请使用 .csv 或 .xlsx")

    # 标准化列名
    df = df.rename(columns=COLUMN_MAPPING)
    df.columns = [c.lower().strip() for c in df.columns]

    # 检查必须列
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        warnings_list.append(f"缺少必要列: {missing}，请检查文件格式")

    # 注入 symbol
    if symbol and 'symbol' not in df.columns:
        df['symbol'] = symbol
        warnings_list.append(f"文件中未找到 symbol 列，已自动填充为: {symbol}")

    # 日期解析
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        null_dates = df['date'].isna().sum()
        if null_dates > 0:
            warnings_list.append(f"有 {null_dates} 行日期解析失败，已删除")
            df = df.dropna(subset=['date'])

    # 数值列转换
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)

    return df, warnings_list


def validate_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """校验数据质量"""
    issues = []

    if df.empty:
        return False, ["数据为空"]

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            issues.append(f"缺少必要列: {col}")

    if 'close' in df.columns:
        zero_prices = (df['close'] <= 0).sum()
        if zero_prices > 0:
            issues.append(f"存在 {zero_prices} 条收盘价 <= 0 的记录")

    if 'high' in df.columns and 'low' in df.columns:
        invalid_hl = (df['high'] < df['low']).sum()
        if invalid_hl > 0:
            issues.append(f"存在 {invalid_hl} 条最高价 < 最低价的记录")

    if 'date' in df.columns:
        dup_dates = df.duplicated(subset=['date', 'symbol'] if 'symbol' in df.columns else ['date']).sum()
        if dup_dates > 0:
            issues.append(f"存在 {dup_dates} 条重复日期记录")

    return len(issues) == 0, issues


# ==================== 示例运行 ====================

if __name__ == '__main__':
    config = {
        'initial_capital': 1_000_000,
        'commission_rate': 0.0003,
        'slippage': 0.0001,
        'position_method': 'fixed_fraction',
        'position_fraction': 0.1,
        'max_positions': 5,
        'strategies': ['dual_ma', 'turtle', 'bollinger', 'macd', 'rsi'],
    }

    cta = CTAStrategy(config)
    print("已启用策略:", cta.get_available_strategies())

    # 用随机数据做演示
    np.random.seed(42)
    n = 500
    price = 3000 + np.cumsum(np.random.randn(n) * 20)
    demo_df = pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=n, freq='B'),
        'symbol': 'IF9999',
        'open': price * (1 + np.random.randn(n) * 0.002),
        'high': price * (1 + np.abs(np.random.randn(n)) * 0.005),
        'low': price * (1 - np.abs(np.random.randn(n)) * 0.005),
        'close': price,
        'volume': np.random.randint(10000, 50000, n).astype(float),
    })

    result = cta.run({'IF9999': demo_df})
    print(f"\n===== 回测报告 =====")
    print(f"初始资金:   {result['initial_capital']:,.0f}")
    print(f"最终资金:   {result['final_capital']:,.0f}")
    print(f"总收益率:   {result['total_return']:.2%}")
    print(f"年化收益率: {result['annual_return']:.2%}")
    print(f"总交易次数: {result['total_trades']}")
    print(f"胜率:       {result['win_rate']:.2%}")
    print(f"盈亏比:     {result['profit_factor']:.2f}")
    print(f"夏普比率:   {result['sharpe_ratio']:.2f}")
    print(f"最大回撤:   {result['max_drawdown']:.2%}")
    print(f"Calmar:     {result['calmar_ratio']:.2f}")
