"""
Mock Trading System
A Chronos-2 powered mock stock trading system with backtesting and dashboard.
"""

from .data_fetcher import DataFetcher, get_sample_tickers
from .forecaster import ChronosForecaster, ForecastResult
from .strategy import TradingStrategy, TradeSignal, Signal
from .portfolio import Portfolio, Position, Trade
from .backtester import Backtester, BacktestConfig, BacktestResult

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    'DataFetcher',
    'get_sample_tickers',
    'ChronosForecaster',
    'ForecastResult',
    'TradingStrategy',
    'TradeSignal',
    'Signal',
    'Portfolio',
    'Position',
    'Trade',
    'Backtester',
    'BacktestConfig',
    'BacktestResult',
]
