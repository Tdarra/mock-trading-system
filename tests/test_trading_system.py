"""
Unit tests for the Mock Trading System.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from strategy import TradingStrategy, TradeSignal, Signal
from portfolio import Portfolio, Position
from forecaster import ForecastResult


class TestTradingStrategy:
    """Tests for TradingStrategy class."""
    
    @pytest.fixture
    def strategy(self):
        return TradingStrategy(
            return_threshold_buy=0.02,
            return_threshold_sell=-0.02,
            min_confidence=0.5,
            max_position_weight=0.25
        )
    
    @pytest.fixture
    def bullish_forecast(self):
        return ForecastResult(
            ticker='AAPL',
            forecast_dates=pd.date_range('2024-01-01', periods=5, freq='B'),
            point_forecast=np.array([155, 157, 159, 158, 160]),
            lower_bound=np.array([150, 151, 152, 151, 153]),
            upper_bound=np.array([160, 163, 166, 165, 167]),
            median_forecast=np.array([155, 157, 159, 158, 160]),
            last_actual_price=150.0,
            last_actual_date=pd.Timestamp('2023-12-29'),
            expected_return=0.067,  # 6.7% expected
            forecast_volatility=0.02,
            confidence_score=0.75
        )
    
    @pytest.fixture
    def bearish_forecast(self):
        return ForecastResult(
            ticker='AAPL',
            forecast_dates=pd.date_range('2024-01-01', periods=5, freq='B'),
            point_forecast=np.array([145, 143, 141, 142, 140]),
            lower_bound=np.array([140, 138, 136, 137, 135]),
            upper_bound=np.array([150, 148, 146, 147, 145]),
            median_forecast=np.array([145, 143, 141, 142, 140]),
            last_actual_price=150.0,
            last_actual_date=pd.Timestamp('2023-12-29'),
            expected_return=-0.067,  # -6.7% expected
            forecast_volatility=0.02,
            confidence_score=0.70
        )
    
    @pytest.fixture
    def neutral_forecast(self):
        return ForecastResult(
            ticker='AAPL',
            forecast_dates=pd.date_range('2024-01-01', periods=5, freq='B'),
            point_forecast=np.array([150, 151, 150, 151, 151]),
            lower_bound=np.array([148, 149, 148, 149, 149]),
            upper_bound=np.array([152, 153, 152, 153, 153]),
            median_forecast=np.array([150, 151, 150, 151, 151]),
            last_actual_price=150.0,
            last_actual_date=pd.Timestamp('2023-12-29'),
            expected_return=0.007,  # 0.7% expected
            forecast_volatility=0.01,
            confidence_score=0.60
        )
    
    def test_buy_signal_generation(self, strategy, bullish_forecast):
        signal = strategy.generate_signal(
            forecast=bullish_forecast,
            portfolio_value=100000,
            current_position=0
        )
        
        assert signal.signal == Signal.BUY
        assert signal.target_weight > 0
        assert signal.shares > 0
        assert signal.stop_loss > 0
        assert signal.take_profit > signal.stop_loss
    
    def test_sell_signal_generation(self, strategy, bearish_forecast):
        # First need to have a position
        signal = strategy.generate_signal(
            forecast=bearish_forecast,
            portfolio_value=100000,
            current_position=10000  # Existing position
        )
        
        assert signal.signal == Signal.SELL
    
    def test_hold_signal_generation(self, strategy, neutral_forecast):
        signal = strategy.generate_signal(
            forecast=neutral_forecast,
            portfolio_value=100000,
            current_position=0
        )
        
        assert signal.signal == Signal.HOLD
        assert signal.shares == 0
    
    def test_low_confidence_results_in_hold(self, strategy, bullish_forecast):
        # Modify forecast to have low confidence
        low_conf_forecast = ForecastResult(
            ticker=bullish_forecast.ticker,
            forecast_dates=bullish_forecast.forecast_dates,
            point_forecast=bullish_forecast.point_forecast,
            lower_bound=bullish_forecast.lower_bound,
            upper_bound=bullish_forecast.upper_bound,
            median_forecast=bullish_forecast.median_forecast,
            last_actual_price=bullish_forecast.last_actual_price,
            last_actual_date=bullish_forecast.last_actual_date,
            expected_return=bullish_forecast.expected_return,
            forecast_volatility=bullish_forecast.forecast_volatility,
            confidence_score=0.3  # Low confidence
        )
        
        signal = strategy.generate_signal(
            forecast=low_conf_forecast,
            portfolio_value=100000,
            current_position=0
        )
        
        assert signal.signal == Signal.HOLD
    
    def test_position_weight_constraint(self, strategy, bullish_forecast):
        signal = strategy.generate_signal(
            forecast=bullish_forecast,
            portfolio_value=100000,
            current_position=0
        )
        
        assert signal.target_weight <= strategy.max_position_weight


class TestPortfolio:
    """Tests for Portfolio class."""
    
    @pytest.fixture
    def portfolio(self):
        return Portfolio(initial_capital=100000)
    
    @pytest.fixture
    def buy_signal(self):
        return TradeSignal(
            ticker='AAPL',
            signal=Signal.BUY,
            target_weight=0.1,
            position_size=10000,
            shares=67,
            confidence=0.75,
            expected_return=0.05,
            stop_loss=145.0,
            take_profit=160.0,
            rationale="Test buy"
        )
    
    @pytest.fixture
    def sell_signal(self):
        return TradeSignal(
            ticker='AAPL',
            signal=Signal.SELL,
            target_weight=0,
            position_size=-10000,
            shares=-67,
            confidence=0.65,
            expected_return=-0.03,
            stop_loss=0,
            take_profit=0,
            rationale="Test sell"
        )
    
    def test_initial_state(self, portfolio):
        assert portfolio.cash == 100000
        assert portfolio.total_value == 100000
        assert portfolio.positions_value == 0
        assert len(portfolio.positions) == 0
    
    def test_execute_buy(self, portfolio, buy_signal):
        trade = portfolio.execute_trade(buy_signal, current_price=150.0)
        
        assert trade is not None
        assert trade.side == 'BUY'
        assert 'AAPL' in portfolio.positions
        assert portfolio.cash < 100000
        assert portfolio.positions_value > 0
    
    def test_execute_sell(self, portfolio, buy_signal, sell_signal):
        # First buy
        portfolio.execute_trade(buy_signal, current_price=150.0)
        
        # Then sell
        trade = portfolio.execute_trade(sell_signal, current_price=155.0)
        
        assert trade is not None
        assert trade.side == 'SELL'
        assert 'AAPL' not in portfolio.positions
        assert portfolio.realized_pnl > 0  # Sold at higher price
    
    def test_update_prices(self, portfolio, buy_signal):
        portfolio.execute_trade(buy_signal, current_price=150.0)
        initial_value = portfolio.total_value
        
        # Price goes up
        portfolio.update_prices({'AAPL': 160.0})
        
        assert portfolio.total_value > initial_value
        assert portfolio.positions['AAPL'].current_price == 160.0
    
    def test_insufficient_cash(self, portfolio, buy_signal):
        # Drain most cash
        portfolio.cash = 1000
        
        trade = portfolio.execute_trade(buy_signal, current_price=150.0)
        
        # Should either not execute or buy fewer shares
        if trade is not None:
            assert trade.shares < buy_signal.shares
    
    def test_performance_metrics(self, portfolio, buy_signal):
        portfolio.execute_trade(buy_signal, current_price=150.0)
        portfolio.update_prices({'AAPL': 155.0})
        portfolio.take_snapshot()
        
        metrics = portfolio.get_performance_metrics()
        
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'num_trades' in metrics


class TestPosition:
    """Tests for Position class."""
    
    def test_position_properties(self):
        position = Position(
            ticker='AAPL',
            shares=100,
            avg_cost=150.0,
            current_price=160.0
        )
        
        assert position.market_value == 16000  # 100 * 160
        assert position.cost_basis == 15000  # 100 * 150
        assert position.unrealized_pnl == 1000  # 16000 - 15000
        assert abs(position.unrealized_pnl_pct - 0.0667) < 0.001  # 1000 / 15000
    
    def test_negative_pnl(self):
        position = Position(
            ticker='AAPL',
            shares=100,
            avg_cost=160.0,
            current_price=150.0
        )
        
        assert position.unrealized_pnl < 0
        assert position.unrealized_pnl_pct < 0


class TestForecastResult:
    """Tests for ForecastResult dataclass."""
    
    def test_forecast_result_creation(self):
        result = ForecastResult(
            ticker='AAPL',
            forecast_dates=pd.date_range('2024-01-01', periods=5, freq='B'),
            point_forecast=np.array([150, 152, 154, 153, 155]),
            lower_bound=np.array([145, 146, 147, 146, 148]),
            upper_bound=np.array([155, 158, 161, 160, 162]),
            median_forecast=np.array([150, 152, 154, 153, 155]),
            last_actual_price=148.0,
            last_actual_date=pd.Timestamp('2023-12-29'),
            expected_return=0.047,
            forecast_volatility=0.02,
            confidence_score=0.72
        )
        
        assert result.ticker == 'AAPL'
        assert len(result.forecast_dates) == 5
        assert len(result.point_forecast) == 5
        assert result.expected_return > 0
        assert 0 <= result.confidence_score <= 1


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_trading_cycle(self):
        """Test a complete buy -> hold -> sell cycle."""
        portfolio = Portfolio(initial_capital=100000)
        strategy = TradingStrategy()
        
        # Create bullish forecast
        bullish = ForecastResult(
            ticker='AAPL',
            forecast_dates=pd.date_range('2024-01-01', periods=5, freq='B'),
            point_forecast=np.array([155, 157, 159, 158, 160]),
            lower_bound=np.array([150, 151, 152, 151, 153]),
            upper_bound=np.array([160, 163, 166, 165, 167]),
            median_forecast=np.array([155, 157, 159, 158, 160]),
            last_actual_price=150.0,
            last_actual_date=pd.Timestamp('2023-12-29'),
            expected_return=0.067,
            forecast_volatility=0.02,
            confidence_score=0.75
        )
        
        # Generate and execute buy signal
        buy_signal = strategy.generate_signal(bullish, portfolio.total_value, 0)
        assert buy_signal.signal == Signal.BUY
        
        portfolio.execute_trade(buy_signal, current_price=150.0)
        assert 'AAPL' in portfolio.positions
        
        # Simulate price increase
        portfolio.update_prices({'AAPL': 160.0})
        portfolio.take_snapshot()
        
        # Create bearish forecast
        bearish = ForecastResult(
            ticker='AAPL',
            forecast_dates=pd.date_range('2024-01-08', periods=5, freq='B'),
            point_forecast=np.array([155, 153, 151, 152, 150]),
            lower_bound=np.array([150, 148, 146, 147, 145]),
            upper_bound=np.array([160, 158, 156, 157, 155]),
            median_forecast=np.array([155, 153, 151, 152, 150]),
            last_actual_price=160.0,
            last_actual_date=pd.Timestamp('2024-01-05'),
            expected_return=-0.0625,
            forecast_volatility=0.02,
            confidence_score=0.70
        )
        
        # Generate and execute sell signal
        current_position = portfolio.positions['AAPL'].market_value
        sell_signal = strategy.generate_signal(bearish, portfolio.total_value, current_position)
        assert sell_signal.signal == Signal.SELL
        
        portfolio.execute_trade(sell_signal, current_price=160.0)
        assert 'AAPL' not in portfolio.positions
        
        # Should have profit
        assert portfolio.realized_pnl > 0
        assert portfolio.total_return > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
