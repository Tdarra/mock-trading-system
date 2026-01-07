"""
Backtester Module
Evaluates trading strategy performance on historical data.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_fetcher import DataFetcher
from forecaster import ChronosForecaster, ForecastResult
from strategy import TradingStrategy, TradeSignal, Signal
from portfolio import Portfolio, PortfolioSnapshot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""
    start_date: str
    end_date: str
    tickers: List[str]
    initial_capital: float = 100000
    rebalance_frequency: str = 'weekly'  # 'daily', 'weekly', 'monthly'
    forecast_horizon: int = 5
    context_length: int = 64
    transaction_cost: float = 0.001
    slippage: float = 0.0005


@dataclass
class BacktestResult:
    """Container for backtest results."""
    config: BacktestConfig
    portfolio_history: List[PortfolioSnapshot]
    trade_history: pd.DataFrame
    performance_metrics: Dict
    benchmark_returns: pd.Series
    daily_returns: pd.Series
    positions_over_time: pd.DataFrame


class Backtester:
    """
    Backtesting engine for evaluating trading strategies.
    
    Runs historical simulations using Chronos forecasts to generate
    trading signals and measure strategy performance.
    """
    
    def __init__(
        self,
        forecaster: Optional[ChronosForecaster] = None,
        strategy: Optional[TradingStrategy] = None,
        data_fetcher: Optional[DataFetcher] = None
    ):
        """
        Initialize the backtester.
        
        Args:
            forecaster: Chronos forecaster instance
            strategy: Trading strategy instance
            data_fetcher: Data fetcher instance
        """
        self.forecaster = forecaster or ChronosForecaster(
            model_size='bolt-small',
            prediction_length=5
        )
        self.strategy = strategy or TradingStrategy()
        self.data_fetcher = data_fetcher or DataFetcher(lookback_days=365)
    
    def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """
        Run a backtest simulation.
        
        Args:
            config: Backtest configuration
            
        Returns:
            BacktestResult with performance metrics and history
        """
        logger.info(f"Starting backtest from {config.start_date} to {config.end_date}")
        logger.info(f"Tickers: {config.tickers}")
        
        # Fetch all historical data
        all_data = self._fetch_data(config)
        
        if not all_data:
            raise ValueError("No data available for backtest")
        
        # Get market data for benchmark
        market_data = self.data_fetcher.fetch_market_data(
            start_date=config.start_date,
            end_date=config.end_date
        )
        
        # Initialize portfolio
        portfolio = Portfolio(
            initial_capital=config.initial_capital,
            transaction_cost=config.transaction_cost,
            slippage=config.slippage
        )
        
        # Determine rebalance dates
        rebalance_dates = self._get_rebalance_dates(
            all_data, config.rebalance_frequency
        )
        
        logger.info(f"Running simulation with {len(rebalance_dates)} rebalance periods")
        
        # Track positions over time
        positions_history = []
        
        # Run simulation
        for i, date in enumerate(rebalance_dates):
            try:
                # Get data up to this date for each ticker
                current_data = {}
                current_prices = {}
                
                for ticker, df in all_data.items():
                    mask = df.index <= date
                    if mask.sum() >= config.context_length:
                        current_data[ticker] = df[mask]
                        current_prices[ticker] = df.loc[mask, 'Close'].iloc[-1]
                
                if not current_data:
                    continue
                
                # Update portfolio prices
                portfolio.update_prices(current_prices)
                
                # Generate forecasts (simulating look-ahead constraint)
                forecasts = self._generate_forecasts(current_data, config)
                
                # Generate trading signals
                signals = self._generate_signals(
                    forecasts, portfolio, current_prices
                )
                
                # Execute trades
                for signal in signals:
                    if signal.ticker in current_prices:
                        portfolio.execute_trade(
                            signal,
                            current_price=current_prices[signal.ticker],
                            timestamp=date
                        )
                
                # Get market return for this period
                market_return = self._get_period_return(market_data, date, rebalance_dates, i)
                
                # Take snapshot
                snapshot = portfolio.take_snapshot(
                    timestamp=date,
                    market_return=market_return
                )
                
                # Record positions
                positions_history.append({
                    'date': date,
                    **{f'{t}_weight': v / snapshot.total_value 
                       for t, v in snapshot.positions.items()},
                    'cash_weight': snapshot.cash / snapshot.total_value
                })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(rebalance_dates)} periods. "
                               f"Portfolio value: ${snapshot.total_value:,.2f}")
                    
            except Exception as e:
                logger.error(f"Error processing date {date}: {e}")
                continue
        
        # Compile results
        result = self._compile_results(
            config=config,
            portfolio=portfolio,
            market_data=market_data,
            positions_history=positions_history
        )
        
        logger.info(f"Backtest complete. Final value: ${portfolio.total_value:,.2f}")
        logger.info(f"Total return: {portfolio.total_return:.2%}")
        
        return result
    
    def _fetch_data(self, config: BacktestConfig) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for all tickers."""
        # Extend start date to have enough context
        extended_start = (
            pd.to_datetime(config.start_date) - 
            timedelta(days=config.context_length * 2)
        ).strftime('%Y-%m-%d')
        
        all_data = {}
        for ticker in config.tickers:
            try:
                df = self.data_fetcher.fetch_stock_data(
                    ticker,
                    start_date=extended_start,
                    end_date=config.end_date
                )
                if not df.empty:
                    all_data[ticker] = df
            except Exception as e:
                logger.warning(f"Could not fetch {ticker}: {e}")
        
        return all_data
    
    def _get_rebalance_dates(
        self,
        all_data: Dict[str, pd.DataFrame],
        frequency: str
    ) -> List[pd.Timestamp]:
        """Get dates on which to rebalance portfolio."""
        # Get common date range
        all_dates = set()
        for df in all_data.values():
            all_dates.update(df.index.tolist())
        
        dates = sorted(all_dates)
        
        if frequency == 'daily':
            return dates
        elif frequency == 'weekly':
            # Get last trading day of each week
            df = pd.DataFrame(index=dates)
            return df.resample('W').last().index.tolist()
        elif frequency == 'monthly':
            df = pd.DataFrame(index=dates)
            return df.resample('M').last().index.tolist()
        else:
            return dates
    
    def _generate_forecasts(
        self,
        current_data: Dict[str, pd.DataFrame],
        config: BacktestConfig
    ) -> Dict[str, ForecastResult]:
        """Generate forecasts for all tickers."""
        forecasts = {}
        
        for ticker, df in current_data.items():
            try:
                # Prepare data for forecasting
                price_series = df['Close'].iloc[-config.context_length:]
                
                # Generate forecast
                forecast = self.forecaster.forecast(
                    price_series,
                    ticker=ticker,
                    num_samples=20
                )
                forecasts[ticker] = forecast
                
            except Exception as e:
                logger.warning(f"Forecast error for {ticker}: {e}")
        
        return forecasts
    
    def _generate_signals(
        self,
        forecasts: Dict[str, ForecastResult],
        portfolio: Portfolio,
        current_prices: Dict[str, float]
    ) -> List[TradeSignal]:
        """Generate trading signals from forecasts."""
        current_positions = {
            t: p.market_value 
            for t, p in portfolio.positions.items()
        }
        
        signals = self.strategy.generate_signals_batch(
            forecasts=forecasts,
            portfolio_value=portfolio.total_value,
            current_positions=current_positions
        )
        
        # Rank and filter signals
        ranked_signals = self.strategy.rank_signals(signals)
        
        return ranked_signals
    
    def _get_period_return(
        self,
        market_data: pd.DataFrame,
        current_date: pd.Timestamp,
        all_dates: List[pd.Timestamp],
        current_idx: int
    ) -> float:
        """Calculate market return for this period."""
        if current_idx == 0:
            return 0
        
        prev_date = all_dates[current_idx - 1]
        
        try:
            current_price = market_data.loc[
                market_data.index <= current_date, 'Close'
            ].iloc[-1]
            prev_price = market_data.loc[
                market_data.index <= prev_date, 'Close'
            ].iloc[-1]
            
            return (current_price - prev_price) / prev_price
        except:
            return 0
    
    def _compile_results(
        self,
        config: BacktestConfig,
        portfolio: Portfolio,
        market_data: pd.DataFrame,
        positions_history: List[Dict]
    ) -> BacktestResult:
        """Compile all backtest results."""
        # Performance metrics
        metrics = portfolio.get_performance_metrics()
        
        # Trade history
        trade_history = portfolio.get_trade_history_df()
        
        # Daily returns as Series
        if portfolio.snapshots:
            dates = [s.timestamp for s in portfolio.snapshots]
            returns = [s.daily_return for s in portfolio.snapshots]
            daily_returns = pd.Series(returns, index=dates)
        else:
            daily_returns = pd.Series()
        
        # Benchmark returns
        benchmark_returns = market_data['Returns'].dropna()
        
        # Positions over time
        positions_df = pd.DataFrame(positions_history)
        if not positions_df.empty:
            positions_df.set_index('date', inplace=True)
        
        # Add comparison metrics
        if len(daily_returns) > 0 and len(benchmark_returns) > 0:
            # Align dates
            common_dates = daily_returns.index.intersection(benchmark_returns.index)
            
            if len(common_dates) > 0:
                strategy_cum = (1 + daily_returns.loc[common_dates]).cumprod()
                benchmark_cum = (1 + benchmark_returns.loc[common_dates]).cumprod()
                
                # Alpha (simplified)
                metrics['alpha'] = (
                    strategy_cum.iloc[-1] - benchmark_cum.iloc[-1]
                ) / len(common_dates) * 252
                
                # Information Ratio
                tracking_diff = daily_returns.loc[common_dates] - benchmark_returns.loc[common_dates]
                if tracking_diff.std() > 0:
                    metrics['information_ratio'] = (
                        tracking_diff.mean() / tracking_diff.std() * np.sqrt(252)
                    )
                else:
                    metrics['information_ratio'] = 0
        
        return BacktestResult(
            config=config,
            portfolio_history=portfolio.snapshots,
            trade_history=trade_history,
            performance_metrics=metrics,
            benchmark_returns=benchmark_returns,
            daily_returns=daily_returns,
            positions_over_time=positions_df
        )
    
    def run_walk_forward(
        self,
        config: BacktestConfig,
        train_periods: int = 60,
        test_periods: int = 20,
        step_size: int = 10
    ) -> List[BacktestResult]:
        """
        Run walk-forward analysis.
        
        Trains on rolling windows and tests on subsequent periods
        to avoid look-ahead bias.
        
        Args:
            config: Base backtest configuration
            train_periods: Number of periods for training
            test_periods: Number of periods for testing
            step_size: Number of periods to step forward
            
        Returns:
            List of BacktestResult for each walk-forward window
        """
        logger.info("Starting walk-forward analysis")
        
        # Fetch full dataset
        all_data = self._fetch_data(config)
        
        if not all_data:
            raise ValueError("No data available")
        
        # Get all dates
        all_dates = sorted(set().union(*[
            set(df.index.tolist()) for df in all_data.values()
        ]))
        
        results = []
        start_idx = train_periods
        
        while start_idx + test_periods <= len(all_dates):
            test_start = all_dates[start_idx]
            test_end = all_dates[min(start_idx + test_periods - 1, len(all_dates) - 1)]
            
            logger.info(f"Walk-forward window: {test_start} to {test_end}")
            
            # Create config for this window
            window_config = BacktestConfig(
                start_date=test_start.strftime('%Y-%m-%d'),
                end_date=test_end.strftime('%Y-%m-%d'),
                tickers=config.tickers,
                initial_capital=config.initial_capital,
                rebalance_frequency=config.rebalance_frequency,
                forecast_horizon=config.forecast_horizon,
                context_length=config.context_length,
                transaction_cost=config.transaction_cost,
                slippage=config.slippage
            )
            
            try:
                result = self.run_backtest(window_config)
                results.append(result)
            except Exception as e:
                logger.error(f"Walk-forward error: {e}")
            
            start_idx += step_size
        
        return results


def generate_backtest_report(result: BacktestResult) -> str:
    """Generate a formatted backtest report."""
    metrics = result.performance_metrics
    
    report = f"""
================================================================================
                         BACKTEST PERFORMANCE REPORT
================================================================================

Configuration:
  Period: {result.config.start_date} to {result.config.end_date}
  Tickers: {', '.join(result.config.tickers)}
  Initial Capital: ${result.config.initial_capital:,.2f}
  Rebalance Frequency: {result.config.rebalance_frequency}

--------------------------------------------------------------------------------
                              RETURNS
--------------------------------------------------------------------------------
  Total Return:         {metrics.get('total_return', 0):.2%}
  Annualized Return:    {metrics.get('total_return', 0) * 252 / max(len(result.daily_returns), 1):.2%}
  Best Day:             {metrics.get('best_day', 0):.2%}
  Worst Day:            {metrics.get('worst_day', 0):.2%}
  Avg Daily Return:     {metrics.get('avg_daily_return', 0):.4%}

--------------------------------------------------------------------------------
                            RISK METRICS
--------------------------------------------------------------------------------
  Volatility (Ann.):    {metrics.get('volatility', 0):.2%}
  Max Drawdown:         {metrics.get('max_drawdown', 0):.2%}
  Beta:                 {metrics.get('beta', 1):.2f}

--------------------------------------------------------------------------------
                         RISK-ADJUSTED RETURNS
--------------------------------------------------------------------------------
  Sharpe Ratio:         {metrics.get('sharpe_ratio', 0):.2f}
  Sortino Ratio:        {metrics.get('sortino_ratio', 0):.2f}
  Calmar Ratio:         {metrics.get('calmar_ratio', 0):.2f}
  Information Ratio:    {metrics.get('information_ratio', 0):.2f}
  Alpha:                {metrics.get('alpha', 0):.4f}

--------------------------------------------------------------------------------
                           TRADING STATISTICS
--------------------------------------------------------------------------------
  Total Trades:         {metrics.get('num_trades', 0)}
  Win Rate:             {metrics.get('win_rate', 0):.2%}
  Total P&L:            ${metrics.get('total_pnl', 0):,.2f}
  Realized P&L:         ${metrics.get('realized_pnl', 0):,.2f}
  Unrealized P&L:       ${metrics.get('unrealized_pnl', 0):,.2f}

================================================================================
"""
    return report


if __name__ == "__main__":
    # Run a sample backtest
    config = BacktestConfig(
        start_date='2024-01-01',
        end_date='2024-06-30',
        tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
        initial_capital=100000,
        rebalance_frequency='weekly',
        forecast_horizon=5
    )
    
    backtester = Backtester()
    
    try:
        result = backtester.run_backtest(config)
        print(generate_backtest_report(result))
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        print(f"Note: Full backtest requires market data access. Error: {e}")
