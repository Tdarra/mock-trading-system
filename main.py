#!/usr/bin/env python3
"""
Mock Trading System - Main Entry Point

Usage:
    python main.py backtest       # Run a backtest
    python main.py dashboard      # Launch the Streamlit dashboard
    python main.py forecast AAPL  # Generate forecast for a ticker
    python main.py trade          # Run live mock trading loop
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime, timedelta


def run_backtest(args):
    """Run a backtest simulation."""
    from backtester import Backtester, BacktestConfig, generate_backtest_report
    
    print("=" * 60)
    print("Running Backtest")
    print("=" * 60)
    
    config = BacktestConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        tickers=args.tickers.split(','),
        initial_capital=args.capital,
        rebalance_frequency=args.frequency,
        forecast_horizon=5
    )
    
    print(f"Period: {config.start_date} to {config.end_date}")
    print(f"Tickers: {config.tickers}")
    print(f"Capital: ${config.initial_capital:,.2f}")
    print(f"Rebalance: {config.rebalance_frequency}")
    print()
    
    backtester = Backtester()
    
    try:
        result = backtester.run_backtest(config)
        print(generate_backtest_report(result))
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                f.write(generate_backtest_report(result))
            print(f"Report saved to {args.output}")
            
    except Exception as e:
        print(f"Backtest failed: {e}")
        sys.exit(1)


def run_dashboard(args):
    """Launch the Streamlit dashboard."""
    import subprocess
    
    dashboard_path = os.path.join(os.path.dirname(__file__), 'src', 'dashboard.py')
    
    print("Launching Streamlit dashboard...")
    print("Access at: http://localhost:8501")
    print()
    
    subprocess.run([
        'streamlit', 'run', dashboard_path,
        '--server.port', str(args.port),
        '--server.address', args.host
    ])


def run_forecast(args):
    """Generate a forecast for a single ticker."""
    from data_fetcher import DataFetcher
    from forecaster import ChronosForecaster
    from strategy import TradingStrategy
    
    print("=" * 60)
    print(f"Generating Forecast for {args.ticker}")
    print("=" * 60)
    
    # Fetch data
    print(f"\nFetching historical data...")
    fetcher = DataFetcher(lookback_days=180)
    data = fetcher.fetch_stock_data(args.ticker)
    
    if data.empty:
        print(f"Error: No data available for {args.ticker}")
        sys.exit(1)
    
    print(f"Retrieved {len(data)} data points")
    print(f"Latest price: ${data['Close'].iloc[-1]:.2f}")
    
    # Generate forecast
    print(f"\nGenerating {args.horizon}-day forecast...")
    forecaster = ChronosForecaster(
        model_size=args.model,
        prediction_length=args.horizon
    )
    
    forecast = forecaster.forecast(data['Close'], args.ticker)
    
    print(f"\n{'=' * 40}")
    print(f"Forecast Results for {forecast.ticker}")
    print(f"{'=' * 40}")
    print(f"Last actual price: ${forecast.last_actual_price:.2f}")
    print(f"Expected return: {forecast.expected_return:+.2%}")
    print(f"Forecast volatility: {forecast.forecast_volatility:.2%}")
    print(f"Confidence score: {forecast.confidence_score:.2f}")
    
    print(f"\n{'Date':<12} {'Forecast':>10} {'Low (10%)':>10} {'High (90%)':>10}")
    print("-" * 45)
    for i, date in enumerate(forecast.forecast_dates):
        print(f"{date.strftime('%Y-%m-%d'):<12} "
              f"${forecast.point_forecast[i]:>9.2f} "
              f"${forecast.lower_bound[i]:>9.2f} "
              f"${forecast.upper_bound[i]:>9.2f}")
    
    # Generate trading signal
    if args.signal:
        print(f"\n{'=' * 40}")
        print("Trading Signal")
        print(f"{'=' * 40}")
        
        strategy = TradingStrategy()
        signal = strategy.generate_signal(
            forecast=forecast,
            portfolio_value=args.capital,
            current_position=0
        )
        
        print(f"Signal: {signal.signal.value}")
        print(f"Target Weight: {signal.target_weight:.1%}")
        print(f"Position Size: ${signal.position_size:,.2f}")
        print(f"Shares: {signal.shares}")
        print(f"Stop Loss: ${signal.stop_loss:.2f}")
        print(f"Take Profit: ${signal.take_profit:.2f}")
        print(f"Rationale: {signal.rationale}")


def run_trade(args):
    """Run a mock trading loop."""
    from data_fetcher import DataFetcher
    from forecaster import ChronosForecaster
    from strategy import TradingStrategy
    from portfolio import Portfolio
    import time
    
    print("=" * 60)
    print("Starting Mock Trading Loop")
    print("=" * 60)
    print(f"Tickers: {args.tickers}")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print(f"Update Interval: {args.interval} seconds")
    print()
    print("Press Ctrl+C to stop")
    print()
    
    # Initialize components
    fetcher = DataFetcher(lookback_days=180)
    forecaster = ChronosForecaster(model_size='bolt-small', prediction_length=5)
    strategy = TradingStrategy()
    portfolio = Portfolio(initial_capital=args.capital)
    
    tickers = args.tickers.split(',')
    
    try:
        iteration = 0
        while True:
            iteration += 1
            print(f"\n{'=' * 40}")
            print(f"Iteration {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'=' * 40}")
            
            for ticker in tickers:
                try:
                    # Fetch latest data
                    data = fetcher.fetch_stock_data(ticker)
                    if data.empty:
                        continue
                    
                    current_price = data['Close'].iloc[-1]
                    
                    # Update portfolio prices
                    portfolio.update_prices({ticker: current_price})
                    
                    # Generate forecast
                    forecast = forecaster.forecast(data['Close'], ticker)
                    
                    # Get current position
                    current_position = 0
                    if ticker in portfolio.positions:
                        current_position = portfolio.positions[ticker].market_value
                    
                    # Generate signal
                    signal = strategy.generate_signal(
                        forecast=forecast,
                        portfolio_value=portfolio.total_value,
                        current_position=current_position
                    )
                    
                    print(f"\n{ticker}: ${current_price:.2f} | "
                          f"Expected: {forecast.expected_return:+.2%} | "
                          f"Signal: {signal.signal.value}")
                    
                    # Execute trade if not HOLD
                    if signal.signal.value != 'HOLD':
                        trade = portfolio.execute_trade(signal, current_price=current_price)
                        if trade:
                            print(f"  -> Executed {trade.side} {trade.shares} shares @ ${trade.price:.2f}")
                    
                except Exception as e:
                    print(f"Error processing {ticker}: {e}")
            
            # Portfolio summary
            print(f"\nPortfolio: ${portfolio.total_value:,.2f} | "
                  f"Return: {portfolio.total_return:+.2%} | "
                  f"Positions: {len(portfolio.positions)}")
            
            # Take snapshot
            portfolio.take_snapshot()
            
            # Wait for next iteration
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n\nStopping trading loop...")
        print(f"\nFinal Portfolio Value: ${portfolio.total_value:,.2f}")
        print(f"Total Return: {portfolio.total_return:+.2%}")
        print(f"Total Trades: {len(portfolio.trade_history)}")


def main():
    parser = argparse.ArgumentParser(
        description='Mock Trading System powered by Chronos-2',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run a backtest')
    backtest_parser.add_argument(
        '--start-date', '-s',
        default=(datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d'),
        help='Backtest start date (YYYY-MM-DD)'
    )
    backtest_parser.add_argument(
        '--end-date', '-e',
        default=datetime.now().strftime('%Y-%m-%d'),
        help='Backtest end date (YYYY-MM-DD)'
    )
    backtest_parser.add_argument(
        '--tickers', '-t',
        default='AAPL,MSFT,GOOGL,AMZN,NVDA',
        help='Comma-separated list of tickers'
    )
    backtest_parser.add_argument(
        '--capital', '-c',
        type=float,
        default=100000,
        help='Initial capital'
    )
    backtest_parser.add_argument(
        '--frequency', '-f',
        choices=['daily', 'weekly', 'monthly'],
        default='weekly',
        help='Rebalance frequency'
    )
    backtest_parser.add_argument(
        '--output', '-o',
        help='Output file for report'
    )
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch the dashboard')
    dashboard_parser.add_argument(
        '--port', '-p',
        type=int,
        default=8501,
        help='Port number'
    )
    dashboard_parser.add_argument(
        '--host', '-H',
        default='localhost',
        help='Host address'
    )
    
    # Forecast command
    forecast_parser = subparsers.add_parser('forecast', help='Generate a forecast')
    forecast_parser.add_argument(
        'ticker',
        help='Ticker symbol'
    )
    forecast_parser.add_argument(
        '--horizon', '-h',
        type=int,
        default=5,
        help='Forecast horizon (days)'
    )
    forecast_parser.add_argument(
        '--model', '-m',
        default='bolt-small',
        help='Chronos model size'
    )
    forecast_parser.add_argument(
        '--signal',
        action='store_true',
        help='Also generate trading signal'
    )
    forecast_parser.add_argument(
        '--capital', '-c',
        type=float,
        default=100000,
        help='Portfolio capital (for signal generation)'
    )
    
    # Trade command
    trade_parser = subparsers.add_parser('trade', help='Run mock trading loop')
    trade_parser.add_argument(
        '--tickers', '-t',
        default='AAPL,MSFT,GOOGL',
        help='Comma-separated list of tickers'
    )
    trade_parser.add_argument(
        '--capital', '-c',
        type=float,
        default=100000,
        help='Initial capital'
    )
    trade_parser.add_argument(
        '--interval', '-i',
        type=int,
        default=60,
        help='Update interval in seconds'
    )
    
    args = parser.parse_args()
    
    if args.command == 'backtest':
        run_backtest(args)
    elif args.command == 'dashboard':
        run_dashboard(args)
    elif args.command == 'forecast':
        run_forecast(args)
    elif args.command == 'trade':
        run_trade(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
