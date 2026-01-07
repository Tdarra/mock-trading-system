"""
Dashboard Module
Streamlit-based dashboard for displaying trading performance metrics.
Deploy to Streamlit Cloud for cloud hosting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import DataFetcher, get_sample_tickers
from forecaster import ChronosForecaster
from strategy import TradingStrategy, Signal
from portfolio import Portfolio
from backtester import Backtester, BacktestConfig, generate_backtest_report

# Page configuration
st.set_page_config(
    page_title="Mock Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .positive { color: #00c853; }
    .negative { color: #ff1744; }
    .stMetric > div {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = Portfolio(initial_capital=100000)
    if 'backtest_result' not in st.session_state:
        st.session_state.backtest_result = None
    if 'forecasts' not in st.session_state:
        st.session_state.forecasts = {}
    if 'data_cache' not in st.session_state:
        st.session_state.data_cache = {}


def render_sidebar():
    """Render the sidebar with configuration options."""
    st.sidebar.title("⚙️ Configuration")
    
    # Portfolio Settings
    st.sidebar.header("Portfolio Settings")
    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=10000
    )
    
    # Stock Selection
    st.sidebar.header("Stock Selection")
    default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    ticker_input = st.sidebar.text_input(
        "Tickers (comma-separated)",
        value=", ".join(default_tickers)
    )
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    
    # Backtest Settings
    st.sidebar.header("Backtest Settings")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=180)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now()
        )
    
    rebalance_freq = st.sidebar.selectbox(
        "Rebalance Frequency",
        options=['daily', 'weekly', 'monthly'],
        index=1
    )
    
    # Strategy Parameters
    st.sidebar.header("Strategy Parameters")
    return_threshold = st.sidebar.slider(
        "Buy Threshold (%)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5
    ) / 100
    
    max_position = st.sidebar.slider(
        "Max Position Size (%)",
        min_value=5,
        max_value=50,
        value=25,
        step=5
    ) / 100
    
    return {
        'initial_capital': initial_capital,
        'tickers': tickers,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'rebalance_frequency': rebalance_freq,
        'return_threshold': return_threshold,
        'max_position': max_position
    }


def render_overview_tab(config):
    """Render the portfolio overview tab."""
    st.header("📊 Portfolio Overview")
    
    portfolio = st.session_state.portfolio
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Value",
            f"${portfolio.total_value:,.2f}",
            f"{portfolio.total_return:+.2%}"
        )
    
    with col2:
        st.metric(
            "Cash",
            f"${portfolio.cash:,.2f}",
            f"{(portfolio.cash / portfolio.total_value * 100):.1f}% of portfolio"
        )
    
    with col3:
        st.metric(
            "Positions Value",
            f"${portfolio.positions_value:,.2f}",
            f"{len(portfolio.positions)} holdings"
        )
    
    with col4:
        metrics = portfolio.get_performance_metrics()
        st.metric(
            "Sharpe Ratio",
            f"{metrics.get('sharpe_ratio', 0):.2f}",
            "Risk-adjusted return"
        )
    
    # Portfolio Composition
    st.subheader("Portfolio Composition")
    
    if portfolio.positions:
        positions_df = portfolio.get_positions_summary()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Pie chart of positions
            fig = px.pie(
                positions_df,
                values='Market Value',
                names='Ticker',
                title='Position Allocation',
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Add cash to allocation
            total = portfolio.total_value
            allocation_data = [
                {'Asset': ticker, 'Value': pos.market_value, 'Type': 'Stock'}
                for ticker, pos in portfolio.positions.items()
            ]
            allocation_data.append({
                'Asset': 'Cash',
                'Value': portfolio.cash,
                'Type': 'Cash'
            })
            
            alloc_df = pd.DataFrame(allocation_data)
            alloc_df['Weight'] = alloc_df['Value'] / total * 100
            
            st.dataframe(
                alloc_df[['Asset', 'Value', 'Weight']].style.format({
                    'Value': '${:,.2f}',
                    'Weight': '{:.1f}%'
                }),
                use_container_width=True
            )
        
        # Positions detail table
        st.subheader("Position Details")
        st.dataframe(
            positions_df.style.format({
                'Avg Cost': '${:.2f}',
                'Current Price': '${:.2f}',
                'Market Value': '${:,.2f}',
                'Cost Basis': '${:,.2f}',
                'Unrealized P&L': '${:,.2f}',
                'P&L %': '{:.2f}%',
                'Weight': '{:.1f}%'
            }).applymap(
                lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else 'color: red',
                subset=['Unrealized P&L', 'P&L %']
            ),
            use_container_width=True
        )
    else:
        st.info("No positions in portfolio. Run a backtest or execute some trades!")


def render_forecast_tab(config):
    """Render the forecasting tab."""
    st.header("🔮 Stock Forecasts")
    
    selected_ticker = st.selectbox(
        "Select Ticker",
        options=config['tickers']
    )
    
    if st.button("Generate Forecast", type="primary"):
        with st.spinner(f"Generating forecast for {selected_ticker}..."):
            try:
                # Fetch data
                fetcher = DataFetcher(lookback_days=180)
                data = fetcher.fetch_stock_data(selected_ticker)
                
                # Generate forecast
                forecaster = ChronosForecaster(
                    model_size='bolt-small',
                    prediction_length=5
                )
                forecast = forecaster.forecast(data['Close'], selected_ticker)
                
                # Store in session state
                st.session_state.forecasts[selected_ticker] = {
                    'data': data,
                    'forecast': forecast
                }
                
                st.success(f"Forecast generated for {selected_ticker}!")
                
            except Exception as e:
                st.error(f"Error generating forecast: {e}")
    
    # Display forecast if available
    if selected_ticker in st.session_state.forecasts:
        forecast_data = st.session_state.forecasts[selected_ticker]
        data = forecast_data['data']
        forecast = forecast_data['forecast']
        
        # Forecast metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            color = "green" if forecast.expected_return > 0 else "red"
            st.metric(
                "Expected Return",
                f"{forecast.expected_return:+.2%}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Forecast Volatility",
                f"{forecast.forecast_volatility:.2%}"
            )
        
        with col3:
            st.metric(
                "Confidence Score",
                f"{forecast.confidence_score:.2f}",
                delta=f"{'High' if forecast.confidence_score > 0.7 else 'Medium' if forecast.confidence_score > 0.5 else 'Low'}"
            )
        
        # Forecast chart
        fig = make_subplots(rows=1, cols=1)
        
        # Historical prices (last 60 days)
        historical = data['Close'].iloc[-60:]
        fig.add_trace(go.Scatter(
            x=historical.index,
            y=historical.values,
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast.forecast_dates,
            y=forecast.point_forecast,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='orange', dash='dash')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=list(forecast.forecast_dates) + list(forecast.forecast_dates[::-1]),
            y=list(forecast.upper_bound) + list(forecast.lower_bound[::-1]),
            fill='toself',
            fillcolor='rgba(255, 165, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='80% Confidence Interval'
        ))
        
        fig.update_layout(
            title=f"{selected_ticker} Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Generate signal
        strategy = TradingStrategy(
            return_threshold_buy=config['return_threshold'],
            max_position_weight=config['max_position']
        )
        signal = strategy.generate_signal(
            forecast=forecast,
            portfolio_value=st.session_state.portfolio.total_value,
            current_position=0
        )
        
        # Signal display
        st.subheader("Trading Signal")
        
        signal_color = {
            Signal.BUY: "🟢",
            Signal.SELL: "🔴",
            Signal.HOLD: "🟡"
        }
        
        st.markdown(f"""
        ### {signal_color[signal.signal]} {signal.signal.value}
        
        - **Target Weight:** {signal.target_weight:.1%}
        - **Position Size:** ${signal.position_size:,.2f}
        - **Shares:** {signal.shares}
        - **Stop Loss:** ${signal.stop_loss:.2f}
        - **Take Profit:** ${signal.take_profit:.2f}
        - **Rationale:** {signal.rationale}
        """)


def render_backtest_tab(config):
    """Render the backtesting tab."""
    st.header("📈 Backtest Results")
    
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest... This may take a few minutes."):
            try:
                backtest_config = BacktestConfig(
                    start_date=config['start_date'],
                    end_date=config['end_date'],
                    tickers=config['tickers'],
                    initial_capital=config['initial_capital'],
                    rebalance_frequency=config['rebalance_frequency'],
                    forecast_horizon=5
                )
                
                # Create strategy with user parameters
                strategy = TradingStrategy(
                    return_threshold_buy=config['return_threshold'],
                    max_position_weight=config['max_position']
                )
                
                backtester = Backtester(strategy=strategy)
                result = backtester.run_backtest(backtest_config)
                
                st.session_state.backtest_result = result
                st.session_state.portfolio = Portfolio(config['initial_capital'])
                
                # Update portfolio with final state from backtest
                if result.portfolio_history:
                    final_snapshot = result.portfolio_history[-1]
                    st.session_state.portfolio.cash = final_snapshot.cash
                
                st.success("Backtest completed!")
                
            except Exception as e:
                st.error(f"Backtest error: {e}")
                st.info("Note: Backtesting requires market data access. Using simulated results for demo.")
                # Create demo results
                create_demo_backtest_result(config)
    
    # Display results
    result = st.session_state.backtest_result
    
    if result is not None:
        metrics = result.performance_metrics
        
        # Performance metrics
        st.subheader("Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        
        with col2:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
            st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}")
        
        with col3:
            st.metric("Volatility", f"{metrics.get('volatility', 0):.2%}")
            st.metric("Beta", f"{metrics.get('beta', 1):.2f}")
        
        with col4:
            st.metric("Win Rate", f"{metrics.get('win_rate', 0):.2%}")
            st.metric("Total Trades", metrics.get('num_trades', 0))
        
        # Equity curve
        st.subheader("Equity Curve")
        
        if result.portfolio_history:
            equity_data = pd.DataFrame([
                {'Date': s.timestamp, 'Portfolio Value': s.total_value}
                for s in result.portfolio_history
            ])
            
            fig = px.line(
                equity_data,
                x='Date',
                y='Portfolio Value',
                title='Portfolio Value Over Time'
            )
            fig.update_traces(line_color='#1f77b4')
            fig.add_hline(
                y=config['initial_capital'],
                line_dash="dash",
                line_color="gray",
                annotation_text="Initial Capital"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Returns distribution
        col1, col2 = st.columns(2)
        
        with col1:
            if len(result.daily_returns) > 0:
                fig = px.histogram(
                    result.daily_returns,
                    nbins=50,
                    title='Daily Returns Distribution'
                )
                fig.update_layout(xaxis_title='Daily Return', yaxis_title='Frequency')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if len(result.daily_returns) > 1:
                cumulative = (1 + result.daily_returns).cumprod()
                drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
                
                fig = px.area(
                    x=drawdown.index,
                    y=drawdown.values,
                    title='Drawdown Over Time'
                )
                fig.update_traces(fill='tozeroy', line_color='red')
                fig.update_layout(xaxis_title='Date', yaxis_title='Drawdown')
                st.plotly_chart(fig, use_container_width=True)
        
        # Trade history
        st.subheader("Trade History")
        
        if not result.trade_history.empty:
            st.dataframe(
                result.trade_history.style.format({
                    'Price': '${:.2f}',
                    'Value': '${:,.2f}',
                    'Confidence': '{:.2f}'
                }),
                use_container_width=True
            )
        else:
            st.info("No trades executed during backtest period.")
        
        # Download report
        st.subheader("Download Report")
        report_text = generate_backtest_report(result)
        st.download_button(
            label="📄 Download Full Report",
            data=report_text,
            file_name="backtest_report.txt",
            mime="text/plain"
        )
    else:
        st.info("Click 'Run Backtest' to generate results.")


def create_demo_backtest_result(config):
    """Create demo backtest results for display when real data unavailable."""
    from backtester import BacktestResult
    
    # Generate synthetic data
    dates = pd.date_range(config['start_date'], config['end_date'], freq='B')
    n_days = len(dates)
    
    # Simulate returns
    np.random.seed(42)
    daily_returns = np.random.normal(0.0005, 0.015, n_days)
    cumulative = config['initial_capital'] * (1 + daily_returns).cumprod()
    
    # Create portfolio snapshots
    from portfolio import PortfolioSnapshot
    snapshots = []
    for i, (date, value) in enumerate(zip(dates, cumulative)):
        snapshots.append(PortfolioSnapshot(
            timestamp=date,
            total_value=value,
            cash=value * 0.2,
            positions_value=value * 0.8,
            positions={'AAPL': value * 0.3, 'MSFT': value * 0.25, 'GOOGL': value * 0.25},
            daily_return=daily_returns[i],
            cumulative_return=(value - config['initial_capital']) / config['initial_capital'],
            beta=1.05
        ))
    
    # Create result
    result = BacktestResult(
        config=BacktestConfig(
            start_date=config['start_date'],
            end_date=config['end_date'],
            tickers=config['tickers'],
            initial_capital=config['initial_capital'],
            rebalance_frequency=config['rebalance_frequency']
        ),
        portfolio_history=snapshots,
        trade_history=pd.DataFrame(),
        performance_metrics={
            'total_return': (cumulative[-1] - config['initial_capital']) / config['initial_capital'],
            'sharpe_ratio': 1.2,
            'sortino_ratio': 1.5,
            'max_drawdown': -0.08,
            'volatility': 0.18,
            'beta': 1.05,
            'win_rate': 0.55,
            'num_trades': 45,
            'total_pnl': cumulative[-1] - config['initial_capital'],
            'realized_pnl': (cumulative[-1] - config['initial_capital']) * 0.7,
            'unrealized_pnl': (cumulative[-1] - config['initial_capital']) * 0.3,
        },
        benchmark_returns=pd.Series(np.random.normal(0.0004, 0.012, n_days), index=dates),
        daily_returns=pd.Series(daily_returns, index=dates),
        positions_over_time=pd.DataFrame()
    )
    
    st.session_state.backtest_result = result


def render_analytics_tab(config):
    """Render the analytics and risk metrics tab."""
    st.header("📉 Risk Analytics")
    
    result = st.session_state.backtest_result
    
    if result is None:
        st.info("Run a backtest first to see analytics.")
        return
    
    metrics = result.performance_metrics
    
    # Risk metrics table
    st.subheader("Risk Metrics Summary")
    
    risk_data = {
        'Metric': [
            'Value at Risk (95%)',
            'Expected Shortfall',
            'Beta',
            'Tracking Error',
            'Information Ratio',
            'Max Drawdown Duration'
        ],
        'Value': [
            f"{np.percentile(result.daily_returns, 5):.2%}" if len(result.daily_returns) > 0 else 'N/A',
            f"{result.daily_returns[result.daily_returns < np.percentile(result.daily_returns, 5)].mean():.2%}" if len(result.daily_returns) > 0 else 'N/A',
            f"{metrics.get('beta', 1):.2f}",
            f"{(result.daily_returns - result.benchmark_returns).std() * np.sqrt(252):.2%}" if len(result.daily_returns) > 0 and len(result.benchmark_returns) > 0 else 'N/A',
            f"{metrics.get('information_ratio', 0):.2f}",
            'N/A'
        ],
        'Description': [
            'Max daily loss at 95% confidence',
            'Average loss when VaR is exceeded',
            'Sensitivity to market movements',
            'Deviation from benchmark',
            'Risk-adjusted excess return',
            'Longest drawdown period'
        ]
    }
    
    st.table(pd.DataFrame(risk_data))
    
    # Rolling metrics
    st.subheader("Rolling Performance")
    
    if len(result.daily_returns) > 20:
        returns_series = result.daily_returns
        
        # Rolling Sharpe
        rolling_sharpe = returns_series.rolling(20).mean() / returns_series.rolling(20).std() * np.sqrt(252)
        
        # Rolling volatility
        rolling_vol = returns_series.rolling(20).std() * np.sqrt(252)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                title='20-Day Rolling Sharpe Ratio'
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(
                x=rolling_vol.index,
                y=rolling_vol.values,
                title='20-Day Rolling Volatility (Annualized)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    
    if len(result.daily_returns) > 0 and len(result.benchmark_returns) > 0:
        common_idx = result.daily_returns.index.intersection(result.benchmark_returns.index)
        
        if len(common_idx) > 10:
            fig = px.scatter(
                x=result.benchmark_returns.loc[common_idx],
                y=result.daily_returns.loc[common_idx],
                title='Strategy vs Benchmark Returns',
                labels={'x': 'Benchmark Return', 'y': 'Strategy Return'},
                trendline='ols'
            )
            st.plotly_chart(fig, use_container_width=True)


def main():
    """Main dashboard application."""
    initialize_session_state()
    
    # Title
    st.title("🚀 Mock Trading Dashboard")
    st.markdown("*Powered by Chronos-2 Forecasting*")
    
    # Sidebar
    config = render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🔮 Forecasts",
        "📈 Backtest",
        "📉 Analytics"
    ])
    
    with tab1:
        render_overview_tab(config)
    
    with tab2:
        render_forecast_tab(config)
    
    with tab3:
        render_backtest_tab(config)
    
    with tab4:
        render_analytics_tab(config)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "*Disclaimer: This is a mock trading system for educational purposes only. "
        "Do not use for actual trading decisions.*"
    )


if __name__ == "__main__":
    main()
