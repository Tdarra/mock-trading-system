# 🚀 Mock Stock Trading System

A mock stock trading system powered by Amazon Chronos-2 for time series forecasting, with a cloud-hosted Streamlit dashboard for performance visualization.

## Overview

This system implements a complete mock trading pipeline:

1. **Data Fetching** - Retrieves up to 1 year of historical stock data via yfinance
2. **Forecasting** - Uses Chronos-2 (Amazon's pretrained time series model) to predict future prices
3. **Signal Generation** - Classifies forecasts as BUY/SELL/HOLD with position sizing
4. **Trade Execution** - Executes mock trades and tracks portfolio metrics
5. **Backtesting** - Evaluates historical strategy performance

## 🏗️ Architecture

```
mock-trading-system/
├── src/
│   ├── data_fetcher.py    # Stock data retrieval (yfinance)
│   ├── forecaster.py      # Chronos-2 time series forecasting
│   ├── strategy.py        # Trading signals & position sizing
│   ├── portfolio.py       # Portfolio management & P&L tracking
│   ├── backtester.py      # Historical strategy evaluation
│   └── dashboard.py       # Streamlit web dashboard
├── tests/
│   └── test_*.py          # Unit tests
├── data/                  # Data storage (gitignored)
├── notebooks/             # Jupyter notebooks for exploration
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── requirements.txt       # Python dependencies
└── README.md
```

## 🚀 Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mock-trading-system.git
cd mock-trading-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run src/dashboard.py
```

### Docker

```bash
# Build the image
docker build -t mock-trading .

# Run the container
docker run -p 8501:8501 mock-trading
```

## ☁️ Cloud Deployment (Streamlit Cloud)

1. Fork this repository to your GitHub account
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app" and connect your GitHub repository
4. Set the main file path to `src/dashboard.py`
5. Deploy!

Your dashboard will be available at `https://your-app-name.streamlit.app`

## 📊 Features

### Data Fetching
- Fetches OHLCV data from Yahoo Finance
- Up to 1 year lookback period
- Automatic computation of technical indicators (SMA, RSI, volatility)
- Market benchmark data (SPY) for beta calculations

### Forecasting with Chronos-2
- Pretrained transformer-based time series model
- Probabilistic forecasts with confidence intervals
- Multiple model sizes available (tiny to large)
- Bolt variants for faster inference

### Trading Strategy
- Signal classification based on expected returns
- Kelly criterion position sizing
- Risk management with stop-loss and take-profit levels
- Configurable thresholds and constraints

### Portfolio Management
- Real-time position tracking
- Transaction costs and slippage modeling
- P&L calculation (realized and unrealized)
- Portfolio beta calculation

### Backtesting
- Walk-forward analysis support
- Comprehensive performance metrics:
  - Sharpe/Sortino/Calmar ratios
  - Maximum drawdown
  - Win rate
  - Alpha and beta
- Downloadable reports

## 📈 Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **Overview** | Portfolio composition, positions, and key metrics |
| **Forecasts** | Generate Chronos-2 forecasts for individual stocks |
| **Backtest** | Run historical backtests and view performance |
| **Analytics** | Risk metrics, rolling performance, correlation analysis |

## ⚙️ Configuration

### Strategy Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `return_threshold_buy` | 2% | Min expected return for buy signal |
| `return_threshold_sell` | -2% | Max expected return for sell signal |
| `min_confidence` | 0.5 | Minimum forecast confidence |
| `max_position_weight` | 25% | Maximum single position size |
| `risk_per_trade` | 2% | Risk budget per trade |
| `kelly_fraction` | 0.25 | Fraction of Kelly optimal to use |

### Backtest Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_capital` | $100,000 | Starting portfolio value |
| `rebalance_frequency` | weekly | Trading frequency |
| `transaction_cost` | 0.1% | Cost per trade |
| `slippage` | 0.05% | Assumed slippage |

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📦 API Usage

### Basic Example

```python
from src.data_fetcher import DataFetcher
from src.forecaster import ChronosForecaster
from src.strategy import TradingStrategy
from src.portfolio import Portfolio

# Initialize components
fetcher = DataFetcher(lookback_days=180)
forecaster = ChronosForecaster(model_size='bolt-small')
strategy = TradingStrategy()
portfolio = Portfolio(initial_capital=100000)

# Fetch data and generate forecast
data = fetcher.fetch_stock_data('AAPL')
forecast = forecaster.forecast(data['Close'], 'AAPL')

# Generate trading signal
signal = strategy.generate_signal(
    forecast=forecast,
    portfolio_value=portfolio.total_value
)

# Execute trade
if signal.signal.value == 'BUY':
    portfolio.execute_trade(signal, current_price=data['Close'].iloc[-1])

print(f"Portfolio value: ${portfolio.total_value:,.2f}")
```

### Running a Backtest

```python
from src.backtester import Backtester, BacktestConfig

config = BacktestConfig(
    start_date='2024-01-01',
    end_date='2024-06-30',
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    initial_capital=100000,
    rebalance_frequency='weekly'
)

backtester = Backtester()
result = backtester.run_backtest(config)

print(f"Total Return: {result.performance_metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {result.performance_metrics['sharpe_ratio']:.2f}")
```

## 🔧 Development

### Adding New Forecasting Models

Extend the `ChronosForecaster` class or create a new forecaster implementing:

```python
class CustomForecaster:
    def forecast(self, data: pd.Series, ticker: str) -> ForecastResult:
        # Your implementation
        pass
```

### Adding New Strategies

Extend the `TradingStrategy` class:

```python
class MomentumStrategy(TradingStrategy):
    def _classify_signal(self, expected_return, confidence, volatility):
        # Custom signal logic
        pass
```

## ⚠️ Disclaimer

**This is a mock trading system for educational and research purposes only.**

- Do not use for actual trading decisions
- Past performance does not guarantee future results
- The forecasts are probabilistic and may be inaccurate
- Always consult a financial advisor for real investment decisions

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 References

- [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)
- [Amazon Chronos GitHub](https://github.com/amazon-science/chronos-forecasting)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)
- [Streamlit Documentation](https://docs.streamlit.io)
