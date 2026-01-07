# 🚀 Mock Stock Trading System

A mock stock trading system powered by Amazon Chronos-2 (via AWS Bedrock) for time series forecasting, with a modern React dashboard deployed on Vercel.

## Overview

This system implements a complete mock trading pipeline:

1. **Data Fetching** - Retrieves up to 1 year of historical stock data via yfinance
2. **Forecasting** - Uses Chronos-2 (120M params) on AWS Bedrock to predict future prices
3. **Signal Generation** - Classifies forecasts as BUY/SELL/HOLD with position sizing
4. **Trade Execution** - Executes mock trades and tracks portfolio metrics
5. **Backtesting** - Evaluates historical strategy performance

## 🏗️ Architecture

```
mock-trading-system/
├── api/
│   └── index.py           # FastAPI backend (Vercel serverless)
├── frontend/
│   ├── src/
│   │   ├── App.jsx        # React dashboard
│   │   └── index.css      # Terminal-style CSS
│   ├── package.json       # Frontend dependencies
│   └── vite.config.js     # Vite configuration
├── src/
│   ├── data_fetcher.py    # Stock data retrieval (yfinance)
│   ├── forecaster.py      # AWS Bedrock Chronos-2 integration
│   ├── strategy.py        # Trading signals & position sizing
│   ├── portfolio.py       # Portfolio management & P&L tracking
│   └── backtester.py      # Historical strategy evaluation
├── tests/
│   └── test_*.py          # Unit tests
├── vercel.json            # Vercel deployment config
└── requirements.txt       # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

1. **AWS Account** with Bedrock access
2. **Node.js** 18+ for frontend development
3. **Python** 3.11+ for backend

### AWS Bedrock Setup

1. Enable Chronos-2 model access in AWS Bedrock console
2. Create IAM credentials with Bedrock invoke permissions
3. Set environment variables:

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION="us-east-1"
```

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/mock-trading-system.git
cd mock-trading-system

# Backend setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start FastAPI backend
uvicorn api.index:app --reload --port 8000

# Frontend setup (new terminal)
cd frontend
npm install
npm run dev
```

Dashboard available at `http://localhost:5173`

## ☁️ Vercel Deployment

### One-Click Deploy

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/yourusername/mock-trading-system)

### Manual Deployment

1. Install Vercel CLI:
```bash
npm i -g vercel
```

2. Add secrets (one-time):
```bash
vercel secrets add aws_access_key_id "your-access-key"
vercel secrets add aws_secret_access_key "your-secret-key"
vercel secrets add aws_region "us-east-1"
```

3. Deploy:
```bash
vercel --prod
```

Your dashboard will be available at `https://your-project.vercel.app`

## 📊 Features

### Data Fetching
- Fetches OHLCV data from Yahoo Finance
- **Daily granularity** (1d intervals)
- Up to 1 year lookback period
- Automatic computation of technical indicators (SMA, RSI, volatility)
- Market benchmark data (SPY) for beta calculations

### Forecasting with Chronos-2 on AWS Bedrock
- **120M parameter** foundation model for time series
- Probabilistic forecasts with confidence intervals
- Serverless, scalable inference via AWS Bedrock
- Up to 512 context points for better predictions
- Automatic fallback to statistical model if Bedrock unavailable

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

## 🖥️ Dashboard

The React dashboard features a terminal-inspired design with:

| Section | Description |
|---------|-------------|
| **Forecast Panel** | Generate Chronos-2 forecasts with confidence intervals |
| **Signal Panel** | View BUY/SELL/HOLD recommendations with reasoning |
| **Portfolio Panel** | Track positions, P&L, and portfolio metrics |
| **Trade Panel** | Execute mock trades by shares or dollar amount |

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/stock/{ticker}` | GET | Fetch historical data |
| `/api/forecast` | POST | Generate Chronos-2 forecast |
| `/api/signal/{ticker}` | GET | Get trading signal |
| `/api/portfolio` | GET | Get portfolio state |
| `/api/trade` | POST | Execute mock trade |
| `/api/backtest` | POST | Run backtest simulation |

### Example: Generate Forecast

```bash
curl -X POST https://your-app.vercel.app/api/forecast \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "horizon": 5}'
```

Response:
```json
{
  "ticker": "AAPL",
  "last_price": 185.50,
  "expected_return": 0.023,
  "confidence_score": 0.72,
  "forecasts": [
    {"date": "2024-01-15", "point": 186.20, "lower": 183.10, "upper": 189.30}
  ]
}
```

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

### AWS Bedrock Settings

| Environment Variable | Description |
|---------------------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `AWS_REGION` | AWS region (default: us-east-1) |

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📦 Python API Usage

```python
from src.data_fetcher import DataFetcher
from src.forecaster import ChronosForecaster
from src.strategy import TradingStrategy
from src.portfolio import Portfolio

# Initialize components
fetcher = DataFetcher(lookback_days=180)
forecaster = ChronosForecaster()  # Uses AWS Bedrock
strategy = TradingStrategy()
portfolio = Portfolio(initial_capital=100000)

# Fetch data and generate forecast
data = fetcher.fetch_stock_data('AAPL')
forecast = forecaster.forecast(data['Close'], 'AAPL')

# Generate trading signal
signal = strategy.generate_signal(forecast)

print(f"Signal: {signal.action.value}")
print(f"Expected Return: {forecast.expected_return:.2%}")
print(f"Confidence: {forecast.confidence_score:.0%}")
```

## 💰 Cost Considerations

AWS Bedrock pricing for Chronos-2:
- Pay per inference request
- No idle costs (serverless)
- See [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) for current rates

To minimize costs:
- Use the fallback forecaster for development/testing
- Implement caching for repeated forecasts
- Batch multiple ticker forecasts when possible

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

- [Amazon Chronos-2 on AWS Bedrock](https://aws.amazon.com/bedrock/)
- [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)
- [Vercel Documentation](https://vercel.com/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
