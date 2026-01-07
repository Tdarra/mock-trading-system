"""
FastAPI Backend for Mock Trading System
Deployed as Vercel Serverless Function
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_fetcher import DataFetcher
from forecaster import ChronosForecaster, ForecastResult
from strategy import TradingStrategy, TradingSignal
from portfolio import Portfolio
from backtester import Backtester

app = FastAPI(
    title="Mock Trading System API",
    description="Stock forecasting and trading API powered by Chronos-2",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
data_fetcher = DataFetcher(lookback_days=365)
forecaster = ChronosForecaster(prediction_length=5)
strategy = TradingStrategy()


# --- Request/Response Models ---

class ForecastRequest(BaseModel):
    ticker: str
    horizon: int = 5

class TradeRequest(BaseModel):
    ticker: str
    action: str  # BUY, SELL
    shares: Optional[float] = None
    dollar_amount: Optional[float] = None

class BacktestRequest(BaseModel):
    tickers: List[str]
    start_date: str
    end_date: str
    initial_capital: float = 100000
    rebalance_frequency: str = "weekly"

class PortfolioState(BaseModel):
    cash: float
    positions: Dict[str, Any]
    total_value: float
    realized_pnl: float
    unrealized_pnl: float


# --- API Endpoints ---

@app.get("/")
async def root():
    return {
        "name": "Mock Trading System API",
        "version": "1.0.0",
        "endpoints": [
            "/api/health",
            "/api/stock/{ticker}",
            "/api/forecast",
            "/api/signal/{ticker}",
            "/api/portfolio",
            "/api/trade",
            "/api/backtest"
        ]
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "aws_configured": bool(os.environ.get('AWS_ACCESS_KEY_ID'))
    }


@app.get("/api/stock/{ticker}")
async def get_stock_data(
    ticker: str,
    days: int = Query(default=90, ge=1, le=365)
):
    """Fetch historical stock data."""
    try:
        df = data_fetcher.fetch_stock_data(ticker)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        
        # Limit to requested days
        df = df.tail(days)
        
        # Convert to JSON-serializable format
        data = {
            "ticker": ticker,
            "data_points": len(df),
            "start_date": df.index[0].isoformat(),
            "end_date": df.index[-1].isoformat(),
            "latest_price": float(df['Close'].iloc[-1]),
            "history": [
                {
                    "date": idx.isoformat(),
                    "open": float(row['Open']),
                    "high": float(row['High']),
                    "low": float(row['Low']),
                    "close": float(row['Close']),
                    "volume": int(row['Volume']),
                    "returns": float(row['Returns']) if not pd.isna(row['Returns']) else None
                }
                for idx, row in df.iterrows()
            ]
        }
        
        return data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/forecast")
async def generate_forecast(request: ForecastRequest):
    """Generate price forecast using Chronos-2."""
    try:
        # Fetch data
        df = data_fetcher.fetch_stock_data(request.ticker)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.ticker}")
        
        # Update forecaster horizon if needed
        forecaster.prediction_length = request.horizon
        
        # Generate forecast
        result = forecaster.forecast(df['Close'], request.ticker)
        
        return {
            "ticker": result.ticker,
            "last_price": float(result.last_actual_price),
            "last_date": result.last_actual_date.isoformat(),
            "expected_return": float(result.expected_return),
            "forecast_volatility": float(result.forecast_volatility),
            "confidence_score": float(result.confidence_score),
            "forecasts": [
                {
                    "date": date.isoformat(),
                    "point": float(point),
                    "median": float(median),
                    "lower": float(lower),
                    "upper": float(upper)
                }
                for date, point, median, lower, upper in zip(
                    result.forecast_dates,
                    result.point_forecast,
                    result.median_forecast,
                    result.lower_bound,
                    result.upper_bound
                )
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/signal/{ticker}")
async def get_trading_signal(ticker: str):
    """Get trading signal for a ticker."""
    try:
        # Fetch data
        df = data_fetcher.fetch_stock_data(ticker)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        
        # Generate forecast
        forecast = forecaster.forecast(df['Close'], ticker)
        
        # Generate signal
        signal = strategy.generate_signal(forecast)
        
        return {
            "ticker": ticker,
            "signal": signal.action.value,
            "confidence": float(signal.confidence),
            "target_weight": float(signal.target_weight) if signal.target_weight else None,
            "expected_return": float(signal.expected_return),
            "stop_loss": float(signal.stop_loss) if signal.stop_loss else None,
            "take_profit": float(signal.take_profit) if signal.take_profit else None,
            "reasoning": signal.reasoning,
            "timestamp": signal.timestamp.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# In-memory portfolio for demo (would use database in production)
demo_portfolio = Portfolio(initial_capital=100000)


@app.get("/api/portfolio")
async def get_portfolio():
    """Get current portfolio state."""
    summary = demo_portfolio.get_summary()
    
    return {
        "cash": float(summary['cash']),
        "positions_value": float(summary['positions_value']),
        "total_value": float(summary['total_value']),
        "realized_pnl": float(summary['realized_pnl']),
        "unrealized_pnl": float(summary['unrealized_pnl']),
        "total_pnl": float(summary['total_pnl']),
        "return_pct": float(summary['return_pct']),
        "num_positions": summary['num_positions'],
        "positions": [
            {
                "ticker": ticker,
                "shares": float(pos.shares),
                "avg_cost": float(pos.avg_cost),
                "current_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "unrealized_pnl": float(pos.unrealized_pnl),
                "unrealized_pnl_pct": float(pos.unrealized_pnl_pct)
            }
            for ticker, pos in demo_portfolio.positions.items()
        ]
    }


@app.post("/api/trade")
async def execute_trade(request: TradeRequest):
    """Execute a mock trade."""
    try:
        ticker = request.ticker.upper()
        
        # Get current price
        df = data_fetcher.fetch_stock_data(ticker)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        
        current_price = float(df['Close'].iloc[-1])
        
        # Calculate shares
        if request.shares:
            shares = request.shares
        elif request.dollar_amount:
            shares = request.dollar_amount / current_price
        else:
            raise HTTPException(status_code=400, detail="Must specify shares or dollar_amount")
        
        # Execute trade
        if request.action.upper() == "BUY":
            success = demo_portfolio.buy(ticker, shares, current_price)
            action = "bought"
        elif request.action.upper() == "SELL":
            success = demo_portfolio.sell(ticker, shares, current_price)
            action = "sold"
        else:
            raise HTTPException(status_code=400, detail="Action must be BUY or SELL")
        
        if not success:
            raise HTTPException(status_code=400, detail="Trade failed - insufficient funds or shares")
        
        return {
            "success": True,
            "action": action,
            "ticker": ticker,
            "shares": float(shares),
            "price": current_price,
            "total_value": float(shares * current_price),
            "portfolio_value": float(demo_portfolio.get_total_value())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/portfolio/reset")
async def reset_portfolio(initial_capital: float = 100000):
    """Reset the demo portfolio."""
    global demo_portfolio
    demo_portfolio = Portfolio(initial_capital=initial_capital)
    
    return {
        "success": True,
        "initial_capital": initial_capital,
        "message": "Portfolio reset successfully"
    }


@app.post("/api/backtest")
async def run_backtest(request: BacktestRequest):
    """Run a backtest simulation."""
    try:
        backtester = Backtester(
            tickers=request.tickers,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            rebalance_frequency=request.rebalance_frequency
        )
        
        results = backtester.run()
        
        # Convert results to JSON-serializable format
        return {
            "tickers": request.tickers,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "initial_capital": request.initial_capital,
            "final_value": float(results['final_value']),
            "total_return": float(results['total_return']),
            "annualized_return": float(results['annualized_return']),
            "sharpe_ratio": float(results['sharpe_ratio']),
            "sortino_ratio": float(results['sortino_ratio']),
            "max_drawdown": float(results['max_drawdown']),
            "win_rate": float(results['win_rate']),
            "num_trades": int(results['num_trades']),
            "equity_curve": [
                {"date": date, "value": float(value)}
                for date, value in zip(
                    results['equity_curve'].index.strftime('%Y-%m-%d').tolist(),
                    results['equity_curve'].values
                )
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tickers")
async def get_sample_tickers():
    """Get list of sample tickers for testing."""
    return {
        "tickers": [
            {"symbol": "AAPL", "name": "Apple Inc."},
            {"symbol": "MSFT", "name": "Microsoft Corporation"},
            {"symbol": "GOOGL", "name": "Alphabet Inc."},
            {"symbol": "AMZN", "name": "Amazon.com Inc."},
            {"symbol": "META", "name": "Meta Platforms Inc."},
            {"symbol": "NVDA", "name": "NVIDIA Corporation"},
            {"symbol": "TSLA", "name": "Tesla Inc."},
            {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
            {"symbol": "V", "name": "Visa Inc."},
            {"symbol": "JNJ", "name": "Johnson & Johnson"}
        ]
    }


# Required import for data serialization
import pandas as pd
