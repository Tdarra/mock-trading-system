"""
Forecaster Module
Uses Amazon Chronos-2 for time series forecasting of stock prices.
"""

import torch
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Container for forecast results."""
    ticker: str
    forecast_dates: pd.DatetimeIndex
    point_forecast: np.ndarray
    lower_bound: np.ndarray  # 10th percentile
    upper_bound: np.ndarray  # 90th percentile
    median_forecast: np.ndarray
    last_actual_price: float
    last_actual_date: pd.Timestamp
    expected_return: float
    forecast_volatility: float
    confidence_score: float


class ChronosForecaster:
    """
    Stock price forecaster using Amazon Chronos-2.
    
    Chronos-2 is a family of pretrained time series forecasting models
    that can generate probabilistic forecasts for unseen time series.
    """
    
    # Available Chronos-2 model sizes
    MODEL_SIZES = {
        'tiny': 'amazon/chronos-t5-tiny',
        'mini': 'amazon/chronos-t5-mini', 
        'small': 'amazon/chronos-t5-small',
        'base': 'amazon/chronos-t5-base',
        'large': 'amazon/chronos-t5-large',
        # Chronos-2 models (Bolt series for faster inference)
        'bolt-tiny': 'amazon/chronos-bolt-tiny',
        'bolt-mini': 'amazon/chronos-bolt-mini',
        'bolt-small': 'amazon/chronos-bolt-small',
        'bolt-base': 'amazon/chronos-bolt-base',
    }
    
    def __init__(
        self, 
        model_size: str = 'bolt-small',
        device: Optional[str] = None,
        context_length: int = 64,
        prediction_length: int = 5
    ):
        """
        Initialize the Chronos forecaster.
        
        Args:
            model_size: Size of Chronos model to use
            device: Device for inference ('cuda', 'cpu', or None for auto)
            context_length: Number of historical points for context
            prediction_length: Number of days to forecast
        """
        self.model_size = model_size
        self.context_length = context_length
        self.prediction_length = prediction_length
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the Chronos model."""
        try:
            from chronos import ChronosPipeline
            
            model_id = self.MODEL_SIZES.get(self.model_size, self.model_size)
            logger.info(f"Loading Chronos model: {model_id}")
            
            self.pipeline = ChronosPipeline.from_pretrained(
                model_id,
                device_map=self.device,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.bfloat16,
            )
            
            logger.info("Chronos model loaded successfully")
            
        except ImportError:
            logger.warning("Chronos not installed. Using fallback forecaster.")
            self.pipeline = None
        except Exception as e:
            logger.error(f"Error loading Chronos model: {e}")
            self.pipeline = None
    
    def forecast(
        self,
        data: pd.Series,
        ticker: str,
        num_samples: int = 20
    ) -> ForecastResult:
        """
        Generate probabilistic forecasts for a time series.
        
        Args:
            data: Historical price data (pandas Series with DatetimeIndex)
            ticker: Stock ticker symbol
            num_samples: Number of forecast samples to generate
            
        Returns:
            ForecastResult with point forecasts and uncertainty bounds
        """
        if self.pipeline is None:
            logger.warning("Using fallback forecaster (Chronos not available)")
            return self._fallback_forecast(data, ticker)
        
        # Prepare context data
        context = data.iloc[-self.context_length:].values
        context_tensor = torch.tensor(context, dtype=torch.float32)
        
        logger.info(f"Forecasting {ticker} with {len(context)} context points")
        
        # Generate forecasts
        with torch.no_grad():
            forecast_samples = self.pipeline.predict(
                context_tensor,
                prediction_length=self.prediction_length,
                num_samples=num_samples
            )
        
        # Convert to numpy
        samples = forecast_samples.numpy()
        
        # Calculate statistics
        point_forecast = samples.mean(axis=0)
        median_forecast = np.median(samples, axis=0)
        lower_bound = np.percentile(samples, 10, axis=0)
        upper_bound = np.percentile(samples, 90, axis=0)
        
        # Generate forecast dates
        last_date = data.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=self.prediction_length,
            freq='B'  # Business days
        )
        
        # Calculate expected return and volatility
        last_price = data.iloc[-1]
        expected_return = (median_forecast[-1] - last_price) / last_price
        forecast_volatility = samples[:, -1].std() / last_price
        
        # Confidence score based on forecast uncertainty
        confidence_score = self._calculate_confidence(
            samples, last_price, expected_return
        )
        
        return ForecastResult(
            ticker=ticker,
            forecast_dates=forecast_dates,
            point_forecast=point_forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            median_forecast=median_forecast,
            last_actual_price=last_price,
            last_actual_date=last_date,
            expected_return=expected_return,
            forecast_volatility=forecast_volatility,
            confidence_score=confidence_score
        )
    
    def _fallback_forecast(
        self, 
        data: pd.Series, 
        ticker: str
    ) -> ForecastResult:
        """
        Simple fallback forecaster when Chronos is not available.
        Uses exponential moving average and historical volatility.
        """
        logger.info(f"Using fallback forecast for {ticker}")
        
        # Calculate historical statistics
        returns = data.pct_change().dropna()
        mean_return = returns.mean()
        volatility = returns.std()
        
        last_price = data.iloc[-1]
        last_date = data.index[-1]
        
        # Generate forecast dates
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=self.prediction_length,
            freq='B'
        )
        
        # Simple random walk with drift forecast
        np.random.seed(42)  # For reproducibility
        num_samples = 100
        samples = np.zeros((num_samples, self.prediction_length))
        
        for i in range(num_samples):
            price = last_price
            for t in range(self.prediction_length):
                daily_return = np.random.normal(mean_return, volatility)
                price = price * (1 + daily_return)
                samples[i, t] = price
        
        point_forecast = samples.mean(axis=0)
        median_forecast = np.median(samples, axis=0)
        lower_bound = np.percentile(samples, 10, axis=0)
        upper_bound = np.percentile(samples, 90, axis=0)
        
        expected_return = (median_forecast[-1] - last_price) / last_price
        forecast_volatility = samples[:, -1].std() / last_price
        confidence_score = 0.5  # Lower confidence for fallback
        
        return ForecastResult(
            ticker=ticker,
            forecast_dates=forecast_dates,
            point_forecast=point_forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            median_forecast=median_forecast,
            last_actual_price=last_price,
            last_actual_date=last_date,
            expected_return=expected_return,
            forecast_volatility=forecast_volatility,
            confidence_score=confidence_score
        )
    
    def _calculate_confidence(
        self,
        samples: np.ndarray,
        last_price: float,
        expected_return: float
    ) -> float:
        """
        Calculate confidence score for the forecast.
        
        Based on:
        - Forecast uncertainty (spread of samples)
        - Direction consistency across samples
        - Magnitude of expected move
        """
        final_prices = samples[:, -1]
        
        # Direction consistency: what fraction of samples agree on direction?
        if expected_return > 0:
            direction_consistency = (final_prices > last_price).mean()
        else:
            direction_consistency = (final_prices < last_price).mean()
        
        # Relative uncertainty: lower is better
        coefficient_of_variation = final_prices.std() / final_prices.mean()
        uncertainty_score = max(0, 1 - coefficient_of_variation)
        
        # Combine scores
        confidence = 0.6 * direction_consistency + 0.4 * uncertainty_score
        
        return min(max(confidence, 0), 1)  # Clamp to [0, 1]
    
    def batch_forecast(
        self,
        data_dict: dict,
        num_samples: int = 20
    ) -> dict:
        """
        Generate forecasts for multiple tickers.
        
        Args:
            data_dict: Dictionary mapping ticker to price Series
            num_samples: Number of forecast samples
            
        Returns:
            Dictionary mapping ticker to ForecastResult
        """
        results = {}
        for ticker, data in data_dict.items():
            try:
                results[ticker] = self.forecast(data['Close'], ticker, num_samples)
            except Exception as e:
                logger.error(f"Error forecasting {ticker}: {e}")
        return results


if __name__ == "__main__":
    # Test the forecaster with sample data
    import sys
    sys.path.insert(0, '/home/claude/mock-trading-system/src')
    from data_fetcher import DataFetcher
    
    # Fetch data
    fetcher = DataFetcher(lookback_days=180)
    data = fetcher.fetch_stock_data('AAPL')
    
    # Create forecaster (will use fallback if Chronos not installed)
    forecaster = ChronosForecaster(
        model_size='bolt-small',
        prediction_length=5
    )
    
    # Generate forecast
    result = forecaster.forecast(data['Close'], 'AAPL')
    
    print(f"\nForecast for {result.ticker}")
    print(f"Last actual price: ${result.last_actual_price:.2f}")
    print(f"Expected return: {result.expected_return:.2%}")
    print(f"Forecast volatility: {result.forecast_volatility:.2%}")
    print(f"Confidence score: {result.confidence_score:.2f}")
    print(f"\nPoint forecasts:")
    for date, price in zip(result.forecast_dates, result.point_forecast):
        print(f"  {date.strftime('%Y-%m-%d')}: ${price:.2f}")
