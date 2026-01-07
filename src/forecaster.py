"""
Forecaster Module
Uses Amazon Chronos-2 via AWS Bedrock for time series forecasting of stock prices.
"""

import boto3
import json
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
import logging
import os
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
    Stock price forecaster using Amazon Chronos-2 via AWS Bedrock.
    
    Chronos-2 is Amazon's 120M parameter time series foundation model
    hosted on AWS Bedrock for scalable, serverless inference.
    
    Required environment variables:
        AWS_ACCESS_KEY_ID: Your AWS access key
        AWS_SECRET_ACCESS_KEY: Your AWS secret key
        AWS_REGION: AWS region (default: us-east-1)
    """
    
    # Bedrock model ID for Chronos-2
    BEDROCK_MODEL_ID = "amazon.chronos-2"
    
    def __init__(
        self, 
        region: Optional[str] = None,
        context_length: int = 512,
        prediction_length: int = 5
    ):
        """
        Initialize the Chronos forecaster with AWS Bedrock.
        
        Args:
            region: AWS region for Bedrock (default from env or us-east-1)
            context_length: Number of historical points for context (max 512)
            prediction_length: Number of days to forecast
        """
        self.region = region or os.environ.get('AWS_REGION', 'us-east-1')
        self.context_length = min(context_length, 512)  # Chronos-2 max context
        self.prediction_length = prediction_length
        
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the AWS Bedrock Runtime client."""
        try:
            self.client = boto3.client(
                'bedrock-runtime',
                region_name=self.region,
                aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
            )
            logger.info(f"AWS Bedrock client initialized (region: {self.region})")
        except Exception as e:
            logger.warning(f"Failed to initialize Bedrock client: {e}")
            self.client = None
    
    def forecast(
        self,
        data: pd.Series,
        ticker: str,
        num_samples: int = 20
    ) -> ForecastResult:
        """
        Generate probabilistic forecasts for a time series using Bedrock.
        
        Args:
            data: Historical price data (pandas Series with DatetimeIndex)
            ticker: Stock ticker symbol
            num_samples: Number of forecast samples to generate
            
        Returns:
            ForecastResult with point forecasts and uncertainty bounds
        """
        if self.client is None:
            logger.warning("Using fallback forecaster (Bedrock not available)")
            return self._fallback_forecast(data, ticker)
        
        # Prepare context data
        context = data.iloc[-self.context_length:].values.tolist()
        
        logger.info(f"Forecasting {ticker} with {len(context)} context points via Bedrock")
        
        try:
            # Prepare request payload for Chronos-2 on Bedrock
            request_body = {
                "inferenceConfig": {
                    "numSamples": num_samples
                },
                "forecast": {
                    "targetTimeSeries": [
                        {
                            "values": context
                        }
                    ],
                    "predictionLength": self.prediction_length
                }
            }
            
            # Call Bedrock
            response = self.client.invoke_model(
                modelId=self.BEDROCK_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            # Extract forecast samples from response
            forecasts = response_body.get('forecast', {}).get('predictions', [])
            
            if not forecasts:
                logger.warning("Empty forecast response, using fallback")
                return self._fallback_forecast(data, ticker)
            
            # Chronos returns samples for each time series
            samples = np.array(forecasts[0].get('samples', []))
            
            if samples.ndim == 1:
                samples = samples.reshape(1, -1)
            
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
            forecast_volatility = samples[:, -1].std() / last_price if samples.shape[0] > 1 else 0.02
            
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
            
        except Exception as e:
            logger.error(f"Bedrock API error: {e}")
            return self._fallback_forecast(data, ticker)
    
    def _fallback_forecast(
        self, 
        data: pd.Series, 
        ticker: str
    ) -> ForecastResult:
        """
        Simple fallback forecaster when Bedrock is not available.
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
        if samples.shape[0] < 2:
            return 0.5
            
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
    
    # Create forecaster (requires AWS credentials)
    # Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION env vars
    forecaster = ChronosForecaster(prediction_length=5)
    
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
