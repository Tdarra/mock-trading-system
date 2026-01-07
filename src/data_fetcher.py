"""
Data Fetcher Module
Fetches historical stock data using yfinance with up to 1 year lookback.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches and preprocesses stock data for trading and forecasting."""
    
    # S&P 500 proxy for market returns (used for beta calculation)
    MARKET_TICKER = "SPY"
    
    def __init__(self, lookback_days: int = 365):
        """
        Initialize the data fetcher.
        
        Args:
            lookback_days: Number of days to look back for historical data (max 365)
        """
        self.lookback_days = min(lookback_days, 365)
        
    def fetch_stock_data(
        self, 
        ticker: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical stock data for a given ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format (optional)
            end_date: End date in 'YYYY-MM-DD' format (optional)
            interval: Data interval ('1d', '1h', '5m', etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            start = datetime.now() - timedelta(days=self.lookback_days)
            start_date = start.strftime('%Y-%m-%d')
        
        logger.info(f"Fetching {ticker} data from {start_date} to {end_date}")
        
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()
            
            # Clean and preprocess
            df = self._preprocess_data(df)
            logger.info(f"Successfully fetched {len(df)} rows for {ticker}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            raise
    
    def fetch_multiple_stocks(
        self, 
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> dict:
        """
        Fetch data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        data = {}
        for ticker in tickers:
            try:
                data[ticker] = self.fetch_stock_data(ticker, start_date, end_date)
            except Exception as e:
                logger.warning(f"Skipping {ticker}: {e}")
        return data
    
    def fetch_market_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch market benchmark data (SPY) for beta calculations.
        
        Returns:
            DataFrame with market returns
        """
        return self.fetch_stock_data(self.MARKET_TICKER, start_date, end_date)
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw stock data.
        
        Args:
            df: Raw DataFrame from yfinance
            
        Returns:
            Cleaned DataFrame with additional features
        """
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Remove timezone info for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Drop any rows with NaN in critical columns
        df = df.dropna(subset=['Close', 'Volume'])
        
        # Add computed features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = pd.np.log(df['Close'] / df['Close'].shift(1))
        
        # Technical indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'])
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_latest_price(self, ticker: str) -> float:
        """Get the latest closing price for a ticker."""
        df = self.fetch_stock_data(ticker)
        if df.empty:
            raise ValueError(f"No data available for {ticker}")
        return df['Close'].iloc[-1]
    
    def prepare_forecast_data(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'Close',
        context_length: int = 64
    ) -> pd.Series:
        """
        Prepare data for Chronos-2 forecasting.
        
        Args:
            df: Stock data DataFrame
            target_col: Column to forecast
            context_length: Number of historical points for context
            
        Returns:
            Series of recent values for forecasting
        """
        if len(df) < context_length:
            logger.warning(f"Data length ({len(df)}) < context_length ({context_length})")
            context_length = len(df)
        
        return df[target_col].iloc[-context_length:]


def get_sample_tickers() -> List[str]:
    """Return a list of sample tickers for testing."""
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        'NVDA', 'TSLA', 'JPM', 'V', 'JNJ'
    ]


if __name__ == "__main__":
    # Test the data fetcher
    fetcher = DataFetcher(lookback_days=180)
    
    # Fetch single stock
    aapl_data = fetcher.fetch_stock_data('AAPL')
    print(f"AAPL data shape: {aapl_data.shape}")
    print(aapl_data.tail())
    
    # Fetch market data
    market_data = fetcher.fetch_market_data()
    print(f"\nMarket (SPY) data shape: {market_data.shape}")
