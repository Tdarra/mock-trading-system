"""
Strategy Module
Classifies forecasts as buy, sell, or hold and sizes positions.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import logging

from forecaster import ForecastResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeSignal:
    """Container for trade signal and sizing information."""
    ticker: str
    signal: Signal
    target_weight: float  # Target portfolio weight (0 to 1)
    position_size: float  # Dollar amount to trade
    shares: int  # Number of shares
    confidence: float  # Signal confidence
    expected_return: float
    stop_loss: float  # Stop loss price
    take_profit: float  # Take profit price
    rationale: str


class TradingStrategy:
    """
    Trading strategy that converts forecasts to actionable signals.
    
    Uses forecast expected returns, volatility, and confidence to generate
    buy/sell/hold signals with position sizing based on risk management rules.
    """
    
    def __init__(
        self,
        return_threshold_buy: float = 0.02,  # 2% expected return to buy
        return_threshold_sell: float = -0.02,  # -2% expected return to sell
        min_confidence: float = 0.5,
        max_position_weight: float = 0.25,  # Max 25% in single position
        risk_per_trade: float = 0.02,  # Risk 2% of portfolio per trade
        use_kelly_sizing: bool = True,
        kelly_fraction: float = 0.25,  # Use 25% of Kelly criterion
    ):
        """
        Initialize the trading strategy.
        
        Args:
            return_threshold_buy: Min expected return to generate buy signal
            return_threshold_sell: Max expected return to generate sell signal
            min_confidence: Minimum forecast confidence to trade
            max_position_weight: Maximum weight of any single position
            risk_per_trade: Fraction of portfolio to risk per trade
            use_kelly_sizing: Whether to use Kelly criterion for sizing
            kelly_fraction: Fraction of Kelly optimal to use (for safety)
        """
        self.return_threshold_buy = return_threshold_buy
        self.return_threshold_sell = return_threshold_sell
        self.min_confidence = min_confidence
        self.max_position_weight = max_position_weight
        self.risk_per_trade = risk_per_trade
        self.use_kelly_sizing = use_kelly_sizing
        self.kelly_fraction = kelly_fraction
    
    def generate_signal(
        self,
        forecast: ForecastResult,
        portfolio_value: float,
        current_position: float = 0,
        current_price: Optional[float] = None
    ) -> TradeSignal:
        """
        Generate a trade signal from a forecast.
        
        Args:
            forecast: Forecast result from Chronos
            portfolio_value: Current total portfolio value
            current_position: Current position in this ticker (dollars)
            current_price: Current price (uses forecast's last price if None)
            
        Returns:
            TradeSignal with signal type and sizing
        """
        price = current_price or forecast.last_actual_price
        expected_return = forecast.expected_return
        confidence = forecast.confidence_score
        volatility = forecast.forecast_volatility
        
        # Determine signal direction
        signal, rationale = self._classify_signal(
            expected_return, confidence, volatility
        )
        
        # Calculate position size
        target_weight, position_size, shares = self._size_position(
            signal=signal,
            expected_return=expected_return,
            volatility=volatility,
            confidence=confidence,
            price=price,
            portfolio_value=portfolio_value,
            current_position=current_position
        )
        
        # Calculate stop loss and take profit levels
        stop_loss, take_profit = self._calculate_risk_levels(
            signal=signal,
            price=price,
            volatility=volatility,
            expected_return=expected_return
        )
        
        return TradeSignal(
            ticker=forecast.ticker,
            signal=signal,
            target_weight=target_weight,
            position_size=position_size,
            shares=shares,
            confidence=confidence,
            expected_return=expected_return,
            stop_loss=stop_loss,
            take_profit=take_profit,
            rationale=rationale
        )
    
    def _classify_signal(
        self,
        expected_return: float,
        confidence: float,
        volatility: float
    ) -> Tuple[Signal, str]:
        """
        Classify the forecast into a trading signal.
        
        Returns:
            Tuple of (Signal, rationale string)
        """
        # Check confidence threshold
        if confidence < self.min_confidence:
            return Signal.HOLD, f"Low confidence ({confidence:.2f} < {self.min_confidence})"
        
        # Risk-adjusted return (simplified Sharpe-like)
        risk_free_rate = 0.05 / 252  # Daily risk-free rate
        risk_adjusted_return = (expected_return - risk_free_rate) / max(volatility, 0.01)
        
        # Generate signal based on expected return
        if expected_return >= self.return_threshold_buy:
            if risk_adjusted_return > 0.5:  # Good risk-adjusted return
                return Signal.BUY, f"Strong buy: {expected_return:.2%} expected return, {confidence:.2f} confidence"
            else:
                return Signal.BUY, f"Moderate buy: {expected_return:.2%} expected return"
        
        elif expected_return <= self.return_threshold_sell:
            return Signal.SELL, f"Sell signal: {expected_return:.2%} expected return"
        
        else:
            return Signal.HOLD, f"Neutral outlook: {expected_return:.2%} expected return"
    
    def _size_position(
        self,
        signal: Signal,
        expected_return: float,
        volatility: float,
        confidence: float,
        price: float,
        portfolio_value: float,
        current_position: float
    ) -> Tuple[float, float, int]:
        """
        Calculate position size using risk management rules.
        
        Returns:
            Tuple of (target_weight, position_size_dollars, num_shares)
        """
        if signal == Signal.HOLD:
            # Maintain current position
            current_weight = current_position / portfolio_value if portfolio_value > 0 else 0
            return current_weight, 0, 0
        
        if signal == Signal.SELL:
            # Reduce or close position
            if current_position > 0:
                return 0, -current_position, -int(current_position / price)
            return 0, 0, 0
        
        # BUY signal - calculate position size
        if self.use_kelly_sizing:
            target_weight = self._kelly_criterion(
                expected_return, volatility, confidence
            )
        else:
            target_weight = self._fixed_risk_sizing(
                volatility, confidence, portfolio_value
            )
        
        # Apply constraints
        target_weight = min(target_weight, self.max_position_weight)
        target_weight = max(target_weight, 0)
        
        # Calculate dollar position and shares
        target_position = target_weight * portfolio_value
        position_change = target_position - current_position
        shares = int(position_change / price) if price > 0 else 0
        
        return target_weight, position_change, shares
    
    def _kelly_criterion(
        self,
        expected_return: float,
        volatility: float,
        confidence: float
    ) -> float:
        """
        Calculate position size using Kelly criterion.
        
        Kelly formula: f* = (p * b - q) / b
        where p = probability of winning, q = probability of losing
        b = win/loss ratio
        
        We adapt this for continuous returns using:
        f* = expected_return / variance
        """
        if volatility <= 0:
            return 0
        
        variance = volatility ** 2
        kelly_optimal = expected_return / variance
        
        # Scale by confidence and safety fraction
        kelly_fraction = kelly_optimal * confidence * self.kelly_fraction
        
        return max(kelly_fraction, 0)
    
    def _fixed_risk_sizing(
        self,
        volatility: float,
        confidence: float,
        portfolio_value: float
    ) -> float:
        """
        Calculate position size using fixed risk per trade.
        
        Position size = (risk_per_trade * portfolio) / (volatility * price)
        Simplified to weight = risk_per_trade / volatility
        """
        if volatility <= 0:
            return 0
        
        base_weight = self.risk_per_trade / volatility
        
        # Scale by confidence
        adjusted_weight = base_weight * confidence
        
        return adjusted_weight
    
    def _calculate_risk_levels(
        self,
        signal: Signal,
        price: float,
        volatility: float,
        expected_return: float
    ) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels.
        
        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        if signal == Signal.HOLD or signal == Signal.SELL:
            return 0, 0
        
        # Stop loss at 2x daily volatility
        stop_loss_pct = 2 * volatility
        stop_loss = price * (1 - stop_loss_pct)
        
        # Take profit at expected return target
        take_profit_pct = max(expected_return, 0.05)  # At least 5%
        take_profit = price * (1 + take_profit_pct)
        
        return round(stop_loss, 2), round(take_profit, 2)
    
    def generate_signals_batch(
        self,
        forecasts: dict,
        portfolio_value: float,
        current_positions: Optional[dict] = None
    ) -> List[TradeSignal]:
        """
        Generate signals for multiple forecasts.
        
        Args:
            forecasts: Dictionary mapping ticker to ForecastResult
            portfolio_value: Current portfolio value
            current_positions: Dictionary mapping ticker to position value
            
        Returns:
            List of TradeSignals
        """
        current_positions = current_positions or {}
        signals = []
        
        for ticker, forecast in forecasts.items():
            position = current_positions.get(ticker, 0)
            signal = self.generate_signal(
                forecast=forecast,
                portfolio_value=portfolio_value,
                current_position=position
            )
            signals.append(signal)
        
        return signals
    
    def rank_signals(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """
        Rank signals by attractiveness (expected return * confidence).
        
        Args:
            signals: List of trade signals
            
        Returns:
            Sorted list with best opportunities first
        """
        def score(signal: TradeSignal) -> float:
            if signal.signal == Signal.HOLD:
                return 0
            return signal.expected_return * signal.confidence
        
        return sorted(signals, key=score, reverse=True)


class RiskManager:
    """Manages portfolio-level risk constraints."""
    
    def __init__(
        self,
        max_portfolio_beta: float = 1.5,
        max_concentration: float = 0.4,  # Max 40% in any sector/position
        max_drawdown_trigger: float = 0.15,  # Reduce risk after 15% drawdown
    ):
        self.max_portfolio_beta = max_portfolio_beta
        self.max_concentration = max_concentration
        self.max_drawdown_trigger = max_drawdown_trigger
    
    def check_constraints(
        self,
        signals: List[TradeSignal],
        current_positions: dict,
        portfolio_value: float,
        portfolio_beta: float,
        current_drawdown: float
    ) -> List[TradeSignal]:
        """
        Adjust signals based on portfolio-level risk constraints.
        
        Returns:
            Adjusted list of signals respecting constraints
        """
        adjusted_signals = []
        
        for signal in signals:
            # Check if we're in drawdown mode
            if current_drawdown > self.max_drawdown_trigger:
                if signal.signal == Signal.BUY:
                    # Reduce position sizes during drawdown
                    signal = TradeSignal(
                        ticker=signal.ticker,
                        signal=signal.signal,
                        target_weight=signal.target_weight * 0.5,
                        position_size=signal.position_size * 0.5,
                        shares=signal.shares // 2,
                        confidence=signal.confidence,
                        expected_return=signal.expected_return,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        rationale=signal.rationale + " (reduced due to drawdown)"
                    )
            
            adjusted_signals.append(signal)
        
        return adjusted_signals


if __name__ == "__main__":
    # Test the strategy
    from forecaster import ForecastResult
    import pandas as pd
    
    # Create a mock forecast
    forecast = ForecastResult(
        ticker='AAPL',
        forecast_dates=pd.date_range('2024-01-01', periods=5, freq='B'),
        point_forecast=np.array([150, 152, 154, 153, 155]),
        lower_bound=np.array([145, 146, 147, 146, 148]),
        upper_bound=np.array([155, 158, 161, 160, 162]),
        median_forecast=np.array([150, 152, 154, 153, 155]),
        last_actual_price=148.0,
        last_actual_date=pd.Timestamp('2023-12-29'),
        expected_return=0.047,  # 4.7% expected return
        forecast_volatility=0.02,
        confidence_score=0.72
    )
    
    # Create strategy and generate signal
    strategy = TradingStrategy()
    signal = strategy.generate_signal(
        forecast=forecast,
        portfolio_value=100000,
        current_position=0
    )
    
    print(f"\n=== Trade Signal for {signal.ticker} ===")
    print(f"Signal: {signal.signal.value}")
    print(f"Target Weight: {signal.target_weight:.2%}")
    print(f"Position Size: ${signal.position_size:,.2f}")
    print(f"Shares: {signal.shares}")
    print(f"Expected Return: {signal.expected_return:.2%}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Stop Loss: ${signal.stop_loss:.2f}")
    print(f"Take Profit: ${signal.take_profit:.2f}")
    print(f"Rationale: {signal.rationale}")
