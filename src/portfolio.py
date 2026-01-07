"""
Portfolio Module
Manages portfolio state, executes mock trades, and tracks performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path

from strategy import TradeSignal, Signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a position in a single security."""
    ticker: str
    shares: int
    avg_cost: float
    current_price: float
    
    @property
    def market_value(self) -> float:
        return self.shares * self.current_price
    
    @property
    def cost_basis(self) -> float:
        return self.shares * self.avg_cost
    
    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis
    
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0
        return self.unrealized_pnl / self.cost_basis


@dataclass
class Trade:
    """Record of a completed trade."""
    timestamp: datetime
    ticker: str
    side: str  # 'BUY' or 'SELL'
    shares: int
    price: float
    value: float
    signal_confidence: float
    rationale: str


@dataclass
class PortfolioSnapshot:
    """Point-in-time snapshot of portfolio state."""
    timestamp: datetime
    total_value: float
    cash: float
    positions_value: float
    positions: Dict[str, float]  # ticker -> value
    daily_return: float
    cumulative_return: float
    beta: float


class Portfolio:
    """
    Mock trading portfolio that tracks positions, executes trades,
    and calculates performance metrics.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,  # 0.1% per trade (10 bps)
        slippage: float = 0.0005,  # 0.05% slippage
    ):
        """
        Initialize the portfolio.
        
        Args:
            initial_capital: Starting cash amount
            transaction_cost: Cost per trade as fraction of trade value
            slippage: Assumed slippage as fraction of price
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        self.snapshots: List[PortfolioSnapshot] = []
        self.daily_returns: List[float] = []
        
        # Track realized P&L
        self.realized_pnl = 0.0
        
        # Market data for beta calculation
        self.market_returns: List[float] = []
        
        logger.info(f"Portfolio initialized with ${initial_capital:,.2f}")
    
    @property
    def positions_value(self) -> float:
        """Total market value of all positions."""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)."""
        return self.cash + self.positions_value
    
    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        return self.realized_pnl + unrealized
    
    @property
    def total_return(self) -> float:
        """Total return since inception."""
        return (self.total_value - self.initial_capital) / self.initial_capital
    
    def execute_trade(
        self,
        signal: TradeSignal,
        current_price: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> Optional[Trade]:
        """
        Execute a trade based on a signal.
        
        Args:
            signal: Trade signal to execute
            current_price: Current market price (uses signal's price if None)
            timestamp: Trade timestamp (uses current time if None)
            
        Returns:
            Trade object if executed, None if skipped
        """
        if signal.signal == Signal.HOLD or signal.shares == 0:
            logger.info(f"No trade for {signal.ticker}: {signal.rationale}")
            return None
        
        timestamp = timestamp or datetime.now()
        price = current_price or signal.stop_loss  # fallback
        
        # Apply slippage
        if signal.signal == Signal.BUY:
            execution_price = price * (1 + self.slippage)
        else:
            execution_price = price * (1 - self.slippage)
        
        shares = abs(signal.shares)
        trade_value = shares * execution_price
        
        # Calculate transaction costs
        costs = trade_value * self.transaction_cost
        
        if signal.signal == Signal.BUY:
            return self._execute_buy(
                ticker=signal.ticker,
                shares=shares,
                price=execution_price,
                costs=costs,
                timestamp=timestamp,
                signal=signal
            )
        else:
            return self._execute_sell(
                ticker=signal.ticker,
                shares=shares,
                price=execution_price,
                costs=costs,
                timestamp=timestamp,
                signal=signal
            )
    
    def _execute_buy(
        self,
        ticker: str,
        shares: int,
        price: float,
        costs: float,
        timestamp: datetime,
        signal: TradeSignal
    ) -> Optional[Trade]:
        """Execute a buy order."""
        total_cost = shares * price + costs
        
        # Check if we have enough cash
        if total_cost > self.cash:
            # Reduce shares to fit budget
            max_shares = int((self.cash - costs) / price)
            if max_shares <= 0:
                logger.warning(f"Insufficient cash for {ticker} buy order")
                return None
            shares = max_shares
            total_cost = shares * price + costs
        
        # Update cash
        self.cash -= total_cost
        
        # Update or create position
        if ticker in self.positions:
            pos = self.positions[ticker]
            total_shares = pos.shares + shares
            avg_cost = (pos.cost_basis + shares * price) / total_shares
            self.positions[ticker] = Position(
                ticker=ticker,
                shares=total_shares,
                avg_cost=avg_cost,
                current_price=price
            )
        else:
            self.positions[ticker] = Position(
                ticker=ticker,
                shares=shares,
                avg_cost=price,
                current_price=price
            )
        
        # Record trade
        trade = Trade(
            timestamp=timestamp,
            ticker=ticker,
            side='BUY',
            shares=shares,
            price=price,
            value=shares * price,
            signal_confidence=signal.confidence,
            rationale=signal.rationale
        )
        self.trade_history.append(trade)
        
        logger.info(f"BUY {shares} {ticker} @ ${price:.2f} (total: ${total_cost:,.2f})")
        return trade
    
    def _execute_sell(
        self,
        ticker: str,
        shares: int,
        price: float,
        costs: float,
        timestamp: datetime,
        signal: TradeSignal
    ) -> Optional[Trade]:
        """Execute a sell order."""
        if ticker not in self.positions:
            logger.warning(f"Cannot sell {ticker}: no position")
            return None
        
        pos = self.positions[ticker]
        shares = min(shares, pos.shares)  # Can't sell more than we have
        
        if shares <= 0:
            return None
        
        # Calculate proceeds
        proceeds = shares * price - costs
        
        # Calculate realized P&L for this trade
        trade_pnl = (price - pos.avg_cost) * shares - costs
        self.realized_pnl += trade_pnl
        
        # Update cash
        self.cash += proceeds
        
        # Update position
        remaining_shares = pos.shares - shares
        if remaining_shares > 0:
            self.positions[ticker] = Position(
                ticker=ticker,
                shares=remaining_shares,
                avg_cost=pos.avg_cost,
                current_price=price
            )
        else:
            del self.positions[ticker]
        
        # Record trade
        trade = Trade(
            timestamp=timestamp,
            ticker=ticker,
            side='SELL',
            shares=shares,
            price=price,
            value=shares * price,
            signal_confidence=signal.confidence,
            rationale=signal.rationale
        )
        self.trade_history.append(trade)
        
        logger.info(f"SELL {shares} {ticker} @ ${price:.2f} (proceeds: ${proceeds:,.2f}, P&L: ${trade_pnl:,.2f})")
        return trade
    
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for all positions."""
        for ticker, price in prices.items():
            if ticker in self.positions:
                pos = self.positions[ticker]
                self.positions[ticker] = Position(
                    ticker=ticker,
                    shares=pos.shares,
                    avg_cost=pos.avg_cost,
                    current_price=price
                )
    
    def calculate_beta(
        self,
        portfolio_returns: Optional[List[float]] = None,
        market_returns: Optional[List[float]] = None,
        window: int = 60
    ) -> float:
        """
        Calculate portfolio beta relative to market.
        
        Beta = Cov(portfolio, market) / Var(market)
        """
        port_returns = portfolio_returns or self.daily_returns
        mkt_returns = market_returns or self.market_returns
        
        if len(port_returns) < window or len(mkt_returns) < window:
            return 1.0  # Default to market beta
        
        port = np.array(port_returns[-window:])
        mkt = np.array(mkt_returns[-window:])
        
        covariance = np.cov(port, mkt)[0, 1]
        market_variance = np.var(mkt)
        
        if market_variance == 0:
            return 1.0
        
        return covariance / market_variance
    
    def take_snapshot(
        self,
        timestamp: Optional[datetime] = None,
        market_return: float = 0
    ) -> PortfolioSnapshot:
        """
        Take a snapshot of current portfolio state.
        
        Args:
            timestamp: Snapshot timestamp
            market_return: Market return for this period (for beta calculation)
        """
        timestamp = timestamp or datetime.now()
        
        # Calculate daily return
        if self.snapshots:
            prev_value = self.snapshots[-1].total_value
            daily_return = (self.total_value - prev_value) / prev_value
        else:
            daily_return = 0
        
        self.daily_returns.append(daily_return)
        self.market_returns.append(market_return)
        
        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            total_value=self.total_value,
            cash=self.cash,
            positions_value=self.positions_value,
            positions={t: p.market_value for t, p in self.positions.items()},
            daily_return=daily_return,
            cumulative_return=self.total_return,
            beta=self.calculate_beta()
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.daily_returns or len(self.daily_returns) < 2:
            return {
                'total_return': self.total_return,
                'total_pnl': self.total_pnl,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'num_trades': len(self.trade_history),
            }
        
        returns = np.array(self.daily_returns)
        
        # Sharpe Ratio (annualized, assuming 252 trading days)
        risk_free_rate = 0.05  # 5% annual
        excess_returns = returns - risk_free_rate / 252
        sharpe = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win Rate (from trades)
        if self.trade_history:
            # Group sells to calculate wins
            sells = [t for t in self.trade_history if t.side == 'SELL']
            # Simplified win rate based on trade value vs position
            wins = sum(1 for t in sells if t.value > 0)
            win_rate = wins / len(sells) if sells else 0
        else:
            win_rate = 0
        
        # Sortino Ratio (downside deviation)
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() if len(negative_returns) > 0 else returns.std()
        sortino = np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0
        
        # Calmar Ratio
        calmar = self.total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': self.total_return,
            'total_pnl': self.total_pnl,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.total_pnl - self.realized_pnl,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            'volatility': returns.std() * np.sqrt(252),
            'beta': self.calculate_beta(),
            'win_rate': win_rate,
            'num_trades': len(self.trade_history),
            'avg_daily_return': returns.mean(),
            'best_day': returns.max(),
            'worst_day': returns.min(),
        }
    
    def get_positions_summary(self) -> pd.DataFrame:
        """Get summary of all current positions."""
        if not self.positions:
            return pd.DataFrame()
        
        data = []
        for ticker, pos in self.positions.items():
            data.append({
                'Ticker': ticker,
                'Shares': pos.shares,
                'Avg Cost': pos.avg_cost,
                'Current Price': pos.current_price,
                'Market Value': pos.market_value,
                'Cost Basis': pos.cost_basis,
                'Unrealized P&L': pos.unrealized_pnl,
                'P&L %': pos.unrealized_pnl_pct * 100,
                'Weight': pos.market_value / self.total_value * 100
            })
        
        return pd.DataFrame(data)
    
    def get_trade_history_df(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trade_history:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'Timestamp': t.timestamp,
                'Ticker': t.ticker,
                'Side': t.side,
                'Shares': t.shares,
                'Price': t.price,
                'Value': t.value,
                'Confidence': t.signal_confidence,
                'Rationale': t.rationale
            }
            for t in self.trade_history
        ])
    
    def save_state(self, filepath: str):
        """Save portfolio state to JSON file."""
        state = {
            'cash': self.cash,
            'initial_capital': self.initial_capital,
            'realized_pnl': self.realized_pnl,
            'positions': {
                t: {
                    'shares': p.shares,
                    'avg_cost': p.avg_cost,
                    'current_price': p.current_price
                }
                for t, p in self.positions.items()
            },
            'daily_returns': self.daily_returns,
            'market_returns': self.market_returns,
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Portfolio state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load portfolio state from JSON file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.cash = state['cash']
        self.initial_capital = state['initial_capital']
        self.realized_pnl = state['realized_pnl']
        self.daily_returns = state['daily_returns']
        self.market_returns = state['market_returns']
        
        self.positions = {}
        for t, p in state['positions'].items():
            self.positions[t] = Position(
                ticker=t,
                shares=p['shares'],
                avg_cost=p['avg_cost'],
                current_price=p['current_price']
            )
        
        logger.info(f"Portfolio state loaded from {filepath}")


if __name__ == "__main__":
    # Test the portfolio
    from strategy import TradeSignal, Signal
    
    portfolio = Portfolio(initial_capital=100000)
    
    # Create a mock buy signal
    buy_signal = TradeSignal(
        ticker='AAPL',
        signal=Signal.BUY,
        target_weight=0.1,
        position_size=10000,
        shares=67,
        confidence=0.75,
        expected_return=0.05,
        stop_loss=145.0,
        take_profit=160.0,
        rationale="Test buy signal"
    )
    
    # Execute buy
    trade = portfolio.execute_trade(buy_signal, current_price=150.0)
    print(f"\nAfter buy:")
    print(f"  Cash: ${portfolio.cash:,.2f}")
    print(f"  Positions: ${portfolio.positions_value:,.2f}")
    print(f"  Total: ${portfolio.total_value:,.2f}")
    
    # Update price
    portfolio.update_prices({'AAPL': 155.0})
    portfolio.take_snapshot()
    
    print(f"\nAfter price update to $155:")
    print(f"  Total Value: ${portfolio.total_value:,.2f}")
    print(f"  Unrealized P&L: ${portfolio.total_pnl:,.2f}")
    
    # Create sell signal
    sell_signal = TradeSignal(
        ticker='AAPL',
        signal=Signal.SELL,
        target_weight=0,
        position_size=-10000,
        shares=-67,
        confidence=0.65,
        expected_return=-0.02,
        stop_loss=0,
        take_profit=0,
        rationale="Test sell signal"
    )
    
    # Execute sell
    trade = portfolio.execute_trade(sell_signal, current_price=155.0)
    
    print(f"\nAfter sell:")
    print(f"  Cash: ${portfolio.cash:,.2f}")
    print(f"  Realized P&L: ${portfolio.realized_pnl:,.2f}")
    print(f"  Total Return: {portfolio.total_return:.2%}")
    
    # Get metrics
    metrics = portfolio.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
