import React, { useState, useEffect } from 'react'
import { 
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, ReferenceLine, ComposedChart, Bar
} from 'recharts'
import { 
  TrendingUp, TrendingDown, Activity, DollarSign, 
  BarChart3, Target, AlertTriangle, Clock, RefreshCw,
  Cpu, Zap, ArrowUpRight, ArrowDownRight, Minus
} from 'lucide-react'

const API_BASE = '/api'

// Terminal Header Component
function TerminalHeader() {
  const [time, setTime] = useState(new Date())
  
  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  return (
    <header className="border-b border-terminal-border bg-terminal-panel px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Cpu className="w-5 h-5 text-terminal-accent" />
            <span className="font-display text-xl font-bold tracking-tight">
              CHRONOS<span className="text-terminal-accent">_</span>TERMINAL
            </span>
          </div>
          <span className="text-terminal-muted text-xs">v2.0.0 | AWS BEDROCK</span>
        </div>
        <div className="flex items-center gap-6 text-sm">
          <div className="flex items-center gap-2">
            <span className="status-dot active"></span>
            <span className="text-terminal-muted">SYSTEM ONLINE</span>
          </div>
          <div className="text-terminal-accent font-mono">
            {time.toLocaleTimeString('en-US', { hour12: false })}
          </div>
        </div>
      </div>
    </header>
  )
}

// Metric Card Component
function MetricCard({ label, value, change, prefix = '', suffix = '', icon: Icon }) {
  const isPositive = change > 0
  const isNegative = change < 0
  
  return (
    <div className="terminal-panel p-4 glow-box">
      <div className="flex items-start justify-between mb-2">
        <span className="text-terminal-muted text-xs uppercase tracking-wider">{label}</span>
        {Icon && <Icon className="w-4 h-4 text-terminal-accent opacity-50" />}
      </div>
      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-bold font-display">
          {prefix}{typeof value === 'number' ? value.toLocaleString() : value}{suffix}
        </span>
        {change !== undefined && (
          <span className={`text-sm flex items-center gap-1 ${
            isPositive ? 'text-terminal-accent' : isNegative ? 'text-terminal-danger' : 'text-terminal-muted'
          }`}>
            {isPositive ? <ArrowUpRight className="w-3 h-3" /> : 
             isNegative ? <ArrowDownRight className="w-3 h-3" /> : 
             <Minus className="w-3 h-3" />}
            {Math.abs(change).toFixed(2)}%
          </span>
        )}
      </div>
    </div>
  )
}

// Signal Badge Component
function SignalBadge({ signal }) {
  const colors = {
    BUY: 'bg-terminal-accent/20 text-terminal-accent border-terminal-accent',
    SELL: 'bg-terminal-danger/20 text-terminal-danger border-terminal-danger',
    HOLD: 'bg-terminal-warning/20 text-terminal-warning border-terminal-warning'
  }
  
  return (
    <span className={`px-3 py-1 text-xs font-bold uppercase tracking-wider border ${colors[signal] || colors.HOLD}`}>
      {signal}
    </span>
  )
}

// Forecast Panel Component
function ForecastPanel({ onForecastGenerated }) {
  const [ticker, setTicker] = useState('AAPL')
  const [loading, setLoading] = useState(false)
  const [forecast, setForecast] = useState(null)
  const [error, setError] = useState(null)

  const generateForecast = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch(`${API_BASE}/forecast`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker: ticker.toUpperCase(), horizon: 5 })
      })
      
      if (!response.ok) throw new Error('Forecast failed')
      
      const data = await response.json()
      setForecast(data)
      onForecastGenerated?.(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const chartData = forecast ? [
    { name: 'Current', price: forecast.last_price, type: 'actual' },
    ...forecast.forecasts.map((f, i) => ({
      name: `Day ${i + 1}`,
      price: f.median,
      lower: f.lower,
      upper: f.upper,
      type: 'forecast'
    }))
  ] : []

  return (
    <div className="terminal-panel p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-display font-semibold flex items-center gap-2">
          <Zap className="w-5 h-5 text-terminal-accent" />
          CHRONOS-2 FORECAST
        </h2>
        <div className="flex items-center gap-3">
          <input
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            placeholder="TICKER"
            className="w-24 text-center uppercase"
          />
          <button 
            onClick={generateForecast}
            disabled={loading}
            className="btn-terminal flex items-center gap-2"
          >
            {loading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span className="loading-dots">PROCESSING</span>
              </>
            ) : (
              <>
                <Target className="w-4 h-4" />
                FORECAST
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-terminal-danger/10 border border-terminal-danger text-terminal-danger p-4 mb-4 text-sm">
          <AlertTriangle className="w-4 h-4 inline mr-2" />
          {error}
        </div>
      )}

      {forecast && (
        <div className="space-y-6">
          {/* Forecast Stats */}
          <div className="grid grid-cols-4 gap-4">
            <div className="text-center p-3 bg-terminal-bg rounded">
              <div className="text-terminal-muted text-xs mb-1">LAST PRICE</div>
              <div className="text-lg font-bold">${forecast.last_price.toFixed(2)}</div>
            </div>
            <div className="text-center p-3 bg-terminal-bg rounded">
              <div className="text-terminal-muted text-xs mb-1">EXPECTED RETURN</div>
              <div className={`text-lg font-bold ${forecast.expected_return >= 0 ? 'text-terminal-accent' : 'text-terminal-danger'}`}>
                {(forecast.expected_return * 100).toFixed(2)}%
              </div>
            </div>
            <div className="text-center p-3 bg-terminal-bg rounded">
              <div className="text-terminal-muted text-xs mb-1">VOLATILITY</div>
              <div className="text-lg font-bold text-terminal-warning">
                {(forecast.forecast_volatility * 100).toFixed(2)}%
              </div>
            </div>
            <div className="text-center p-3 bg-terminal-bg rounded">
              <div className="text-terminal-muted text-xs mb-1">CONFIDENCE</div>
              <div className="text-lg font-bold">{(forecast.confidence_score * 100).toFixed(0)}%</div>
            </div>
          </div>

          {/* Forecast Chart */}
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2a36" />
                <XAxis dataKey="name" stroke="#64748b" fontSize={10} />
                <YAxis stroke="#64748b" fontSize={10} domain={['auto', 'auto']} />
                <Tooltip 
                  contentStyle={{ 
                    background: '#111920', 
                    border: '1px solid #1e2a36',
                    borderRadius: '4px',
                    fontFamily: 'JetBrains Mono'
                  }}
                  labelStyle={{ color: '#64748b' }}
                />
                <Area
                  type="monotone"
                  dataKey="upper"
                  stroke="none"
                  fill="#00d4aa"
                  fillOpacity={0.1}
                />
                <Area
                  type="monotone"
                  dataKey="lower"
                  stroke="none"
                  fill="#0a0f14"
                  fillOpacity={1}
                />
                <Line
                  type="monotone"
                  dataKey="price"
                  stroke="#00d4aa"
                  strokeWidth={2}
                  dot={{ fill: '#00d4aa', strokeWidth: 0, r: 4 }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {!forecast && !loading && (
        <div className="h-64 flex items-center justify-center text-terminal-muted">
          <div className="text-center">
            <Cpu className="w-12 h-12 mx-auto mb-4 opacity-30" />
            <p>Enter a ticker symbol and click FORECAST</p>
            <p className="text-xs mt-2">Powered by Amazon Chronos-2 (120M params)</p>
          </div>
        </div>
      )}
    </div>
  )
}

// Signal Panel Component
function SignalPanel() {
  const [ticker, setTicker] = useState('AAPL')
  const [loading, setLoading] = useState(false)
  const [signal, setSignal] = useState(null)

  const getSignal = async () => {
    setLoading(true)
    try {
      const response = await fetch(`${API_BASE}/signal/${ticker.toUpperCase()}`)
      if (!response.ok) throw new Error('Failed to get signal')
      const data = await response.json()
      setSignal(data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="terminal-panel p-6">
      <h2 className="text-lg font-display font-semibold mb-4 flex items-center gap-2">
        <Activity className="w-5 h-5 text-terminal-accent" />
        TRADING SIGNAL
      </h2>
      
      <div className="flex items-center gap-3 mb-6">
        <input
          type="text"
          value={ticker}
          onChange={(e) => setTicker(e.target.value.toUpperCase())}
          placeholder="TICKER"
          className="w-24 text-center uppercase"
        />
        <button onClick={getSignal} disabled={loading} className="btn-terminal">
          {loading ? 'ANALYZING...' : 'ANALYZE'}
        </button>
      </div>

      {signal && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-terminal-muted">RECOMMENDATION</span>
            <SignalBadge signal={signal.signal} />
          </div>
          
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-terminal-muted block">Confidence</span>
              <div className="flex items-center gap-2 mt-1">
                <div className="flex-1 h-2 bg-terminal-bg rounded overflow-hidden">
                  <div 
                    className="h-full bg-terminal-accent" 
                    style={{ width: `${signal.confidence * 100}%` }}
                  />
                </div>
                <span>{(signal.confidence * 100).toFixed(0)}%</span>
              </div>
            </div>
            <div>
              <span className="text-terminal-muted block">Expected Return</span>
              <span className={signal.expected_return >= 0 ? 'text-terminal-accent' : 'text-terminal-danger'}>
                {(signal.expected_return * 100).toFixed(2)}%
              </span>
            </div>
            {signal.stop_loss && (
              <div>
                <span className="text-terminal-muted block">Stop Loss</span>
                <span className="text-terminal-danger">${signal.stop_loss.toFixed(2)}</span>
              </div>
            )}
            {signal.take_profit && (
              <div>
                <span className="text-terminal-muted block">Take Profit</span>
                <span className="text-terminal-accent">${signal.take_profit.toFixed(2)}</span>
              </div>
            )}
          </div>

          {signal.reasoning && (
            <div className="text-xs text-terminal-muted bg-terminal-bg p-3 rounded mt-4">
              <span className="text-terminal-accent">REASONING:</span> {signal.reasoning}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Portfolio Panel Component
function PortfolioPanel() {
  const [portfolio, setPortfolio] = useState(null)
  const [loading, setLoading] = useState(false)

  const fetchPortfolio = async () => {
    setLoading(true)
    try {
      const response = await fetch(`${API_BASE}/portfolio`)
      if (!response.ok) throw new Error('Failed to fetch portfolio')
      const data = await response.json()
      setPortfolio(data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchPortfolio()
    const interval = setInterval(fetchPortfolio, 30000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="terminal-panel p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-display font-semibold flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-terminal-accent" />
          PORTFOLIO
        </h2>
        <button onClick={fetchPortfolio} className="text-terminal-muted hover:text-terminal-accent">
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {portfolio && (
        <>
          <div className="grid grid-cols-3 gap-4 mb-6">
            <MetricCard 
              label="Total Value" 
              value={portfolio.total_value.toFixed(2)} 
              prefix="$"
              change={portfolio.return_pct}
              icon={DollarSign}
            />
            <MetricCard 
              label="Cash" 
              value={portfolio.cash.toFixed(2)} 
              prefix="$"
              icon={DollarSign}
            />
            <MetricCard 
              label="P&L" 
              value={portfolio.total_pnl.toFixed(2)} 
              prefix="$"
              change={portfolio.return_pct}
              icon={TrendingUp}
            />
          </div>

          {portfolio.positions.length > 0 ? (
            <table className="terminal-table">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Shares</th>
                  <th>Avg Cost</th>
                  <th>Current</th>
                  <th>P&L</th>
                </tr>
              </thead>
              <tbody>
                {portfolio.positions.map(pos => (
                  <tr key={pos.ticker}>
                    <td className="font-bold">{pos.ticker}</td>
                    <td>{pos.shares.toFixed(2)}</td>
                    <td>${pos.avg_cost.toFixed(2)}</td>
                    <td>${pos.current_price.toFixed(2)}</td>
                    <td className={pos.unrealized_pnl >= 0 ? 'text-terminal-accent' : 'text-terminal-danger'}>
                      ${pos.unrealized_pnl.toFixed(2)} ({pos.unrealized_pnl_pct.toFixed(1)}%)
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="text-center text-terminal-muted py-8">
              No positions. Execute a trade to get started.
            </div>
          )}
        </>
      )}
    </div>
  )
}

// Trade Execution Panel
function TradePanel({ onTradeExecuted }) {
  const [ticker, setTicker] = useState('AAPL')
  const [action, setAction] = useState('BUY')
  const [amount, setAmount] = useState('')
  const [amountType, setAmountType] = useState('dollars')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  const executeTrade = async () => {
    setLoading(true)
    setResult(null)
    
    try {
      const body = {
        ticker: ticker.toUpperCase(),
        action,
        ...(amountType === 'dollars' 
          ? { dollar_amount: parseFloat(amount) }
          : { shares: parseFloat(amount) }
        )
      }
      
      const response = await fetch(`${API_BASE}/trade`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      })
      
      const data = await response.json()
      
      if (!response.ok) {
        setResult({ success: false, message: data.detail })
      } else {
        setResult({ success: true, ...data })
        onTradeExecuted?.()
      }
    } catch (err) {
      setResult({ success: false, message: err.message })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="terminal-panel p-6">
      <h2 className="text-lg font-display font-semibold mb-4 flex items-center gap-2">
        <Zap className="w-5 h-5 text-terminal-accent" />
        EXECUTE TRADE
      </h2>

      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-terminal-muted text-xs block mb-2">TICKER</label>
            <input
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              className="w-full uppercase"
            />
          </div>
          <div>
            <label className="text-terminal-muted text-xs block mb-2">ACTION</label>
            <select 
              value={action} 
              onChange={(e) => setAction(e.target.value)}
              className="w-full"
            >
              <option value="BUY">BUY</option>
              <option value="SELL">SELL</option>
            </select>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-terminal-muted text-xs block mb-2">AMOUNT TYPE</label>
            <select 
              value={amountType} 
              onChange={(e) => setAmountType(e.target.value)}
              className="w-full"
            >
              <option value="dollars">DOLLARS</option>
              <option value="shares">SHARES</option>
            </select>
          </div>
          <div>
            <label className="text-terminal-muted text-xs block mb-2">
              {amountType === 'dollars' ? 'USD AMOUNT' : 'SHARES'}
            </label>
            <input
              type="number"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              placeholder={amountType === 'dollars' ? '1000' : '10'}
              className="w-full"
            />
          </div>
        </div>

        <button 
          onClick={executeTrade}
          disabled={loading || !amount}
          className={`btn-terminal w-full ${action === 'SELL' ? 'border-terminal-danger text-terminal-danger hover:bg-terminal-danger' : ''}`}
        >
          {loading ? 'EXECUTING...' : `${action} ${ticker}`}
        </button>

        {result && (
          <div className={`p-4 text-sm ${
            result.success 
              ? 'bg-terminal-accent/10 border border-terminal-accent text-terminal-accent'
              : 'bg-terminal-danger/10 border border-terminal-danger text-terminal-danger'
          }`}>
            {result.success ? (
              <>
                ✓ {result.action} {result.shares.toFixed(2)} shares of {result.ticker} @ ${result.price.toFixed(2)}
                <br />
                <span className="text-terminal-muted">Portfolio: ${result.portfolio_value.toFixed(2)}</span>
              </>
            ) : (
              <>✗ {result.message}</>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

// Main App Component
export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard')
  
  const tabs = [
    { id: 'dashboard', label: 'DASHBOARD' },
    { id: 'forecast', label: 'FORECAST' },
    { id: 'trade', label: 'TRADE' }
  ]

  return (
    <div className="min-h-screen flex flex-col scanlines">
      <TerminalHeader />
      
      {/* Tab Navigation */}
      <nav className="border-b border-terminal-border bg-terminal-panel px-6">
        <div className="flex gap-1">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-6 py-3 text-sm font-medium transition-colors relative ${
                activeTab === tab.id 
                  ? 'text-terminal-accent' 
                  : 'text-terminal-muted hover:text-terminal-text'
              }`}
            >
              {tab.label}
              {activeTab === tab.id && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-terminal-accent" />
              )}
            </button>
          ))}
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 p-6">
        {activeTab === 'dashboard' && (
          <div className="grid grid-cols-12 gap-6">
            <div className="col-span-8">
              <ForecastPanel />
            </div>
            <div className="col-span-4 space-y-6">
              <SignalPanel />
            </div>
            <div className="col-span-12">
              <PortfolioPanel />
            </div>
          </div>
        )}

        {activeTab === 'forecast' && (
          <div className="max-w-4xl mx-auto">
            <ForecastPanel />
          </div>
        )}

        {activeTab === 'trade' && (
          <div className="grid grid-cols-12 gap-6">
            <div className="col-span-4">
              <TradePanel />
            </div>
            <div className="col-span-8">
              <PortfolioPanel />
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-terminal-border bg-terminal-panel px-6 py-3">
        <div className="flex items-center justify-between text-xs text-terminal-muted">
          <span>CHRONOS TERMINAL | MOCK TRADING SYSTEM</span>
          <span className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <Clock className="w-3 h-3" />
              Data: 1D granularity
            </span>
            <span className="flex items-center gap-1">
              <Cpu className="w-3 h-3" />
              Model: Chronos-2 (120M)
            </span>
          </span>
        </div>
      </footer>
    </div>
  )
}
