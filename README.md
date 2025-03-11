# Educational-Trading-Strategies-Framework
A Python tool for learning and backtesting trading strategies using Yahoo Finance data. Implements Moving Average, Mean Reversion, and custom RSI strategies with interactive parameters, visualizations, and performance metrics to help beginners understand algorithmic trading principles.

## Features
- **Data Acquisition**: Download historical stock data using yfinance
- **Multiple Strategies**:
- - Moving Average Crossover
- - Mean Reversion (Bollinger Bands)
- - Custom Strategy (RSI + Moving Average)
- **Interactive Parameters**: Customize strategy parameters
- **Performance Analysis**: Calculate key metrics like Sharpe ratio, drawdown, win rate
- **Visualization**: Plot price movements, indicators, signals, and returns
- **Educational Content**: Detailed comments explaining strategy logic
- **Export Functionality**: Save results to CSV files

## Requirements
```
pip install pandas numpy matplotlib yfinance
```

## Usage

1. Run the script:

```
bashCopypython trading_strategies.py
```

2. Follow the prompts to:
- Enter a ticker symbol (e.g., AAPL, MSFT)
- Select date range
- Choose a strategy
- Set strategy parameters



## Example
```
Enter stock ticker symbol (e.g., AAPL, MSFT): AAPL
Use default date range (1 year)? (y/n): y

Select a trading strategy:
1. Moving Average Crossover
2. Mean Reversion (Bollinger Bands)
3. Custom Strategy (RSI + Moving Average)

Enter your choice (1-3): 1

Set Moving Average Crossover parameters:
Short window length (default=20): 
Long window length (default=50):
```

## Strategies Explained
### Moving Average Crossover
Uses two moving averages of different periods to generate signals. A buy signal occurs when the short-term MA crosses above the long-term MA (golden cross). A sell signal occurs when the short-term MA crosses below the long-term MA (death cross).

### Mean Reversion (Bollinger Bands)
Based on the concept that prices tend to revert to their mean. Uses Bollinger Bands (mean ± standard deviation) to identify overbought/oversold conditions. Buy when price touches lower band, sell when price touches upper band.

### Custom Strategy (RSI + Moving Average)
Combines Relative Strength Index (RSI) with moving averages. Buy when RSI is below a threshold (oversold) AND price is above MA. Sell when RSI is above a threshold (overbought) OR price falls below MA.

### Limitations
- Does not include transaction costs or slippage
- No risk management features included
- Past performance is not indicative of future results
- For educational purposes only, not financial advice

## Future Improvements
- Add position sizing based on volatility
- Implement stop-loss mechanisms
- Add more technical indicators
- Incorporate fundamental data analysis
- Optimize parameters using machine learning

## Disclaimer
This tool is for educational purposes only. It is not financial advice, and should not be used for actual trading without proper risk management and further development.
