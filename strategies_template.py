"""
CU Quants Educational Trading Strategies Framework
---------------------------
This script demonstrates various trading strategies using financial data from Yahoo Finance.
It's designed as an educational tool to help understand how different trading strategies
work and how changing parameters affects their performance.

Strategies included:
1. Moving Average Crossover
2. Mean Reversion
3. Custom Strategy (template for creating your own)

Author: Claude
Date: March 10, 2025
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Set pandas display options for better readability
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

def download_data(ticker, start_date, end_date):
    """
    Download historical stock data using yfinance.
    
    Parameters:
    -----------
    ticker : str
        Stock symbol (e.g., 'AAPL', 'MSFT')
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
        
    Returns:
    --------
    pandas.DataFrame
        Historical stock data with columns: Open, High, Low, Close, Volume, etc.
    """
    # Print info message for educational purposes
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    
    try:
        # Download the data
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Check if data is empty
        if data.empty:
            print(f"No data found for {ticker}.")
            return None
            
        print(f"Successfully downloaded {len(data)} days of data for {ticker}.")
        return data
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def moving_average_strategy(data, short_window=20, long_window=50):
    """
    Implements a moving average crossover strategy.
    
    Strategy Logic:
    - Buy when the short-term MA crosses above the long-term MA (golden cross)
    - Sell when the short-term MA crosses below the long-term MA (death cross)
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Historical price data with a 'Close' column
    short_window : int
        Number of periods for the short-term moving average
    long_window : int
        Number of periods for the long-term moving average
        
    Returns:
    --------
    pandas.DataFrame
        Original data with added columns for signals and moving averages
    """
    # Make a copy of the data to avoid modifying the original
    df = data.copy()
    
    # Create short and long moving averages
    df['Short_MA'] = df['Close'].rolling(window=short_window, min_periods=1).mean()
    df['Long_MA'] = df['Close'].rolling(window=long_window, min_periods=1).mean()
    
    # Create signals: 1 for buy, -1 for sell, 0 for hold
    df['Signal'] = 0
    
    # Golden Cross (short MA crosses above long MA)
    df.loc[(df['Short_MA'] > df['Long_MA']) & (df['Short_MA'].shift(1) <= df['Long_MA'].shift(1)), 'Signal'] = 1
    
    # Death Cross (short MA crosses below long MA)
    df.loc[(df['Short_MA'] < df['Long_MA']) & (df['Short_MA'].shift(1) >= df['Long_MA'].shift(1)), 'Signal'] = -1
    
    # Calculate strategy performance
    df['Position'] = df['Signal'].replace(to_replace=0, method='ffill')
    df['Position'] = df['Position'].fillna(0)
    
    # Calculate strategy returns (simple implementation)
    df['Strategy_Return'] = df['Position'].shift(1) * df['Close'].pct_change()
    df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
    
    return df

def mean_reversion_strategy(data, window=20, std_dev=2):
    """
    Implements a mean reversion strategy using Bollinger Bands.
    
    Strategy Logic:
    - Buy when price falls below lower Bollinger Band (oversold condition)
    - Sell when price rises above upper Bollinger Band (overbought condition)
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Historical price data with a 'Close' column
    window : int
        Number of periods for the moving average and standard deviation calculation
    std_dev : float
        Number of standard deviations for the Bollinger Bands
        
    Returns:
    --------
    pandas.DataFrame
        Original data with added columns for signals and Bollinger Bands
    """
    # Make a copy of the data to avoid modifying the original
    df = data.copy()
    
    # Calculate middle band (moving average)
    df['Middle_Band'] = df['Close'].rolling(window=window, min_periods=1).mean()
    
    # Calculate standard deviation
    df['Std_Dev'] = df['Close'].rolling(window=window, min_periods=1).std()
    
    # Calculate upper and lower bands
    df['Upper_Band'] = df['Middle_Band'] + (df['Std_Dev'] * std_dev)
    df['Lower_Band'] = df['Middle_Band'] - (df['Std_Dev'] * std_dev)
    
    # Create signals: 1 for buy, -1 for sell, 0 for hold
    df['Signal'] = 0
    
    # Buy signal: price crosses below lower band
    df.loc[df['Close'] < df['Lower_Band'], 'Signal'] = 1
    
    # Sell signal: price crosses above upper band
    df.loc[df['Close'] > df['Upper_Band'], 'Signal'] = -1
    
    # We don't want to have multiple consecutive buy or sell signals
    # Only keep the first signal of a sequence of same signals
    df['Signal'] = df['Signal'].mask(df['Signal'] == df['Signal'].shift(1), 0)
    
    # Calculate strategy performance
    df['Position'] = df['Signal'].replace(to_replace=0, method='ffill')
    df['Position'] = df['Position'].fillna(0)
    
    # Calculate strategy returns (simple implementation)
    df['Strategy_Return'] = df['Position'].shift(1) * df['Close'].pct_change()
    df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
    
    return df

def custom_strategy(data, params=None):
    """
    Template for a custom strategy. Users can modify this to implement their own ideas.
    This is just a simple example combining RSI and moving averages.
    
    Strategy Logic (example):
    - Buy when RSI is below 30 (oversold) AND price is above short-term MA
    - Sell when RSI is above 70 (overbought) OR price falls below short-term MA
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Historical price data with a 'Close' column
    params : dict
        Dictionary of parameters for the strategy
        
    Returns:
    --------
    pandas.DataFrame
        Original data with added columns for signals and indicators
    """
    # Default parameters
    if params is None:
        params = {
            'rsi_window': 14,
            'rsi_buy_threshold': 30,
            'rsi_sell_threshold': 70,
            'ma_window': 20
        }
    
    # Make a copy of the data to avoid modifying the original
    df = data.copy()
    
    # Calculate RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=params['rsi_window'], min_periods=1).mean()
    avg_loss = loss.rolling(window=params['rsi_window'], min_periods=1).mean()
    
    rs = avg_gain / avg_loss.where(avg_loss != 0, 1)  # Avoid division by zero
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate moving average
    df['MA'] = df['Close'].rolling(window=params['ma_window'], min_periods=1).mean()
    
    # Create signals based on the strategy logic
    df['Signal'] = 0
    
    # Buy signal: RSI below buy threshold AND price above MA
    buy_condition = (df['RSI'] < params['rsi_buy_threshold']) & (df['Close'] > df['MA'])
    df.loc[buy_condition & (~buy_condition.shift(1, fill_value=False)), 'Signal'] = 1
    
    # Sell signal: RSI above sell threshold OR price below MA
    sell_condition = (df['RSI'] > params['rsi_sell_threshold']) | (df['Close'] < df['MA'])
    df.loc[sell_condition & (~sell_condition.shift(1, fill_value=False)) & (df['Signal'].shift(1) != 1), 'Signal'] = -1
    
    # Calculate strategy performance
    df['Position'] = df['Signal'].replace(to_replace=0, method='ffill')
    df['Position'] = df['Position'].fillna(0)
    
    # Calculate strategy returns (simple implementation)
    df['Strategy_Return'] = df['Position'].shift(1) * df['Close'].pct_change()
    df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
    
    return df

def plot_strategy_results(data, strategy_name):
    """
    Plot the strategy results including:
    - Price and indicators
    - Buy/Sell signals
    - Cumulative returns
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataframe with strategy results
    strategy_name : str
        Name of the strategy for plot titles
    """
    plt.figure(figsize=(14, 12))
    
    # Plot 1: Price and indicators
    plt.subplot(3, 1, 1)
    plt.plot(data.index, data['Close'], label='Close Price')
    
    # Plot different indicators based on strategy
    if 'Short_MA' in data.columns and 'Long_MA' in data.columns:
        plt.plot(data.index, data['Short_MA'], label=f'Short MA ({len(data) // 10})')
        plt.plot(data.index, data['Long_MA'], label=f'Long MA ({len(data) // 5})')
    
    if 'Upper_Band' in data.columns and 'Lower_Band' in data.columns:
        plt.plot(data.index, data['Upper_Band'], 'r--', label='Upper Band')
        plt.plot(data.index, data['Middle_Band'], 'g--', label='Middle Band')
        plt.plot(data.index, data['Lower_Band'], 'r--', label='Lower Band')
    
    if 'MA' in data.columns:
        plt.plot(data.index, data['MA'], label=f'MA ({len(data) // 10})')
    
    plt.title(f'{strategy_name} - Price and Indicators')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Buy/Sell signals
    plt.subplot(3, 1, 2)
    plt.plot(data.index, data['Close'], label='Close Price', alpha=0.5)
    
    # Plot buy signals
    buy_signals = data[data['Signal'] == 1]
    plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='g', label='Buy Signal', s=100)
    
    # Plot sell signals
    sell_signals = data[data['Signal'] == -1]
    plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='r', label='Sell Signal', s=100)
    
    plt.title(f'{strategy_name} - Buy/Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Cumulative returns
    plt.subplot(3, 1, 3)
    
    # Calculate buy and hold returns for comparison
    data['Buy_Hold_Return'] = data['Close'].pct_change()
    data['Buy_Hold_Cumulative'] = (1 + data['Buy_Hold_Return']).cumprod()
    
    plt.plot(data.index, data['Cumulative_Return'], label='Strategy Returns')
    plt.plot(data.index, data['Buy_Hold_Cumulative'], label='Buy & Hold Returns')
    
    plt.title(f'{strategy_name} - Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def calculate_performance_metrics(data):
    """
    Calculate performance metrics for the strategy.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataframe with strategy results
        
    Returns:
    --------
    dict
        Dictionary of performance metrics
    """
    # Filter out NaN values
    returns = data['Strategy_Return'].dropna()
    buy_hold_returns = data['Buy_Hold_Return'].dropna()
    
    # Calculate metrics
    total_return = data['Cumulative_Return'].iloc[-1] - 1 if not data.empty else 0
    buy_hold_return = data['Buy_Hold_Cumulative'].iloc[-1] - 1 if not data.empty else 0
    
    # Annualized return (approximate)
    days = (data.index[-1] - data.index[0]).days
    years = days / 365
    annualized_return = ((1 + total_return) ** (1 / years)) - 1 if years > 0 else 0
    
    # Volatility (annualized standard deviation)
    daily_volatility = returns.std()
    annualized_volatility = daily_volatility * (252 ** 0.5)  # √252 trading days
    
    # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
    sharpe_ratio = (annualized_return / annualized_volatility) if annualized_volatility > 0 else 0
    
    # Maximum drawdown
    cumulative = data['Cumulative_Return']
    max_drawdown = (cumulative / cumulative.cummax() - 1).min()
    
    # Win rate
    trades = data[data['Signal'] != 0]
    if len(trades) > 0:
        win_rate = len(trades[trades['Strategy_Return'] > 0]) / len(trades)
    else:
        win_rate = 0
    
    # Number of trades
    num_trades = len(data[data['Signal'] != 0])
    
    metrics = {
        'Total Return': f"{total_return:.2%}",
        'Buy & Hold Return': f"{buy_hold_return:.2%}",
        'Annualized Return': f"{annualized_return:.2%}",
        'Annualized Volatility': f"{annualized_volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Maximum Drawdown': f"{max_drawdown:.2%}",
        'Win Rate': f"{win_rate:.2%}",
        'Number of Trades': num_trades
    }
    
    return metrics

def display_performance_metrics(metrics):
    """
    Print the performance metrics in a readable format.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of performance metrics
    """
    print("\n" + "="*50)
    print(" "*15 + "PERFORMANCE METRICS")
    print("="*50)
    
    for key, value in metrics.items():
        print(f"{key:.<30} {value}")
    
    print("="*50 + "\n")

def main():
    """
    Main function to run the trading strategy template.
    Allows user to select a strategy and set parameters.
    """
    print("\n" + "="*60)
    print(" "*15 + "TRADING STRATEGIES TEMPLATE")
    print("="*60)
    print("\nThis program allows you to test different trading strategies")
    print("on historical stock data downloaded from Yahoo Finance.")
    
    # Get user inputs
    ticker = input("\nEnter stock ticker symbol (e.g., AAPL, MSFT): ").upper()
    
    # Default to 1 year of data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Allow custom date range
    custom_dates = input("Use default date range (1 year)? (y/n): ").lower()
    if custom_dates == 'n':
        start_date = input("Enter start date (YYYY-MM-DD): ")
        end_date = input("Enter end date (YYYY-MM-DD): ")
    
    # Download the data
    data = download_data(ticker, start_date, end_date)
    
    if data is None:
        print("Failed to download data. Exiting...")
        return
    
    # Strategy selection
    print("\nSelect a trading strategy:")
    print("1. Moving Average Crossover")
    print("2. Mean Reversion (Bollinger Bands)")
    print("3. Custom Strategy (RSI + Moving Average)")
    
    strategy_choice = input("\nEnter your choice (1-3): ")
    
    # Set strategy parameters
    if strategy_choice == '1':
        strategy_name = "Moving Average Crossover"
        
        print("\nSet Moving Average Crossover parameters:")
        short_window = int(input("Short window length (default=20): ") or 20)
        long_window = int(input("Long window length (default=50): ") or 50)
        
        # Validate parameters
        if short_window >= long_window:
            print("Error: Short window must be less than long window.")
            return
        
        # Run the strategy
        results = moving_average_strategy(data, short_window, long_window)
        
    elif strategy_choice == '2':
        strategy_name = "Mean Reversion (Bollinger Bands)"
        
        print("\nSet Mean Reversion parameters:")
        window = int(input("Window length (default=20): ") or 20)
        std_dev = float(input("Standard deviation multiplier (default=2): ") or 2)
        
        # Run the strategy
        results = mean_reversion_strategy(data, window, std_dev)
        
    elif strategy_choice == '3':
        strategy_name = "Custom Strategy (RSI + Moving Average)"
        
        print("\nSet Custom Strategy parameters:")
        rsi_window = int(input("RSI window length (default=14): ") or 14)
        rsi_buy = int(input("RSI buy threshold (default=30): ") or 30)
        rsi_sell = int(input("RSI sell threshold (default=70): ") or 70)
        ma_window = int(input("Moving Average window (default=20): ") or 20)
        
        # Validate parameters
        if rsi_buy >= rsi_sell:
            print("Error: RSI buy threshold must be less than sell threshold.")
            return
        
        # Set the parameters
        params = {
            'rsi_window': rsi_window,
            'rsi_buy_threshold': rsi_buy,
            'rsi_sell_threshold': rsi_sell,
            'ma_window': ma_window
        }
        
        # Run the strategy
        results = custom_strategy(data, params)
        
    else:
        print("Invalid choice. Exiting...")
        return
    
    # Display and plot results
    print(f"\nStrategy: {strategy_name} for {ticker}")
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(results)
    display_performance_metrics(metrics)
    
    # Plot the results
    plot_strategy_results(results, f"{strategy_name} - {ticker}")
    
    # Example of how to save results to a CSV file
    save_results = input("Save results to CSV? (y/n): ").lower()
    if save_results == 'y':
        filename = f"{ticker}_{strategy_name.replace(' ', '_')}_{start_date}_{end_date}.csv"
        results.to_csv(filename)
        print(f"Results saved to {filename}")

    print("\nThank you for using the Trading Strategies Template!")

# Execute the main function if this script is run directly
if __name__ == "__main__":
    main()

# EDUCATIONAL NOTES
"""
Educational Notes on Trading Strategies
--------------------------------------

1. Moving Average Crossover Strategy:
   - One of the most popular trend-following strategies
   - Buy signal: short-term MA crosses above long-term MA (golden cross)
   - Sell signal: short-term MA crosses below long-term MA (death cross)
   - Works well in trending markets, performs poorly in sideways markets
   - Common parameter values: (10, 50), (20, 50), (50, 200) days

2. Mean Reversion Strategy:
   - Based on the concept that prices tend to return to their average
   - Buy signal: price falls below lower Bollinger Band (oversold)
   - Sell signal: price rises above upper Bollinger Band (overbought)
   - Works well in range-bound markets, performs poorly in trending markets
   - Typical parameter values: 20-day window with 2 standard deviations

3. Custom Strategy (RSI + Moving Average):
   - RSI (Relative Strength Index) measures momentum
   - Values below 30 typically indicate oversold conditions
   - Values above 70 typically indicate overbought conditions
   - Combining with moving averages can help filter out false signals
   - RSI tends to work better in range-bound markets

4. Risk Management Considerations:
   - These strategies don't include proper risk management
   - In practice, you should implement stop-loss orders and position sizing
   - Consider adding a maximum loss per trade (e.g., 2% of portfolio)
   - Diversify across multiple instruments to reduce risk

5. Backtesting Limitations:
   - Past performance doesn't guarantee future results
   - Watch out for look-ahead bias and overfitting
   - Transaction costs, slippage, and liquidity are not considered
   - Data quality issues can affect backtest results

6. Improvement Ideas:
   - Add position sizing based on volatility
   - Implement trailing stop-loss mechanisms
   - Consider market regime filters (trending vs. range-bound)
   - Combine multiple signals for more robust strategies
   - Use machine learning for parameter optimization

Remember that successful trading requires discipline, patience, and continuous learning.
No strategy works in all market conditions, so it's important to understand when to
apply each strategy and when to stay out of the market.
"""