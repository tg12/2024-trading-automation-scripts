import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta


def generate_stock_data(days=100):
    """Generate synthetic stock data"""
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range(start="2022-01-01", periods=days, freq="D")
    prices = np.random.normal(loc=0.0, scale=0.5, size=days).cumsum() + 100
    return pd.DataFrame({"Date": dates, "Close": prices})

def calculate_moving_averages(df):
    """Calculate various moving averages and add them to the DataFrame"""
    # ma_types = ["dema", "ema", "fwma", "hma", "linreg", "midpoint", "pwma", "rma", 
    #             "sinwma", "sma", "swma", "t3", "tema", "trima", "vidya", "wma", "zlma"]
    
    ma_types = ["jma", "hma"]
                
                
    for ma in ma_types:
        df[f"{ma.upper()}"] = getattr(df.ta, ma)(close='Close', length=10)
        
    return df

def plot_moving_averages(df):
    """Plot the stock data and its moving averages"""
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['Close'], label='Close Price', linewidth=2)

    # Plot each moving average
    for col in df.columns:
        if col not in ['Date', 'Close']:
            plt.plot(df['Date'], df[col], label=col, alpha=0.7)

    plt.title('Stock Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Generate stock data
stock_data = generate_stock_data(days=200)

# Calculate moving averages
stock_data = calculate_moving_averages(stock_data.set_index('Date'))

# Plot moving averages and stock data
plot_moving_averages(stock_data.reset_index())
