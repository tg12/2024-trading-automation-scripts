"""THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND
NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR ANYONE
DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR OTHER LIABILITY,
WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# pylint: disable=C0116, W0621, W1203, C0103, C0301, W1201
# Author : James Sawyer
# Maintainer : James Sawyer
# Version : 1.0
# Copyright : Copyright (c) 2024 James Sawyer

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf  # fix_yahoo_finance is deprecated

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parabolic_sar(stock_data):
    """
    Calculate Parabolic SAR for a given stock data.
    
    Parameters:
    stock_data (DataFrame): DataFrame with 'High', 'Low', and 'Close' columns.

    Returns:
    DataFrame: Input DataFrame with additional columns for 'trend', 'sar', 'real_sar', 'ep', and 'af'.
    """
    # Initial parameters
    initial_af = 0.02
    step_af = 0.02
    end_af = 0.2

    # Adding necessary columns
    for column in ['trend', 'sar', 'real_sar', 'ep', 'af']:
        stock_data[column] = 0.0

    stock_data.at[1, 'trend'] = 1 if stock_data['Close'].iloc[1] > stock_data['Close'].iloc[0] else -1
    stock_data.at[1, 'sar'] = stock_data['High'].iloc[0] if stock_data['trend'].iloc[1] > 0 else stock_data['Low'].iloc[0]
    stock_data.at[1, 'real_sar'] = stock_data['sar'].iloc[1]
    stock_data.at[1, 'ep'] = stock_data['High'].iloc[1] if stock_data['trend'].iloc[1] > 0 else stock_data['Low'].iloc[1]
    stock_data.at[1, 'af'] = initial_af

    # SAR calculation loop
    for i in range(2, len(stock_data)):
        temp = stock_data.at[i - 1, 'sar'] + stock_data.at[i - 1, 'af'] * (stock_data.at[i - 1, 'ep'] - stock_data.at[i - 1, 'sar'])
        if stock_data.at[i - 1, 'trend'] < 0:
            stock_data.at[i, 'sar'] = max(temp, stock_data['High'].iloc[i - 1], stock_data['High'].iloc[i - 2])
            temp_trend = 1 if stock_data.at[i, 'sar'] < stock_data['High'].iloc[i] else stock_data.at[i - 1, 'trend'] - 1
        else:
            stock_data.at[i, 'sar'] = min(temp, stock_data['Low'].iloc[i - 1], stock_data['Low'].iloc[i - 2])
            temp_trend = -1 if stock_data.at[i, 'sar'] > stock_data['Low'].iloc[i] else stock_data.at[i - 1, 'trend'] + 1
        stock_data.at[i, 'trend'] = temp_trend

        if stock_data.at[i, 'trend'] < 0:
            temp_ep = min(stock_data['Low'].iloc[i], stock_data.at[i - 1, 'ep']) if stock_data.at[i, 'trend'] != -1 else stock_data['Low'].iloc[i]
        else:
            temp_ep = max(stock_data['High'].iloc[i], stock_data.at[i - 1, 'ep']) if stock_data.at[i, 'trend'] != 1 else stock_data['High'].iloc[i]
        stock_data.at[i, 'ep'] = temp_ep

        if abs(stock_data.at[i, 'trend']) == 1:
            stock_data.at[i, 'af'] = initial_af
        else:
            stock_data.at[i, 'af'] = min(end_af, stock_data.at[i - 1, 'af'] + step_af)

        stock_data.at[i, 'real_sar'] = temp

    return stock_data

def signal_generation(df, method):
    """
    Generate trading signals based on the Parabolic SAR method.

    Parameters:
    df (DataFrame): DataFrame with stock data.
    method (function): Function to calculate the Parabolic SAR.

    Returns:
    DataFrame: DataFrame with additional columns for 'positions' and 'signals'.
    """
    processed_df = method(df)

    processed_df['positions'] = np.where(processed_df['real_sar'] < processed_df['Close'], 1, 0)
    processed_df['signals'] = processed_df['positions'].diff()

    return processed_df

def plot_signals(processed_data, ticker):
    """
    Plot stock closing prices, Parabolic SAR, and trade signals.

    Parameters:
    processed_data (DataFrame): DataFrame with 'Close', 'real_sar', and 'signals'.
    ticker (str): Stock ticker symbol.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(processed_data['Close'], label=f'{ticker} Close Price', lw=2)
    plt.plot(processed_data['real_sar'], linestyle=':', label='Parabolic SAR', color='k')
    plt.scatter(processed_data[processed_data['signals'] == 1].index, processed_data['Close'][processed_data['signals'] == 1], label='LONG', marker='^', color='g', s=100)
    plt.scatter(processed_data[processed_data['signals'] == -1].index, processed_data['Close'][processed_data['signals'] == -1], label='SHORT', marker='v', color='r', s=100)
    
    plt.title('Parabolic SAR Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def main():
    """
    Main function to execute the script logic.
    """
    ticker = 'AAPL'
    start_date = '2019-01-01'
    end_date = '2021-01-01'

    logging.info(f'Downloading {ticker} stock data from {start_date} to {end_date}')
    df = yf.download(ticker, start=start_date, end=end_date)

    # Preprocessing data
    df.reset_index(inplace=True)
    df.drop(['Adj Close', 'Volume'], axis=1, inplace=True)

    logging.info('Generating trading signals...')
    signals_data = signal_generation(df, parabolic_sar)

    # Setting index as date for plotting
    signals_data.set_index('Date', inplace=True)

    logging.info('Plotting trading signals...')
    plot_signals(signals_data.iloc[-450:], ticker)

if __name__ == '__main__':
    main()
