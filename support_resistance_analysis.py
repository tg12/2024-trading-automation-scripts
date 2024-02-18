import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
import logging
import matplotlib.pyplot as plt  # Import for plotting

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_cluster(df, col, quantile, samples):
    """
    Perform MeanShift clustering on the given column of DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - col: Column name for clustering.
    - quantile: Quantile for bandwidth estimation.
    - samples: Number of samples for bandwidth estimation.

    Returns:
    - List of cluster pivots.
    """
    data = df[col].values.reshape(-1, 1)
    try:
        bandwidth = estimate_bandwidth(data, quantile=quantile, n_samples=samples)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(data)
    except Exception as ex:
        logger.error('Error occurred during clustering: %s', str(ex))
        return []

    pivots = []
    for k in range(len(np.unique(ms.labels_))):
        members = ms.labels_ == k
        values = data[members, 0]
        if len(values) > 0:
            pivots.append(min(values))
            pivots.append(max(values))

    pivots = sorted(pivots)
    return pivots

def get_sure_OHLC(df, intervals, n=2, quantile=0.05, samples=None, up_thresh=0.02, down_thresh=0.02):
    """
    Get support and resistance levels based on the DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - intervals: List of intervals for which support and resistance levels are calculated.
    - n: Number of clusters.
    - quantile: Quantile for bandwidth estimation.
    - samples: Number of samples for bandwidth estimation.
    - up_thresh: Upward threshold for filtering resistance levels.
    - down_thresh: Downward threshold for filtering support levels.

    Returns:
    - DataFrame with added support and resistance levels.
    - List of support levels.
    - List of resistance levels.
    """
    if samples is None:
        samples = len(df)

    su = get_cluster(df, 'low', quantile, samples)
    logger.info('Support cluster: %s', su)

    re = get_cluster(df, 'high', quantile, samples)
    logger.info('Resistance cluster: %s', re)

    su_gap = [su[0]] if su else []
    for i in range(1, len(su)):
        if su[i] <= (su_gap[-1] * (1 + down_thresh)):
            su_gap.append(su[i])
    su = np.array(su_gap)

    re_gap = [re[0]] if re else []
    for i in range(1, len(re)):
        if re[i] <= (re_gap[-1] * (1 - up_thresh)):
            re_gap.append(re[i])
    re = np.array(re_gap)

    for interval in intervals:
        df[f's1_{interval}'], df[f'r1_{interval}'] = zip(*df.apply(lambda row: (np.max(su[np.where(row['low'] >= su)]), np.min(re[np.where(row['high'] <= re)])), axis=1))

    return df, su, re
    
def main():
    # Read data from CSV into DataFrame
    df = pd.read_csv('backtest_prices.csv')  # Update with your CSV file path
    # Assuming CSV contains 'close' column
    # rename any columns if necessary
    # remove mid_ from column names
    df = df.rename(columns={c: c.replace('mid_', '') for c in df.columns})

    # Calculate support and resistance levels
    df, su, re = get_sure_OHLC(df, intervals=['1', '2', '3'])

    # Plot data with support and resistance lines
    plt.figure(figsize=(10, 6))
    plt.plot(df['close'], label='Close Price')
    for s in su:
        plt.axhline(y=s, color='g', linestyle='--', alpha=0.5, label='Support')
    for r in re:
        plt.axhline(y=r, color='r', linestyle='--', alpha=0.5, label='Resistance')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support and Resistance Levels')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
