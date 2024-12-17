"""Copyright (C) 2024 James Sawyer
All rights reserved.

This script and the associated files are private
and confidential property. Unauthorized copying of
this file, via any medium, and the divulgence of any
contained information without express written consent
is strictly prohibited.

This script is intended for personal use only and should
not be distributed or used in any commercial or public
setting unless otherwise authorized by the copyright holder.
By using this script, you agree to abide by these terms.

DISCLAIMER: This script is provided 'as is' without warranty
of any kind, either express or implied, including, but not
limited to, the implied warranties of merchantability,
fitness for a particular purpose, or non-infringement. In no
event shall the authors or copyright holders be liable for
any claim, damages, or other liability, whether in an action
of contract, tort or otherwise, arising from, out of, or in
connection with the script or the use or other dealings in
the script.
"""

# -*- coding: utf-8 -*-
# pylint: disable=C0116, W0621, W1203, C0103, C0301, W1201, W0511, E0401, E1101, E0606

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Example Sutte calculation function (as provided)
def calculate_sutte(data):
    """
    Calculates the Sutte Indicator components:
    - SUTTE%L: Predictive lower boundary.
    - SUTTE%H: Predictive upper boundary.
    - SUTTE-PRED: Average prediction based on SUTTE%L and SUTTE%H.

    Args:
        data (pd.DataFrame): DataFrame with columns 'close', 'low', and 'high'.

    Returns:
        pd.DataFrame: Updated DataFrame with Sutte Indicator components.
    """
    # Calculate the previous day's close
    data["prev_close"] = data["close"].shift(1)

    # SUTTE%L: Predictive lower boundary
    data["sutte%l"] = ((data["close"] + data["prev_close"]) / 2) + (
        data["low"] - data["close"]
    )

    # SUTTE%H: Predictive upper boundary
    data["sutte%h"] = ((data["close"] + data["prev_close"]) / 2) + (
        data["high"] - data["close"]
    )

    # SUTTE-PRED: Average prediction of SUTTE%L and SUTTE%H
    data["sutte-pred"] = (data["sutte%l"] + data["sutte%h"]) / 2

    return data


# ---------------------------
# Create Sample Data
# ---------------------------
np.random.seed(42)  # For reproducible results
dates = pd.date_range(start="2021-01-01", periods=100, freq="D")
close_prices = np.cumsum(np.random.randn(100)) + 100  # Simulated closing prices
high_prices = close_prices + np.random.uniform(
    0.5, 1.5, size=100
)  # High prices a bit above close
low_prices = close_prices - np.random.uniform(
    0.5, 1.5, size=100
)  # Low prices a bit below close

data = pd.DataFrame(
    {"close": close_prices, "high": high_prices, "low": low_prices}, index=dates
)

# Calculate the Sutte Indicators
data = calculate_sutte(data)

# ---------------------------
# Plot the Results
# ---------------------------
plt.figure(figsize=(12, 6))
plt.plot(data.index, data["close"], label="Close Price", color="blue")
plt.plot(data.index, data["sutte%l"], label="SUTTE%L", color="green", linestyle="--")
plt.plot(data.index, data["sutte%h"], label="SUTTE%H", color="red", linestyle="--")
plt.plot(
    data.index, data["sutte-pred"], label="SUTTE-PRED", color="purple", linestyle=":"
)

plt.title("Sutte Indicator Example")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
