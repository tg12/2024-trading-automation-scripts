"""THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND
NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR ANYONE
DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR OTHER LIABILITY,
WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# -*- coding: utf-8 -*-
# pylint: disable=C0116, W0621, W1203, C0103, C0301, W1201
# C0116: Missing function or method docstring
# W0621: Redefining name %r from outer scope (line %s)
# W1203: Use % formatting in logging functions and pass the % parameters as arguments
# C0103: Constant name "%s" doesn't conform to UPPER_CASE naming style
# C0301: Line too long (%s/%s)
# W1201: Specify string format arguments as logging function parameters

# Author : James Sawyer
# Maintainer : James Sawyer
# Version : 1.7.9
# Status : Production
# Copyright : Copyright (c) 2024 James Sawyer


import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

DATE_FORMAT = "%Y:%m:%d-%H:%M:%S"

if __name__ == "__main__":
    stock_data = pd.read_csv("backtest_prices.csv")
    stock_data["snapshotTime"] = pd.to_datetime(
        stock_data["snapshotTime"],
        format=DATE_FORMAT,
    )

    # dates = stock_data["snapshotTime"]
    # prices = stock_data["mid_open"]
    # plot_with_breakpoints_and_trend_forecast(prices, dates, model_to_use)

    # rename "mid_close" to "close"
    stock_data.rename(columns={"mid_close": "close"}, inplace=True)
    # rename "mid_high" to "high"
    stock_data.rename(columns={"mid_high": "high"}, inplace=True)
    # rename "mid_low" to "low"
    stock_data.rename(columns={"mid_low": "low"}, inplace=True)
    # rename "mid_open" to "open"
    stock_data.rename(columns={"mid_open": "open"}, inplace=True)

    # make sure all the numbers are rounded to 2 decimal places
    stock_data = stock_data.round(2)

    # Create a new column for the index
    stock_data["Index"] = range(1, len(stock_data) + 1)

    # print with tabulate
    print(tabulate(stock_data, headers="keys", tablefmt="pretty"))

    # calculate the slopes
    stock_data["delta_x"] = stock_data.index
    # Calculate the difference between the close prices
    stock_data["delta_y"] = stock_data["close"].diff().fillna(0)

    # Calculate the angles using arctan2 and degrees
    stock_data["angles"] = np.degrees(
        np.arctan2(stock_data["delta_y"], stock_data["delta_x"]),
    )

    # convert to numpy float64 and round to 2 decimal places
    stock_data["angles"] = stock_data["angles"].astype(np.float64).round(2)

    # plot the close prices and a line with the angles
    fig, ax = plt.subplots()
    ax.plot(stock_data.index, stock_data["angles"], label="Angles")
    # plot the angle values as text labels
    for i, angle in enumerate(stock_data["angles"]):
        ax.text(i, angle, f"{angle:.2f}", ha="right")

    # draw a horizontal line at 45 and -45 degrees
    ax.axhline(y=45, color="r", linestyle="--", label="45 degrees")
    ax.axhline(y=-45, color="g", linestyle="--", label="-45 degrees")

    ax.grid(True)
    ax.set_xlabel("Index")
    ax.set_ylabel("Angles")
    ax.legend()
    plt.show()