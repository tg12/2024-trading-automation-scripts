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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt

date_format = "%Y:%m:%d-%H:%M:%S"


def get_stock_data(csv_file):
    stock_data = pd.read_csv(csv_file)
    stock_data["snapshotTime"] = pd.to_datetime(
        stock_data["snapshotTime"], format=date_format
    )
    return stock_data


def estimate_pen(points):
    mad = np.median(np.abs(points - np.median(points)))
    pen = 2 * np.log(len(points)) * mad
    return pen


def estimate_n_bkps(points):
    # Use a heuristic for estimating the number of breakpoints for Dynp
    max_n_bkps = min(len(points) - 2, 30)  # Limit max_n_bkps to prevent IndexError
    return int(min(len(points) / 10, max_n_bkps)), max_n_bkps


def detect_change_points(points, method):
    if method == "Pelt":
        return None  # Pelt does not have predict method
    elif method == "Dynp":
        n_bkps, max_n_bkps = estimate_n_bkps(points)
        algo = getattr(rpt, method)(model="l1", min_size=3, jump=5).fit(points)
        bkps = algo.predict(n_bkps=n_bkps)  # Specify 'n_bkps' for Dynp
    else:
        pen = estimate_pen(points)
        algo = getattr(rpt, method)(model="l2").fit(points)
        bkps = algo.predict(pen=pen)  # Use 'pen' for other methods
    return bkps


def score_method(points, bkps):
    scores = []
    if bkps is not None and len(bkps) > 1:
        for i in range(len(bkps) - 1):
            if bkps[i + 1] >= len(points):
                continue  # Skip if index is out of bounds
            change_magnitude = abs(points[bkps[i + 1]] - points[bkps[i]])
            scores.append(change_magnitude)
    return sum(scores)


def extract_up_down_points(points, bkps):
    up_points, down_points = [], []
    if bkps is not None and len(bkps) > 1:
        for i in range(len(bkps) - 1):
            if points[bkps[i + 1]] > points[bkps[i]]:
                up_points.append((bkps[i], points[bkps[i]]))
            else:
                down_points.append((bkps[i], points[bkps[i]]))
    return up_points, down_points


def plot_change_points(ax, points, bkps, method_name, is_best_method):
    ax.plot(points, color="blue", label="Price")
    ax.set_title(f"Change Point Detection: {method_name} Method", fontsize=10)
    ax.set_xlabel("Time", fontsize=8)
    ax.set_ylabel("Price", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    if bkps is not None and len(bkps) > 1:
        for i in range(len(bkps) - 1):
            if bkps[i + 1] >= len(points):
                continue  # Skip if index is out of bounds
            if points[bkps[i + 1]] > points[bkps[i]]:
                change = "Up"
                color = "green"
            else:
                change = "Down"
                color = "red"
            ax.axvspan(bkps[i], bkps[i + 1], facecolor=color, alpha=0.3)
    ax.legend(loc="upper left", prop={"size": 8})
    ax.grid(True)
    if is_best_method:
        ax.set_facecolor("lightyellow")


def main():
    csv_file = "backtest_prices.csv"
    stock_data = get_stock_data(csv_file)
    points = np.array(stock_data["price"])

    methods = ["Pelt", "Binseg", "Window", "Dynp"]

    n_methods = len(methods)
    n_cols = 2
    n_rows = (n_methods + 1) // n_cols  # Round up to the nearest integer

    best_method = None
    best_score = float("-inf")

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows), squeeze=False)
    for i, method in enumerate(methods):
        bkps = detect_change_points(points, method)
        row, col = i // n_cols, i % n_cols
        is_best_method = False
        if bkps is not None:
            score = score_method(points, bkps)
            if score > best_score:
                best_score = score
                best_method = method
                is_best_method = True
        plot_change_points(axs[row, col], points, bkps, method, is_best_method)

    # Remove empty subplots if any
    for i in range(n_methods, n_rows * n_cols):
        fig.delaxes(axs.flatten()[i])

    up_points, down_points = extract_up_down_points(points, bkps)
    for ax in axs.flatten():
        for p in up_points:
            ax.plot(p[0], p[1], marker="o", markersize=5, color="green")
        for p in down_points:
            ax.plot(p[0], p[1], marker="o", markersize=5, color="red")

    plt.tight_layout()
    plt.show()

    print(f"The best method for capturing changes is: {best_method}")


if __name__ == "__main__":
    main()
