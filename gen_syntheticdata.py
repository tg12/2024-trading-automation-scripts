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
# C0116: Missing function or method docstring
# W0621: Redefining name %r from outer scope (line %s)
# W1203: Use % formatting in logging functions and pass the % parameters as arguments
# C0103: Constant name "%s" doesn't conform to UPPER_CASE naming style
# C0301: Line too long (%s/%s)
# W1201: Specify string format arguments as logging function parameters
# W0511: TODOs
# E1101: Module 'holidays' has no 'US' member (no-member) ... it does, so ignore this
# E0606: possibly-used-before-assignment, ignore this
# UP018: native-literals (UP018)

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_realistic_stock_data(num_years=6):
    """Generate an hourly stock dataset for `num_years` (weekdays only).

    Returns a pandas DataFrame with columns:
    [Gmt time, Open, High, Low, Close, Volume]
    in a simulated 'realistic' pattern without requiring manual parameter tweaking.
    """
    # -----------------------
    # 1. Automatic Parameter Selection
    # -----------------------
    # For annual drift, let it range from -2% to +4% (commodities can be flat or negative)
    annual_drift = np.random.uniform(-0.10, 0.10)

    # For annual volatility, let it range from 30% to 60%
    annual_volatility = np.random.uniform(0.30, 0.60)

    initial_price = np.random.uniform(5000, 7000)

    # -----------------------
    # 2. Create date range (hourly, skipping weekends)
    # -----------------------
    # We'll start from a fixed date, say 1st Jan 2022
    start_date = datetime(2022, 1, 1)
    end_date = start_date + timedelta(days=365 * num_years)

    # Create an hourly range from start_date to end_date, but skip weekends
    current = start_date
    timestamps = []
    while current < end_date:
        # Monday=0, Sunday=6
        if current.weekday() < 5:  # 0-4 => Monday-Friday
            timestamps.append(current)
        current += timedelta(hours=1)

    # Number of data points
    n = len(timestamps)

    # -----------------------
    # 3. Prepare Arrays
    # -----------------------
    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    closes = np.zeros(n)
    volumes = np.zeros(n)

    # The first open is initial_price
    opens[0] = initial_price
    closes[0] = initial_price
    highs[0] = initial_price
    lows[0] = initial_price

    # For hourly steps, delta_t in years is 1 / (365 * 24) approx
    delta_t = 1.0 / (365 * 24)

    # Pre-calculate for GBM
    mu_term = (annual_drift - 0.5 * annual_volatility**2) * delta_t
    sigma_term = annual_volatility * np.sqrt(delta_t)

    # Get random normal draws
    z = np.random.normal(0, 1, n)

    # -----------------------
    # 4. Generate Price Data
    # -----------------------
    for i in range(1, n):
        # The open price = previous close
        opens[i] = closes[i - 1]

        # GBM for close
        closes[i] = opens[i] * np.exp(mu_term + sigma_term * z[i])

        # High/Low around open-close range
        # Add random fluctuations
        price_range = abs(closes[i] - opens[i])
        # randomize a fluctuation between 0.001 and 0.01 of the price range
        rand_fluct = np.random.uniform(0.0001, 0.001) * np.random.lognormal(mean=0, sigma=0.75)
        high_fluct = np.random.uniform(0, price_range * rand_fluct)
        low_fluct = np.random.uniform(0, price_range * rand_fluct)

        highs[i] = max(opens[i], closes[i]) + high_fluct
        lows[i] = min(opens[i], closes[i]) - low_fluct

    # -----------------------
    # 5. Generate Volume Data
    # -----------------------
    # Very rough intraday pattern:
    # - Higher volume near "market open" (e.g., 9:00 - 10:00)
    # - Lower volume in the middle of the day
    # - Higher near "close" (e.g., 15:00 - 16:00)
    # Even though we're generating 24 hours, let's pretend "9-10" and "15-16" are "peak".

    for i, t in enumerate(timestamps):
        hour_of_day = t.hour
        # Basic pattern: peak hours (9-10, 15-16) get higher base volume
        if 9 <= hour_of_day < 10 or 15 <= hour_of_day < 16:
            base_volume = np.random.uniform(10000, 15000)
        else:
            base_volume = np.random.uniform(3000, 7000)

        # Random multiplier for extra variation
        multiplier = np.random.lognormal(mean=0, sigma=0.75)

        # Final volume
        volumes[i] = base_volume * multiplier

    # Convert volume to something smaller or bigger if you want, but let's keep it as is

    # -----------------------
    # 6. Build DataFrame
    # -----------------------
    df = pd.DataFrame(
        {
            "Gmt time": timestamps,
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        }
    )

    return df


def save_to_csv(df, filename="simulated_stock_data.csv"):
    """Save the DataFrame to CSV in the desired format:
    dd.mm.yyyy HH:MM:SS.000,Open,High,Low,Close,Volume
    """
    df_to_save = df.copy()

    # Format timestamps
    df_to_save["Gmt time"] = df_to_save["Gmt time"].dt.strftime("%d.%m.%Y %H:%M:%S.000")

    # Round numeric columns for readability
    df_to_save["Open"] = df_to_save["Open"].round(4)
    df_to_save["High"] = df_to_save["High"].round(4)
    df_to_save["Low"] = df_to_save["Low"].round(4)
    df_to_save["Close"] = df_to_save["Close"].round(4)
    df_to_save["Volume"] = df_to_save["Volume"].round(4)

    df_to_save.to_csv(filename, index=False)


if __name__ == "__main__":
    # Example usage: generate 1 year of data (weekdays only)
    df_data = generate_realistic_stock_data(num_years=6)
    save_to_csv(df_data, "realistic_stock_data.csv")
    print("Simulation complete. Check 'realistic_stock_data.csv' for the output.")
