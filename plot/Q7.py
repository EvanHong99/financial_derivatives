# -*- coding=utf-8 -*-
# @Project  : AMA568-advanced_topics_in_quantitative_finance
# @FilePath : financial_derivatives/plot
# @File     : Q7.py
# @Time     : 2023/10/29 09:59
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define parameters
S0 = 10  # Initial stock price
r = 0.02  # Risk-free rate
T = 0.25  # Time to maturity (3 months)
strike_prices = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14])  # Strike prices
implied_vols = np.array([0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22])  # Implied volatilities

# Calculate d2 for each strike price
d2_values = (np.log(S0/strike_prices) + (r - 0.5 * implied_vols**2) * T) / (implied_vols * np.sqrt(T))

# Calculate the implied probabilities using the cumulative distribution function of the standard normal distribution
implied_probs = np.exp(-r * T) * norm.cdf(-d2_values)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(strike_prices, implied_probs)
plt.xlabel('Strike Price')
plt.ylabel('Implied Probability')
plt.title('Implied Probability Distribution')
plt.grid(True)
plt.savefig('/Users/hongyifan/Desktop/study/polyu/AMA568-advanced_topics_in_quantitative_finance/assignment2/Q7.png',dpi=320)
