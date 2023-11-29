# -*- coding=utf-8 -*-
# @Project  : AMA568-advanced_topics_in_quantitative_finance
# @FilePath : financial_derivatives/pricing/option_pricing_engines
# @File     : similarity_reduction.py
# @Time     : 2023/10/28 23:52
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:

import numpy as np
from scipy.stats import norm


def asian_option_pricing(S0, K, T, r, sigma, N):
    """
    Function to calculate the price of a floating strike geometric Asian option.

    Parameters:
    S0: Initial stock price
    K: Strike price
    T: Time to maturity
    r: Risk-free rate
    sigma: Volatility
    N: Number of time steps

    Returns:
    Call_price: Price of the call option
    Put_price: Price of the put option
    """

    # Time parameters
    dt = T / N

    # Adjusted parameters for geometric Asian option
    adj_sigma = sigma * np.sqrt((2 * N + 1) / (6 * (N + 1)))
    adj_r = 0.5 * (r - 0.5 * sigma ** 2 + adj_sigma ** 2)

    # Calculate d1 and d2 for Black-Scholes formula
    d1 = (np.log(S0 / K) + (adj_r + 0.5 * adj_sigma ** 2) * T) / (adj_sigma * np.sqrt(T))
    d2 = d1 - adj_sigma * np.sqrt(T)

    # Calculate call and put prices using Black-Scholes formula
    Call_price = np.exp(-r * T) * (S0 * np.exp(adj_r * T) * norm.cdf(d1) - K * norm.cdf(d2))
    Put_price = np.exp(-r * T) * (K * norm.cdf(-d2) - S0 * np.exp(adj_r * T) * norm.cdf(-d1))

    return Call_price, Put_price

if __name__ == '__main__':

    S0 = 3563  # initial stock price
    K = 3550  # strike price
    T = 0.8/12  # time to maturity in years
    r = 0.02741  # annual risk-free rate at
    N = 252  # number of time steps
    opttype = 'C'  # Option Type 'C' or 'P'
    sigma=0.18
    # sigma=np.sqrt(2*r)+1e-2
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u


    Call_price, Put_price = asian_option_pricing(S0, K, T, r, sigma, N)

    print(f"Call Price: {Call_price}")
    print(f"Put Price: {Put_price}")
