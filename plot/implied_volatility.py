# -*- coding=utf-8 -*-
# @Project  : AMA568-advanced_topics_in_quantitative_finance
# @FilePath : financial_derivatives/plot
# @File     : implied_volatility.py
# @Time     : 2023/10/28 23:54
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skewnorm


def implied_volatility_curve(S0,r,T,strike_prices, scenario):



    # Calculate implied volatilities
    implied_vols = []
    for K in strike_prices:
        if scenario == 1:
            d1 = (np.log(S0 / K) + (r + 0.5 * norm.var()) * T) / (norm.std() * np.sqrt(T))
        elif scenario == 2:
            d1 = (np.log(S0 / K) + (r + 0.5 * skewnorm.var(a=-4)) * T) / (skewnorm.std(a=-4) * np.sqrt(T))
        implied_vol = norm.pdf(d1) / (S0 * np.sqrt(T))
        implied_vols.append(implied_vol)

    return implied_vols

if __name__ == '__main__':

    # Define parameters
    S0 = 3563  # Initial stock price
    T = 0.8/12  # time to maturity in years
    r = 0.02741  # annual risk-free rate at

    # Define strike prices
    strike_prices = np.linspace(3000, 4100, 100)

    # Calculate implied volatilities for both scenarios
    implied_vols_1 = implied_volatility_curve(S0=S0,r=r,T=T,strike_prices=strike_prices, scenario=1)
    implied_vols_2 = implied_volatility_curve(S0=S0,r=r,T=T,strike_prices=strike_prices, scenario=2)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(strike_prices, implied_vols_1, label='Scenario 1')
    plt.plot(strike_prices, implied_vols_2, label='Scenario 2')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.title('Implied Volatility Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('/Users/hongyifan/Desktop/study/polyu/AMA568-advanced_topics_in_quantitative_finance/assignment2/Q6.png',dpi=320)
