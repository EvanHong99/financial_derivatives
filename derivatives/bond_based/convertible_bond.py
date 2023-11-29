# # -*- coding=utf-8 -*-
# # @Project  : AMA568-advanced_topics_in_quantitative_finance
# # @FilePath : financial_derivatives/derivatives/bond_based
# # @File     : convertible_bond.py
# # @Time     : 2023/11/29 09:28
# # @Author   : EvanHong
# # @Email    : 939778128@qq.com
# # @Description: gpt
#

import numpy as np

def binomial_tree_convertible_bond_with_dividend(S0, K, T, r, sigma, N, face_value, coupon_rate, conversion_ratio, dividend_yield):
    """
    Price a convertible bond using the binomial tree model, accounting for stock dividends.

    :param S0: Initial stock price
    :param K: Conversion price
    :param T: Time to maturity in years
    :param r: Risk-free interest rate
    :param sigma: Volatility of the stock
    :param N: Number of steps in the binomial tree
    :param face_value: Face value of the bond
    :param coupon_rate: Coupon rate of the bond
    :param conversion_ratio: Number of shares the bond can be converted into
    :param dividend_yield: Continuous dividend yield of the stock
    :return: Price of the convertible bond
    """
    dt = T / N  # Time step
    u = np.exp((r - dividend_yield) * dt + sigma * np.sqrt(dt))  # Up factor adjusted for dividend yield
    d = 1 / u  # Down factor
    q = (np.exp((r - dividend_yield) * dt) - d) / (u - d)  # Risk-neutral probability adjusted for dividend yield

    # The rest of the code remains the same as the previous implementation

    # Initialize arrays to store stock prices and bond values
    stock_prices = np.zeros((N + 1, N + 1))
    bond_values = np.zeros((N + 1, N + 1))

    # Generate stock price binomial tree
    for i in range(N + 1):
        for j in range(i + 1):
            stock_prices[j, i] = S0 * (u ** (i - j)) * (d ** j)

    # Calculate bond value at maturity
    for i in range(N + 1):
        conversion_value = conversion_ratio * stock_prices[i, N]
        bond_values[i, N] = max(face_value, conversion_value)

    # Backward induction to calculate bond value at each node
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            expected_bond_value = np.exp(-r * dt) * (q * bond_values[j, i + 1] + (1 - q) * bond_values[j + 1, i + 1])
            conversion_value = conversion_ratio * stock_prices[j, i]
            bond_values[j, i] = max(expected_bond_value + coupon_rate * face_value, conversion_value)

    return bond_values[0, 0]


S0 = 4.99  # Initial stock price
K = 5.43  # Conversion price
T = 6  # Time to maturity in years
r = 0.025  # Risk-free interest rate
sigma = 0.223  # Volatility of the stock
N = 1000  # Number of steps in the binomial tree
face_value = 100  # Face value of the bond
coupon_rate = 0.01  # Coupon rate of the bond (1% per year)
conversion_ratio = face_value/K  # Number of shares the bond can be converted into
dividend_yield = 0.00  # Assuming a 0% continuous dividend yield

# # Calculate the price of the convertible bond
# convertible_bond_price = binomial_tree_convertible_bond(S0, K, T, r, sigma, N, face_value, coupon_rate, conversion_ratio)
# print(convertible_bond_price)

# Calculate the price of the convertible bond including dividend yield
convertible_bond_price_with_dividend = binomial_tree_convertible_bond_with_dividend(
    S0, K, T, r, sigma, N, face_value, coupon_rate, conversion_ratio, dividend_yield
)
print(convertible_bond_price_with_dividend)

