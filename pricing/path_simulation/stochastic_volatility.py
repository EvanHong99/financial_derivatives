# -*- coding=utf-8 -*-
# @Project  : AMA568-advanced_topics_in_quantitative_finance
# @FilePath : financial_derivatives/pricing/path_simulation
# @File     : stochastic_volatility.py
# @Time     : 2023/11/27 22:33
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from py_vollib_vectorized import vectorized_implied_volatility as implied_vol
from financial_derivatives.pricing.path_simulation.base_simulator import BaseSimulator


class HestonSimulator(BaseSimulator):

    def __init__(self, S0, v0, rho, kappa, theta, sigma, r, T, N, M, *args, **kwargs):
        """

        Parameters
        ----------
        rho   : correlation between asset returns and variance
        kappa : rate of mean reversion in variance process
        theta : long-term mean of variance process
        sigma : vol of vol / volatility of variance process
        r     : risk-free rate
         """
        super().__init__(T, M, N, S0, *args, **kwargs)
        self.v0 = v0
        self.rho = rho
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.r = r

    def gen_matrix(self, T: float, M, N):
        """
        Inputs:
         - S0, v0: initial parameters for asset and variance
         - T     : time of simulation
         - N     : number of time steps
         - M     : number of scenarios / simulations

        Outputs:
        - asset prices over time (numpy array) with shape ()
        - variance over time (numpy array)
        """
        # initialise other parameters
        dt = T / N
        mu = np.array([0, 0])
        cov = np.array([[1, self.rho],
                        [self.rho, 1]])

        # arrays for storing prices and variances
        S = np.full(shape=(N + 1, M), fill_value=self.S0)
        v = np.full(shape=(N + 1, M), fill_value=self.v0)

        # sampling correlated brownian motions under risk-neutral measure
        Z = np.random.multivariate_normal(mu, cov, (N, M))  # N*M*2

        for i in range(1, N + 1):
            S[i] = S[i - 1] * np.exp((self.r - 0.5 * v[i - 1]) * dt + np.sqrt(v[i - 1] * dt) * Z[i - 1, :, 0])
            v[i] = np.maximum(
                v[i - 1] + self.kappa * (self.theta - v[i - 1]) * dt + self.sigma * np.sqrt(v[i - 1] * dt) * Z[i - 1, :,
                                                                                                             1],
                0)

        return S, v

    def gen_series(self, T: float, N):
        return self.gen_matrix(T=T, M=1, N=N)

    def simulate(self) -> np.ndarray:
        if self.M == 1:
            return self.gen_series(T=self.T, N=self.N)
        elif self.M > 1:
            return self.gen_matrix(T=self.T,N=self.N,M=self.M)


class GPTHestonSimulator(BaseSimulator):
    def __init__(self, S0, v0, rho, kappa, theta, sigma, r, T, N, M, *args, **kwargs):
        """

        Parameters
        ----------
        rho   : correlation between asset returns and variance
        kappa : rate of mean reversion in variance process
        theta : long-term mean of variance process
        sigma : vol of vol / volatility of variance process
        r     : risk-free rate
         """
        super().__init__(T, M, N, S0, *args, **kwargs)
        self.v0 = v0
        self.rho = rho
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.r = r


    def simulate(self) -> np.ndarray:
        # Generate random Brownian Motion
        Z1 = np.random.standard_normal(self.N)
        Z2 = self.rho * Z1 + np.sqrt(1 - self.rho ** 2) * np.random.standard_normal(self.N)

        # Generate paths
        V = np.zeros(self.N)
        V[0] = self.v0
        S = np.zeros(self.N)
        S[0] = self.S0

        for t in range(1, self.N):
            V[t] = V[t - 1] + self.kappa * (self.theta - np.maximum(0, V[t - 1])) - self.sigma * np.sqrt(np.maximum(0, V[t - 1])) * Z1[
                t - 1]
            S[t] = S[t - 1] * np.exp((self.r - 0.5 * V[t - 1]) + np.sqrt(V[t - 1]) * Z2[t - 1])

        # Calculate the payoff
        payoff = np.maximum(S[-1] - self.K, 0)

        # Discount the payoff back to today
        option_price = np.exp(-self.r * self.T) * np.mean(payoff)

        print('The price of the option under the Heston model is:', option_price)
        return option_price


if __name__ == '__main__':
    # Parameters
    # simulation dependent
    S0 = 100.0  # asset price
    T = 1.0  # time in years
    r = 0.02  # risk-free rate
    N = 252  # number of time steps in simulation
    M = 10  # number of simulations

    # Heston dependent parameters
    kappa = 3  # rate of mean reversion of variance under risk-neutral dynamics
    theta = 0.20 ** 2  # long-term mean of variance under risk-neutral dynamics
    v0 = 0.25 ** 2  # initial variance under risk-neutral dynamics
    rho = 0.7  # correlation between returns and variances under risk-neutral dynamics
    sigma = 0.6  # volatility of volatility

    simulator=HestonSimulator(S0, v0, rho, kappa, theta, sigma, r, T, N, M)
    S, v = simulator.simulate()
