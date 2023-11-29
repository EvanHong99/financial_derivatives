# -*- coding=utf-8 -*-
# @File     : barrier.py
# @Time     : 2023/9/24 20:56
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : AMA568-advanced_topics_in_quantitative_finance
# @Description:
from typing import Union

import numpy as np
from financial_derivatives.derivatives.options.base_option import BaseOption
from financial_derivatives.pricing.option_pricing_engines.monte_carlo import BarrierMonteCarloPricingEngine
from financial_derivatives.pricing.path_simulation.stochastic_volatility import HestonSimulator

class BarrierOption(BaseOption):
    def __init__(self, barrier_level, strike_price, expiry_time, spot_price, risk_free_rate, volatility):
        self.barrier_level = barrier_level
        self.strike_price = strike_price
        self.expiry_time = expiry_time
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility

    def calc_delta(self):
        pass

    def calc_gamma(self):
        pass

    def calc_vega(self):
        pass

    def calc_theta(self):
        pass

    def calc_rho(self):
        pass

    def payoff(self, S):
        # This is a simple example for an up-and-out call barrier option
        if max(S) > self.barrier_level:
            return 0
        else:
            return max(S[-1] - self.strike_price, 0)

    def price(self, num_simulations):
        # Monte Carlo simulation
        dt = self.expiry_time / num_simulations
        S = np.zeros(num_simulations)
        S[0] = self.spot_price
        for t in range(1, num_simulations):
            S[t] = S[t-1] * np.exp((self.risk_free_rate - 0.5 * self.volatility**2) * dt
                                   + self.volatility * np.sqrt(dt) * np.random.standard_normal())
        return np.exp(-self.risk_free_rate * self.expiry_time) * self.payoff(S)

    def calc_price(self,engine:Union[BarrierMonteCarloPricingEngine]=None):
        # Parameters
        # simulation dependent
        S0 = 100.0  # asset price
        T = 1.0  # time in years
        r = 0.02  # risk-free rate
        N = 252  # number of time steps in simulation
        M = 10000  # number of simulations

        # Heston dependent parameters
        kappa = 3  # rate of mean reversion of variance under risk-neutral dynamics
        theta = 0.20 ** 2  # long-term mean of variance under risk-neutral dynamics
        v0 = 0.25 ** 2  # initial variance under risk-neutral dynamics
        rho = 0.7  # correlation between returns and variances under risk-neutral dynamics
        sigma = 0.6  # volatility of volatility

        kop=110 # knock out price
        K=105 # strike

        simulator = HestonSimulator(S0, v0, rho, kappa, theta, sigma, r, T, N, M)
        s,v=simulator.simulate()
        S_max=np.max(s,axis=0)
        S_terminal=np.where(S_max>=kop,0,s[-1,:])
        payoff=S_terminal-K
        payoff=np.where(payoff>0,payoff,0)
        expectation=np.mean(payoff)
        return expectation

if __name__ == '__main__':
    barrier=BarrierOption(0,0,0,0,0,0)
    print(barrier.calc_price(engine=None))
