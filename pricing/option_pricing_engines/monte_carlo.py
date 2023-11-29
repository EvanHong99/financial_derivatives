# -*- coding=utf-8 -*-
# @Project  : AMA568-advanced_topics_in_quantitative_finance
# @FilePath : financial_derivatives/pricing/option_pricing_engines
# @File     : monte_carlo.py
# @Time     : 2023/10/28 14:42
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:
import logging
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from pricing.path_simulation.brownian_motion import GeometricBrownianMotionSimulator,BrownianMotionSimulator
from pricing.path_simulation.stochastic_volatility import HestonSimulator
from derivatives.options.asian import AsianOption
from financial_derivatives.pricing.option_pricing_engines.base_engine import BaseBarrierPricingEngine,BaseAsianPricingEngine
from scipy.stats import gmean

class BarrierMonteCarloPricingEngine(BaseBarrierPricingEngine):
    def __init__(self, S0, K, T, r, sigma, n, uo: Optional[bool] = None, do: Optional[bool] = None,
                 ub: Optional[float] = None, lb: Optional[float] = None):
        super().__init__(S0, K, T, r, sigma, n, uo, do, ub, lb)

    def calc_price(self,simulator:Union[HestonSimulator]=None):
        S, v=simulator.simulate()




class AsianMonteCarloPricingEngine(BaseAsianPricingEngine):

    def __init__(self, option=AsianOption()):
        super().__init__(S0=option.S0, K=option.K, T=option.T, r=option.r, sigma=option.sigma, fixed_strike=option.fixed_strike,opttype=option.opttype)

    def calc_price(self, m=10000,n=10000,simulator=None,method='arithmetic', *args, **kwargs):
        """

        Parameters
        ----------
        m :
        n :
        method : `str`
            geometric/arithmetic
        args :
        kwargs :

        Returns
        -------

        """
        def mean_func(mat:np.ndarray):
            """

            Parameters
            ----------
            mat :
                m $\times$ n

            Returns
            -------

            """
            if method == 'arithmetic':
                return np.mean(mat,axis=1)
            elif method == 'geometric':
                return gmean(mat,axis=1)


        price_paths=simulator.gen_matrix(T=self.T, M=m, N=n, S0=self.S0, r=self.r, sigma=self.sigma)


        terminal_payoff=0
        if self.fixed_strike:
            terminal_payoff=mean_func(price_paths)-self.K if self.opttype=='C' else self.K-mean_func(price_paths)
            terminal_payoff=np.where(terminal_payoff<0,0,terminal_payoff)
        elif not self.fixed_strike:
            terminal_payoff = price_paths[:,-1]-mean_func(price_paths) if self.opttype == 'C' else mean_func(
                price_paths) - price_paths[:,-1]
            terminal_payoff = np.where(terminal_payoff < 0, 0, terminal_payoff)

        option_price=np.exp(-self.r*self.T)*np.mean(terminal_payoff)
        return option_price





if __name__ == '__main__':
    # a=np.cumsum(np.ones(shape=(3,4)),axis=1)
    # print(a)
    # print(np.mean(a,axis=0))
    # print(gmean(a,axis=0))

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

    contract=AsianOption(S0=S0, K=K, T=T, sigma=sigma, r=r, dividend=0, opttype=opttype, fixed_strike=True)
    engine=AsianMonteCarloPricingEngine(contract)
    price=engine.calc_price(m=1000,n=N,method='arithmetic',simulator=GeometricBrownianMotionSimulator())
    print(price)
