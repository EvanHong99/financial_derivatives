# -*- coding=utf-8 -*-
# @File     : binomial_tree.py
# @Time     : 2023/9/24 19:55
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : AMA568-advanced_topics_in_quantitative_finance
# @Description: [1] https://www.codearmo.com/python-tutorial/options-trading-binomial-pricing-model
import logging
from typing import Optional, Union

import numpy as np
from scipy.stats import binom
import math
from base_engine import BasePricingEngine,BaseAsianPricingEngine,BaseBarrierPricingEngine,BaseLookbackPricingEngine
from financial_derivatives.derivatives.options.base_option import BaseOption
from financial_derivatives.derivatives.options.lookback import LookbackOption
from financial_derivatives.pricing.path_simulation.binomial_tree import BinomialTreeSimulator

from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap

def combos(n, i):
    """
    排列组合 C_n^i
    :param n:
    :param i:
    :return:
    """
    return math.factorial(n) / (math.factorial(n-i)*math.factorial(i))


def european_option(S0, K, T, r, sigma, n, type_='call', underlying='stock'):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p = (np.exp(r * dt) - d) / (u - d)
    value = 0
    # forward
    for i in range(n + 1):
        node_prob = combos(n, i) * p ** i * (1 - p) ** (n - i)
        ST = S0 * (u) ** i * (d) ** (n - i)
        if type_ == 'call':
            value += max(ST - K, 0) * node_prob
        elif type_ == 'put':
            value += max(K - ST, 0) * node_prob
        else:
            raise ValueError("type_ must be 'call' or 'put'")

    return value * np.exp(-r * T)

def american_option_slow(S0, K, T, r, sigma, n, q=0, type_='call', underlying='stock'):
    """american_option

    Args:
        S0 (_type_): _description_
        K (_type_): _description_
        T (_type_): _description_
        r (_type_): _description_
        sigma (_type_): _description_
        n (_type_): _description_
        q (int, optional): dividend rate. Defaults to 0.
        type_ (str, optional): _description_. Defaults to 'call'.
        underlying (str, optional): _description_. Defaults to 'stock'.

    Raises:
        ValueError: _description_
    """
    
    assert type_ in ['call','put']

    dt=T/n
    u=np.exp(sigma*np.sqrt(dt))
    d=1/u
    p=(np.exp(r*dt)-d)/(u-d) # risk neutral probability
    disc = np.exp(-r*dt) # discount rate

    # initialize
    prices = np.zeros(n + 1) # prices at T
    for i in range(n+1): # top to bottom
        prices[i] = S0 * u ** (n -i) * d ** i

    # payoff at the last layer
    payoff = np.zeros(n + 1) # prices at T
    for i in range(n+1):
        payoff[i] =  max(prices[i]-K,0) if type_=='call' else max(K-prices[i],0)

    # backwark from the last second layer
    for i in range(n-1,-1,-1): # layer no.
        for j in range(i+1): # i+1 nodes in i-th layer
            St= S0 * u ** (i-j) * d ** j 
            early_exercise=max(St-K,0) if type_=='call' else max(K-St,0)
            hold=disc*(p*payoff[j]+(1-p)*payoff[j+1])
            payoff[j]=max(early_exercise,hold)

    return payoff[0]

#
# def american_slow_tree(S0,K,T,r,sigma,n,opttype='P'):
#     """
#     References
#     ----------
#     [1] https://quantpy.com.au/binomial-tree-model/american-put-options-with-the-binomial-asset-pricing-model/
#     """
#     #precompute values
#     dt = T/n
#     u=np.exp(sigma*np.sqrt(dt))
#     d=1/u
#     q = (np.exp(r*dt) - d)/(u-d)
#     disc = np.exp(-r*dt)
#
#     # initialise stock prices at maturity
#     S = np.zeros(n+1)
#     for j in range(0, n+1):
#         S[j] = S0 * u**j * d**(n-j)
#
#     # option payoff
#     C = np.zeros(n+1)
#     for j in range(0, n+1):
#         if opttype == 'P':
#             C[j] = max(0, K - S[j])
#         else:
#             C[j] = max(0, S[j] - K)
#
#     # backward recursion through the tree
#     for i in np.arange(n-1,-1,-1):
#         for j in range(0,i+1):
#             S = S0 * u**j * d**(i-j)
#             C[j] = disc * ( q*C[j+1] + (1-q)*C[j] )
#             if opttype == 'P':
#                 C[j] = max(C[j], K - S)
#             else:
#                 C[j] = max(C[j], S - K)
#
#     return C[0]
#
# # american_slow_tree(K,T,S0,r,N,u,d,opttype='P')
#
# def american_call(S0, K, T, r, sigma, n, underlying='stock'):
#     """
#
#     :param S0: spot price
#     :param K: strike
#     :param T: maturity
#     :param r: risk free rate
#     :param sigma: volatility
#     :param n: n steps
#     :return:
#     """
#     dt = T / n
#     u = np.exp(sigma * np.sqrt(dt))
#     d = 1 / u
#     p = (np.exp(r * dt) - d) / (u - d)
#     prices = np.zeros(n + 1) # prices at T
#     prices[0] = S0 * d ** n
#     for i in range(1, n + 1):
#         prices[i] = u * u * prices[i - 1] # each two prices at time T have
#     call = np.zeros(n + 1)
#     for i in range(n + 1):
#         call[i] = max(prices[i] - K, 0)
#     for j in range(n):
#         for i in range(n - j):
#             early_exercise = max(prices[i] - K, 0)
#             hold = (p * call[i + 1] + (1 - p) * call[i]) / np.exp(r * dt)
#             call[i] = max(early_exercise, hold)
#     return call[0]
#
# def american_put(S0, K, T, r, sigma, n, underlying='stock'):
#     """
#
#     :param S0: spot price
#     :param K: strike
#     :param T: maturity
#     :param r: risk-free rate
#     :param sigma: volatility
#     :param n: n steps
#     :return:
#     """
#     dt = T / n
#     u = np.exp(sigma * np.sqrt(dt))
#     d = 1 / u
#     p = (np.exp(r * dt) - d) / (u - d)
#     prices = np.zeros(n + 1)
#     prices[0] = S0 * d ** n
#     for i in range(1, n + 1):
#         prices[i] = u * u * prices[i - 1]
#     put = np.zeros(n + 1)
#     for i in range(n + 1):
#         put[i] = max(K - prices[i], 0)
#     for j in range(n):
#         for i in range(n - j):
#             early_exercise = max(K - prices[i], 0)
#             hold = (p * put[i + 1] + (1 - p) * put[i]) / np.exp(r * dt)
#             put[i] = max(early_exercise, hold)
#     return put[0]



class BarrierBinomialTreePricingEngine(BasePricingEngine):
    def calc_price(self,option:BaseOption,s,):
        # if _type=='call':
        #     ...
        # elif _type=='put':
        #     ...
        option.k

    def __calc_price_call__(self,s, k, call_price, t,):
        assert k<=call_price

    def __indicator_func__(self, prices:Union[list,np.ndarray], uo:Optional[bool]=None, do:Optional[bool]=None, ub:Optional[float]=None, lb:Optional[float]=None):
        """指示函数，用于标示是否保留值，还是置0

        Parameters
        ----------
        prices :
        uo :
        do :
        ub :
        lb :

        Returns
        -------

        """
        prices=np.array(prices)
        indicator=np.full_like(prices,fill_value=1) # which nodes can keep the value
        if uo is not None and do is not None: # double barrier
            if uo and do: # up-out and down-out
                indicator[np.logical_or(prices>=ub,prices<=lb)]=0
            elif uo and not do:# up-out and down-in
                raise NotImplementedError()
            elif not uo and do:# up-in and down-out
                raise NotImplementedError()
            else:
                raise NotImplementedError()

        elif uo is not None and do is None: # 单边barrier
            if uo: # up-out
                indicator[prices>=ub]=0
            elif not uo: # up-in
                raise NotImplementedError()

        elif uo is None and do is not None:  # 单边barrier
            if do:  # up-out
                indicator[prices <= lb] = 0
            elif not do:  # up-in
                raise NotImplementedError()

        return indicator


    @timing
    def calc_price_barrier_slow_call(self,S0, K, T, r, sigma, n,uo:Optional[bool]=None,do:Optional[bool]=None, ub:Optional[float]=None,lb:Optional[float]=None):
        """

        Raises
        ------
        NotImplementedError for all the barriers except up-out-call-european

        up-out对于一个call来说很怪，但也不是不行，也就只有在小涨并且不高于upper bound时才能获利
        todo vectorize computation

        Parameters
        ----------
        S0 :
        K :
        T :
        r :
        sigma :
        n :
        ub :
        lb :
        uo : Optional[bool], default is None
            is up-out option? If not, then is up-in
        do : Optional[bool], default is None
            is down-out option? If not, then is down-in

        Returns
        -------

        """
        dt=T/n
        u=np.exp(sigma*np.sqrt(dt))
        d = 1 / u
        p=(np.exp(r*dt)-d)/(u-d)
        disc = np.exp(-r * dt)  # discount rate

        # initialize
        prices = np.zeros(n + 1)  # prices at T
        for i in range(n + 1):  # top to bottom
            prices[i] = S0 * u ** (n - i) * d ** i

        # payoff at the last layer
        payoff = np.zeros(n + 1) # prices at T
        for i in range(n+1):
            payoff[i] =max(prices[i]-K,0)

        #
        indicators=self.__indicator_func__(prices=prices, uo=uo, do=do, ub=ub, lb=lb)
        payoff=payoff*indicators
        if len(indicators)!=np.sum(indicators):
            print("terminal reaches barrier")

        # backward
        for i in range(n-1,-1,-1):
            # prices = S0 * d ** (np.arange(i, -1, -1)) * u ** (np.arange(0, i + 1, 1))
            for j in range(0,i+1): # i+1 nodes in i-th layer
                St=S0*u**(i-j)*d**j # j down movements
                if uo is not None and do is not None:
                    raise NotImplementedError()
                elif uo is not None and uo:
                    if St>=ub:
                        payoff[j]=0
                    else:
                        payoff[j]=disc*(p*payoff[j]+(1-p)*payoff[j+1])
                elif do is not None and do:
                    if St<=lb:
                        payoff[j]=0
                    else:
                        payoff[j]=disc*(p*payoff[j]+(1-p)*payoff[j+1])
                else:
                    raise NotImplementedError()
        return payoff[0]


# @timing
# def barrier_tree_slow(K,T,S0,H,r,N,u,d,opttype='C'):
#     """only for double check
#
#     Parameters
#     ----------
#     K :
#     T :
#     S0 :
#     H :
#     r :
#     N :
#     u :
#     d :
#     opttype :
#
#     Returns
#     -------
#
#     References
#     ----------
#     [1] https://github.com/TheQuantPy/youtube-tutorials/blob/main/2021/003%20Jul-Sep/2021-07-08%20Barrier%20Option%20Pricing%20with%20Binomial%20Trees%20_%20Theory%20_%20Implementation%20in%20Python.ipynb
#
#     """
#     #precompute values
#     dt = T/N
#     q = (np.exp(r*dt) - d)/(u-d)
#     disc = np.exp(-r*dt)
#
#     # initialise asset prices at maturity
#     S = np.zeros(N+1)
#     for j in range(0,N+1):
#         S[j] = S0 * u**j * d**(N-j)
#
#     # option payoff
#     C = np.zeros(N+1)
#     for j in range(0,N+1):
#         if opttype == 'C':
#             C[j] = max(0, S[j] - K)
#         else:
#             C[j] = max(0, K - S[j])
#
#     # check terminal condition payoff
#     for j in range(0, N+1):
#         S = S0 * u**j * d**(N-j)
#         if S >= H:
#             C[j] = 0
#
#     # backward recursion through the tree
#     for i in np.arange(N-1,-1,-1):
#         for j in range(0,i+1):
#             S = S0 * u**j * d**(i-j)
#             if S >= H:
#                 C[j] = 0
#             else:
#                 C[j] = disc * (q*C[j+1]+(1-q)*C[j])
#     return C[0]

class AsianBTPricingEngine(BaseLookbackPricingEngine    ):
    def __init__(self, S0, K, T, r, sigma, fixed_strike=True, opttype='C'):
        super().__init__(S0, K, T, r, sigma, fixed_strike, opttype)

    def calc_price(self, n=1000, *args, **kwargs):
        raise NotImplementedError()
        return 0

class LookbackBTPricingEngine(BaseLookbackPricingEngine):
    def __init__(self, option=LookbackOption()):
        super().__init__(S0=option.S0, K=option.K, T=option.T, r=option.r, sigma=option.sigma)
        self.opttype=option.opttype
        self.fixed_strike=option.fixed_strike

    def calc_price(self, n=1000, *args, **kwargs):
        """

        Parameters
        ----------
        n :
        args :
        kwargs :

        Returns
        -------

        """

        '''        
        # my code
        dt = self.T / n
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d)
        disc = np.exp(-self.r * dt)  # discount rate

        # initialize
        simulator=BinomialTreeSimulator(S0=self.S0,n=n,u=u,d=d)
        price_mat=simulator.gen_tree()
        '''

        assert self.opttype=='C'
        # Initialise parameters
        S0 = self.S0 # initial stock price
        K = self.K  # strike price
        T = self.T  # time to maturity in years
        r = self.r  # annual risk-free rate
        vol = self.sigma  # volatility (%)
        div = 0  # continuous dividend yield

        # Heston parameters
        kappa = 5.0
        vt0 = vol ** 2  # variance
        theta = 0.2 ** 2
        sigma = 0.02

        # fast steps
        N = n  # number of time intervals
        M = 1000  # number of simulations

        # Precompute constants
        dt = T / N

        # Heston model adjustments for time steps
        kappadt = kappa * dt
        sigmasdt = sigma * np.sqrt(dt)

        # Generate Wiener variables
        W = np.random.normal(0.0, 1.0, size=(N, M, 2))

        # arrays for storing prices and variances
        St = np.full(shape=(N + 1, M), fill_value=S0)
        vt = np.full(shape=(N + 1, M), fill_value=vt0)

        # array for storing maximum's
        St_max = np.full(shape=(M), fill_value=S0)

        for j in range(1, N + 1):
            # Simulate variance processes
            vt[j] = vt[j - 1] + kappadt * (theta - vt[j - 1]) + sigmasdt * np.sqrt(vt[j - 1]) * W[j - 1, :, 0]

            # Simulate log asset prices
            nudt = (r - div - 0.5 * vt[j]) * dt
            St[j] = St[j - 1] * np.exp(nudt + np.sqrt(vt[j] * dt) * W[j - 1, :, 1])

            mask = np.where(St[j] > St_max)
            St_max[mask] = St[j][mask]

        # Compute Expectation and SE
        CT = np.maximum(0, St_max - K)
        C0_fast = np.exp(-r * T) * np.sum(CT) / M

        logging.warning("needs to be checked")

        return C0_fast


if __name__ == '__main__':
    # S0 = 100  # initial stock price
    # K = 100  # strike price
    # T = 1  # time to maturity in years
    # H = 125  # up-and-out barrier price/value
    # r = 0.06  # annual risk-free rate
    # N = 3000  # number of time steps
    # opttype = 'C'  # Option Type 'C' or 'P'
    #
    # sigma=0.05
    # dt = T / N
    # u = np.exp(sigma * np.sqrt(dt))
    # d = 1 / u

    # print(american_option_slow(S0=S0, K=K, T=T, r=r, sigma=sigma, n=N, q=q, type_=type_, underlying='stock'))
    # print(american_slow_tree(S0=S0,K=K,T=T,r=r,sigma=sigma,n=N,opttype='P'))



    # S0 = 18675  # initial stock price
    # K = 17100  # strike price
    # T = 11/12  # time to maturity in years 02-12-2022 ~ 30-10-2023
    # Hd = 17200  # lower bound
    # r = 0.03451  # annual risk-free rate at 02-12-2022
    # N = 3000  # number of time steps
    # opttype = 'C'  # Option Type 'C' or 'P'
    #
    # sigma=0.18
    # dt = T / N
    # u = np.exp(sigma * np.sqrt(dt))
    # d = 1 / u
    #
    # pricing_engine=BarrierBinomialTreePricingEngine()
    # print(pricing_engine.calc_price_barrier_slow_call(S0, K, T, r, sigma, N, uo=None, do=True, ub=None, lb=Hd))
    # # print(barrier_tree_slow(K, T, S0, Hd, r, N, u, d, opttype='C'))



    S0 = 3563.31  # initial stock price
    K = 3550  # strike price
    T = 0.8/12  # time to maturity in years
    r = 0.02741  # annual risk-free rate at
    N = 3000  # number of time steps
    opttype = 'C'  # Option Type 'C' or 'P'
    sigma=0.18
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    option=LookbackOption(S0=S0, K=K, T=T, sigma=sigma, r=r, dividend=0,opttype=opttype,fixed_strike=True)
    engine=LookbackBTPricingEngine(option=option)
    print(engine.calc_price(n=1000))