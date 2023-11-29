# -*- coding=utf-8 -*-
# @Project  : AMA568-advanced_topics_in_quantitative_finance
# @FilePath : financial_derivatives/pricing/option_pricing_engines
# @File     : base_engine.py
# @Time     : 2023/10/22 21:44
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:

from abc import ABC,abstractmethod
from typing import Optional


class BasePricingEngine(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def calc_price(self,*args,**kwargs):
        pass

class BaseBarrierPricingEngine(BasePricingEngine):
    def __init__(self,S0, K, T, r, sigma,n,uo:Optional[bool]=None,do:Optional[bool]=None, ub:Optional[float]=None,lb:Optional[float]=None):
        self.S0=S0
        self.K=K
        self.T=T
        self.r=r
        self.sigma=sigma
        self.n=n
        self.uo=uo
        self.do=do
        self.ub=ub
        self.lb=lb

    def calc_price(self, *args, **kwargs):
        pass

class BaseAsianPricingEngine(BasePricingEngine):
    def __init__(self,S0, K, T, r, sigma,fixed_strike=True,opttype='C'):
        self.S0=S0
        self.K=K
        self.T=T
        self.r=r
        self.sigma=sigma
        self.fixed_strike=fixed_strike
        self.opttype=opttype

    def calc_price(self,n=1000, *args, **kwargs):
        pass

class BaseLookbackPricingEngine(BasePricingEngine):
    def __init__(self, S0, K, T, r, sigma, fixed_strike=True, opttype='C'):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.fixed_strike = fixed_strike
        self.opttype = opttype

    def calc_price(self, n=1000, *args, **kwargs):
        pass