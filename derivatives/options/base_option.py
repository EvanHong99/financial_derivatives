# -*- coding=utf-8 -*-
# @Project  : AMA568-advanced_topics_in_quantitative_finance
# @FilePath : financial_derivatives/options
# @File     : base_option.py
# @Time     : 2023/10/22 15:54
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:

from abc import ABC
from abc import abstractmethod
from financial_derivatives.pricing.option_pricing_engines.base_engine import BasePricingEngine


class BaseOption(ABC):
    """
    一切定价器、希腊字母计算器都是最终整合到BaseOption类
    """

    def __init__(self, S0=None, K=None, T=None, sigma=None, r=None, dividend=None, underlying=None, name='',
                 pricing_engine=None, is_expired=False, multiplier=10000, *args, **kwargs):
        """
        使用不同的pricing engines

        Attributes
        ----------
        pricing_engine : BasePricingEngine,
            可以用不同的定价引擎

        Returns
        -------

        """
        self.S0 = S0
        self.T = T
        self.sigma = sigma
        self.r = r
        self.dividend = dividend
        self.underlying = underlying
        self.name = name
        self.pricing_engine = pricing_engine
        self.is_expired = is_expired
        self.multiplier = multiplier
        self.K = K
        self.args = args
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    @abstractmethod
    def calc_price(self):
        pass

    @abstractmethod
    def calc_delta(self):
        pass

    @abstractmethod
    def calc_gamma(self):
        pass

    @abstractmethod
    def calc_vega(self):
        pass

    @abstractmethod
    def calc_theta(self):
        pass

    @abstractmethod
    def calc_rho(self):
        pass
