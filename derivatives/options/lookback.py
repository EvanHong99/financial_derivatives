# -*- coding=utf-8 -*-
# @Project  : AMA568-advanced_topics_in_quantitative_finance
# @FilePath : financial_derivatives/derivatives/options
# @File     : lookback.py
# @Time     : 2023/10/28 21:46
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:


from .base_option import BaseOption

class LookbackOption(BaseOption):

    def __init__(self, S0=None, K=None, T=None, sigma=None, r=None, dividend=None, opttype='C', fixed_strike=True, underlying=None, name='', pricing_engine=None, is_expired=False, multiplier=10000,
                 *args, **kwargs):
        super().__init__(S0=S0, K=K, T=T, sigma=sigma, r=r, dividend=dividend, underlying=underlying, name=name, pricing_engine=pricing_engine, is_expired=is_expired, multiplier=multiplier, *args, **kwargs)
        self.opttype=opttype
        self.fixed_strike=fixed_strike


    def calc_price(self):
        pass

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


