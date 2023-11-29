# -*- coding=utf-8 -*-
# @Project  : AMA568-advanced_topics_in_quantitative_finance
# @FilePath : financial_derivatives/options
# @File     : callable_barrier.py
# @Time     : 2023/10/22 15:46
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description: 可赎回障碍期权，一般发行时都处于深度实值。具体可见文档 https://ew6bbou7lb.feishu.cn/docx/S4K5dNfgDoXeybxOfFLcSQ4QnYd#part-JU09dreuxoqhTWxt2khciU3inGe

from base_option import BaseOption

class BaseCallableBarrier(BaseOption):
    """可赎回障碍期权.

    Attributes
    ----------


    References
    ----------
    [1] https://www.hkex.com.hk/chi/cbbc/cbbcsummarydelist_c.asp?id=403282

    """

    def __init__(self, S0, k, call_price, T, sigma, r, dividend, _type, style, expiry=None, underlying=None, name='', pricing_engine=None, is_expired=False, multiplier=10000):
        """

        Parameters
        ----------
        S0 : underlying price
        k : strike
        call_price : 回购价格
        T : time to maturity
        sigma :
        r :
        dividend :
        _type : direction,
            call or put
        style :
            European or American
        expiry :
        underlying :
        name :
        pricing_engine :
        is_expired :
        multiplier :
        """
        super().__init__(S0, T, sigma, r, dividend, underlying, name, pricing_engine, is_expired=is_expired, multiplier=multiplier)
        self.k=k
        self.call_price=call_price
        self.type=_type
        self.style=style
        self.expiry=expiry

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


class CallableBull(BaseCallableBarrier):
    """down-out call

    """

    def __init__(self, S0, k, call_price, T, sigma, r, dividend, _type, style, expiry=None, underlying=None, name='',
                 pricing_engine=None, is_expired=False, multiplier=10000):
        super().__init__(S0, k, call_price, T, sigma, r, dividend, _type, style, expiry, underlying, name, pricing_engine,
                         is_expired, multiplier)
        assert self.call_price>=self.k

    def calc_price(self):


    def calc_delta(self):
        super().calc_delta()

    def calc_gamma(self):
        super().calc_gamma()

    def calc_vega(self):
        super().calc_vega()

    def calc_theta(self):
        super().calc_theta()

    def calc_rho(self):
        super().calc_rho()


class CallableBear(BaseCallableBarrier):
    """up-out put


    """
    def __init__(self, S0, T, sigma, r, dividend, underlying=None, name='', pricing_engine=None):
        super().__init__(S0, T, sigma, r, dividend, underlying, name, pricing_engine)

    def calc_price(self):
        super().calc_price()

    def calc_delta(self):
        super().calc_delta()

    def calc_gamma(self):
        super().calc_gamma()

    def calc_vega(self):
        super().calc_vega()

    def calc_theta(self):
        super().calc_theta()

    def calc_rho(self):
        super().calc_rho()