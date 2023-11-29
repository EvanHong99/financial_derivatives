# -*- coding=utf-8 -*-
# @File     : european.py
# @Time     : 2023/9/24 20:50
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : AMA568-advanced_topics_in_quantitative_finance
# @Description:

"""
标准欧式看涨
Vanilla Call Function Based On Black-Scholes Model
"""

import numpy as np
from scipy.stats import norm
from base_option import BaseOption
# from greeks import greeks


class VanillaCallOption(BaseOption):
    def __init__(self, S0, k, T, sigma, r, dividend):
        BaseOption.__init__(self, S0, T, r, dividend)
        self.k=k
        self.v=sigma
        raise NotImplementedError("这个类只是抄的，完全未适配")

    def calc_price(self):
        if self.is_expired:
            price = np.maximum(self.S0 - self.k, 0)
        else:
            price = self.S0 * np.exp(-self.d * self.T) * norm.cdf(
                self.blsd(self.S0 / self.k, self.r - self.d + 0.5 * self.v * self.v, self.v, self.T)) - self.k * np.exp(
                -self.r * self.T) * norm.cdf(
                self.blsd(self.S0 / self.k, self.r - self.d - 0.5 * self.v * self.v, self.v, self.T))
        self.price = price

        return self.price

    def calc_delta(self):
        if self.is_expired:
            delta = 0
        else:
            delta = np.exp(-self.d * self.T) * norm.cdf(
                self.blsd(self.S0 / self.k, self.r - self.d + 0.5 * self.v * self.v, self.v, self.T))
        return delta

    def calc_gamma(self):
        if self.is_expired:
            gamma = 0
        else:
            gamma = np.exp(-self.d * self.T) * norm.pdf(
                self.blsd(self.S0 / self.k, self.r - self.d + 0.5 * self.v * self.v, self.v, self.T)) / (
                            self.S0 * self.v * (self.T ** 0.5))
        return gamma

    def calc_vega(self):
        if self.is_expired:
            vega = 0
        else:
            vega = self.S0 * np.exp(-self.d * self.T) * norm.pdf(
                self.blsd(self.S0 / self.k, self.r - self.d + 0.5 * self.v * self.v, self.v, self.T)) * (self.T ** 0.5)
        return vega

    def calc_theta(self):
        if self.is_expired:
            theta = 0
        else:

            theta = self.d * self.S0 * np.exp(-self.d * self.T) * norm.cdf(
                self.blsd(self.S0 / self.k, self.r - self.d + 0.5 * self.v * self.v, self.v, self.T)) - np.exp(
                -self.d * self.T) * self.S0 * norm.pdf(
                self.blsd(self.S0 / self.k, self.r - self.d + 0.5 * self.v * self.v, self.v, self.T)) * self.v / (
                            2 * (self.T ** 0.5)) - self.r * self.k * np.exp(-self.r * self.T) * norm.cdf(
                self.blsd(self.S0 / self.k, self.r - self.d - 0.5 * self.v * self.v, self.v, self.T))
        return theta

    def calc_rho(self):
        return 0


if __name__ == '__main__':
    s = 1
    k = 1
    r = 0.05
    d = 0.05
    t = 1
    v = 0.2
    option = VanillaCallOption(s, k, t, v, r, d)
    price = option.calc_price()
    print('price: {price}'.format(price=price))

    greeks = option.calc_greeks()
    print('delta: {delta}'.format(delta=greeks.delta))
    print('gamma: {gamma}'.format(gamma=greeks.gamma))
    print('vega: {vega}'.format(vega=greeks.vega))
    print('theta: {theta}'.format(theta=greeks.theta))
