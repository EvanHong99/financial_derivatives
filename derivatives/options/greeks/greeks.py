# -*- coding=utf-8 -*-
# @Project  : AMA568-advanced_topics_in_quantitative_finance
# @FilePath : financial_derivatives/derivatives/options/greeks
# @File     : greeks.py
# @Time     : 2023/10/22 22:09
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:

class Greeks:
    def __init__(self, delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0):
        """
        :param delta: 标准delta
        :param gamma: 标准gamma
        :param vega:
        :param theta:
        :param rho:
        """
        self.delta = delta
        self.gamma = gamma
        self.vega = vega
        self.theta = theta
        self.rho = rho

    def __add__(self, other):
        """
        :param other: 只能是greeks
        :return:
        """
        self.delta += other.delta
        self.gamma += other.gamma
        self.vega += other.vega
        self.theta += other.theta
        self.rho += other.rho
        return self

    def __sub__(self, other):
        """

        :param other:
        :return:
        """
        self.delta -= other.delta
        self.gamma -= other.gamma
        self.vega -= other.vega
        self.theta -= other.theta
        self.rho -= other.rho
        return self

    def __mul__(self, other):
        """
        :param other: 只能是标量
        :return:
        """
        self.delta *= other
        self.gamma *= other
        self.vega *= other
        self.theta *= other
        self.rho *= other
        return self

    def __truediv__(self, other):
        """
        :param other: 只能是标量
        :return:
        """
        self.delta /= other
        self.gamma /= other
        self.vega /= other
        self.theta /= other
        self.rho /= other
        return self

    def __str__(self):
        s = """
            delta: {delta}
            gamma: {gamma}
            vega:  {vega}
            theta: {theta}
            rho:   {rho}
            """.format(
            delta=self.delta,
            gamma=self.gamma,
            vega=self.vega,
            theta=self.theta,
            rho=self.rho
        )
        return s


if __name__ == '__main__':
    delta = 0.5
    gamma = 0.1
    vega = 0.3
    theta = 0.5
    rho = 0.2
    greeks_1 = Greeks(delta, gamma, vega, theta, rho)
    greeks_2 = Greeks(delta, gamma, vega, theta, rho)
    print(greeks_1 - greeks_2)