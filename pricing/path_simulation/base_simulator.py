# -*- coding=utf-8 -*-
# @Project  : AMA568-advanced_topics_in_quantitative_finance
# @FilePath : financial_derivatives/pricing/path_simulation
# @File     : base_simulator.py
# @Time     : 2023/11/27 23:05
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description: 功能类

from typing import Optional
import logging
import matplotlib.pyplot as plt
import numpy as np


class BaseSimulator(object):
    def __init__(self, T: float, M, N, S0=100, *args, **kwargs):
        self.T = T
        self.M = M
        self.N = N
        self.S0 = S0
        self.args = args
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def gen_series(self, T: float, N):
        pass

    def gen_matrix(self, T: float, M, N) -> np.ndarray:
        pass

    def simulate(self) -> np.ndarray:
        pass
