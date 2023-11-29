# -*- coding=utf-8 -*-
# @Project  : AMA568-advanced_topics_in_quantitative_finance
# @FilePath : financial_derivatives/pricing/path_simulation
# @File     : brownian_motion.py
# @Time     : 2023/10/28 14:54
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:

from typing import Optional
import logging
import matplotlib.pyplot as plt
import numpy as np

from financial_derivatives.pricing.path_simulation.base_simulator import BaseSimulator

class BrownianMotionSimulator(BaseSimulator):
    @staticmethod
    def gen_series(T: float, N, *args, **kwargs):
        return BrownianMotionSimulator.gen_matrix(T=T, M=1, N=N)

    @staticmethod
    def gen_matrix(T: float, M, N, S0=1) -> np.ndarray:
        """

        Parameters
        ----------
        T : float
            terminal time
        M : int,
            number of series
        N : int,
            number of time steps

        Returns
        -------
        brownian motion matrix : np.ndarray,
            m serires $\times$ n time steps

        """
        dt = T / N
        # Bt2 - Bt1 ~ Normal with mean 0 and variance t2-t1
        dB = np.sqrt(dt) * np.random.normal(size=(M, N - 1))
        B0 = np.zeros(shape=(M, 1))
        B = np.cumsum(np.concatenate((B0, dB), axis=1), axis=1)
        logging.warning("该函数仅对一些列dB进行相加，若要生成股价序列，需要使用GBM几何布朗运动")
        return B


class GeometricBrownianMotionSimulator(BaseSimulator):
    @staticmethod
    def gen_matrix(T, M: int = 1, N: int = 1, S0: float = 1, r: float = 0, sigma: float = 0.20):
        """

        .. math::
            dlnS=(r-0.5{\sigma}^2)dt+\sigma*dB


        Parameters
        ----------
        T :
        M : int,
            number of simulations
        N :
        S0 :
        r :
        sigma :

        Returns
        -------

        """
        dt = T / N
        lnS0 = np.full(shape=(M, 1), fill_value=np.log(S0))
        dB = np.random.normal(loc=0, scale=np.sqrt(dt), size=(M, N - 1))
        delta_mat = (r - 0.5 * sigma ** 2) * dt + sigma * dB
        mat = np.concatenate((lnS0, delta_mat), axis=1)
        mat = np.cumsum(mat, axis=1)
        mat = np.exp(mat)
        return mat

    @staticmethod
    def gen_series(T, N, S0: float = 1, r: float = 0, sigma: float = 0.20):
        return GeometricBrownianMotionSimulator.gen_matrix(T=T, M=1, N=N, S0=S0, r=r, sigma=sigma)


if __name__ == '__main__':
    generator = GeometricBrownianMotionSimulator()
    series_mat = generator.gen_matrix(1, 10, N=1000, r=0.04, sigma=0.2)
    plt.plot(series_mat.T)

    plt.show()
