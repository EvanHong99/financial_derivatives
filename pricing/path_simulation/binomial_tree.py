# -*- coding=utf-8 -*-
# @Project  : AMA568-advanced_topics_in_quantitative_finance
# @FilePath : financial_derivatives/pricing/path_simulation
# @File     : binomial_tree.py
# @Time     : 2023/10/28 22:04
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:
import numpy as np


class BinomialTreeSimulator(object):
    def __init__(self, S0, n, u, d):
        self.S0 = S0
        self.n = n
        self.u = u
        self.d = d

    def gen_tree(self):
        """
        decompose the tree into three parts: first colume [1,1,1,1,1].T, last row of [d,d,d,d], upper right matrix consists of u and d

        matrix shape= (n+1,n+1)

        1	u	uu	uuu	uuuu		1	u	u	u	u
        1	d	du	duu	duuu		1	d	u	u	u
        1	1	dd	ddu	dduu   ->   1	d	d	u	u
        1	1		ddd	dddu		1	d	d	d	u
        1	1			dddd		1	d	d	d	d


        Returns
        -------

        """
        upper_right = np.full(shape=(self.n, self.n), fill_value=self.d)
        idx = np.triu(np.full(shape=(self.n, self.n), fill_value=True), k=0)
        upper_right[idx] = self.u
        last_row = np.full(shape=(1, self.n), fill_value=self.d)
        first_colume = np.full(shape=(self.n + 1, 1),fill_value=self.S0)
        idx = np.triu(np.full(shape=(self.n, self.n), fill_value=True), k=0)
        upper_right[idx] = self.u
        mat = np.concatenate((upper_right, last_row), axis=0)
        mat = np.concatenate((first_colume, mat), axis=1)
        mat = np.cumprod(mat, axis=1)
        return mat


if __name__ == '__main__':
    n = 5
    d = 0.9
    u = 1.1
    upper_right = np.full(shape=(n, n), fill_value=d)
    last_row = np.full(shape=(1, n), fill_value=d)
    first_colume = np.ones(shape=(n + 1, 1))
    idx = np.triu(np.full(shape=(n, n), fill_value=True), k=0)
    upper_right[idx] = u
    mat = np.concatenate((upper_right, last_row), axis=0)
    print(mat)
    mat = np.concatenate((first_colume, mat), axis=1)
    print(mat)
    mat = np.cumprod(mat, axis=1)
    print(mat)
