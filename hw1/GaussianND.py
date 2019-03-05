import math

import numpy as np


class Gaussian:
    def __init__(self, mu, cov):
        '''
        mu: mean
        sigma: standard deviation
        '''
        self.mu = mu
        self.cov = cov
        self.inv_cov = np.linalg.inv(self.cov)
        self.a = 1 / math.sqrt(((2 * math.pi) ** 3) * abs(np.linalg.det(self.cov)))

    # probability density function:
    def pdf(self, datum):
        "Probability of a data point given the current parameters"
        # print(datum - self.mu)
        # print(np.transpose(datum - self.mu))
        # print(np.transpose(datum - self.mu) * np.linalg.inv(self.cov) * (datum - self.mu))
        # print(datum)
        # print(datum.shape)
        # print(self.mu.shape)
        #b = np.exp((-0.5) * np.matmul(np.matmul(np.transpose(datum - self.mu), np.linalg.inv(self.cov)) , datum - self.mu))
        b = np.exp((-0.5) * np.matmul(np.matmul(datum - self.mu, self.inv_cov), np.transpose(datum - self.mu)))
        # print(a)
        # print(b)
        y = self.a * b
        #y = 1/(((2 * math.pi) ** 1.5) * abs(np.linalg.det(self.cov)) ** 0.5) * np.exp(-0.5 * np.matrix(datum - self.mu).T * inv(self.cov) * (datum - self.mu))
        return y

    # def __repr__(self):
    #     '''
    #     :return: print Gaussian model values.
    #     '''
    #     return 'Gaussian({0:4.6}, {1:4.6})'.format(self.mu, self.sigma)

