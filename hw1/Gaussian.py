import math

import numpy as np


class Gaussian:
    def __init__(self, mu, sigma):
        '''
        mu: mean
        sigma: standard deviation
        '''
        self.mu = mu
        self.sigma = sigma

    # probability density function:
    def pdf(self, datum):
        "Probability of a data point given the current parameters"
        u = (datum - self.mu) / abs(self.sigma)
        #print('u:', u)
        y = (1 / (math.sqrt(2 * math.pi) * abs(self.sigma))) * np.exp(-u * u / 2)
        #y = (1 / (math.sqrt(2 * math.pi) * abs(self.sigma))) * math.exp(-(datum-self.mu) ** 2 / (2 * (self.sigma ** 2)))
        #print('y:', y)
        return y

    # def __repr__(self):
    #     '''
    #     :return: print Gaussian model values.
    #     '''
    #     return 'Gaussian({0:4.6}, {1:4.6})'.format(self.mu, self.sigma)

