import numpy as np
#import pandas as pd
import matplotlib
import glob
import os

import pylab as pl
import math
import csv

for one_class in glob.glob('bright.csv'):
    data = []
    with open(one_class) as input:
        reader = csv.reader(input, delimiter=',', quotechar='|')
        data = [[float(row[0]), float(row[1]), float(row[2])] for row in reader]
    data = np.asarray(data)

    # print(data.shape)
    # print(data[0])
    # print(data[0][0])
    # print(type(data[0][0]))

    # the amount of data point in each class
    num_of_pixels = data.shape[0]

    sum = np.sum(data, axis = 0)
    # print('sum = ')
    # print(sum)

    mean = np.divide(sum, num_of_pixels)
    print(mean)
    print(mean.shape)
    print(type(mean))

    mean = np.matrix(mean)
    # the shape of mean is a 1x3 matrix
    print(mean.shape)
    print(type(mean))

    print('mean = ')
    print(mean)

    sum_cov = np.matrix(np.zeros([3, 3]))

    print(data.shape)

    # for each pixel in a class
    for data_point in data:
        # calculate the transpose of one pixel to calculate covariance
        #print(data_point.shape)
        data_point_transpose = np.matrix(data_point).T

        # calculate covariance
        sum_cov += (data_point_transpose - mean) * ((data_point_transpose - mean).T)

    print(sum_cov.shape)
    cov = sum_cov / num_of_pixels
    print('cov = ')
    print(cov)

    # with open('gaussian_' + one_class, 'a') as output:
    #     writer = csv.writer(output)
    #     writer.writerows(np.asarray(mean))
    #     writer.writerows(np.asarray(cov))