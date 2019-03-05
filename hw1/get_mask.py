import numpy as np
# import pandas as pd
from matplotlib import pyplot as plt
#import glob
import os
import cv2

import pylab as pl
import math
import csv

from GaussianND import Gaussian


def get_mu_cov(data):
    with open(data) as input:
        reader = csv.reader(input, delimiter=',', quotechar='|')
        mu = [float(ele) for ele in next(reader)]
        cov = [[float(row[0]), float(row[1]), float(row[2])] for row in reader]

    return np.array(mu), np.array(cov)


def get_mask(image, gaussian):
    barrel_blue_mu, barrel_blue_cov = get_mu_cov(gaussian[0])
    brown_mu, brown_cov = get_mu_cov(gaussian[1])
    sky_black_mu, sky_black_cov = get_mu_cov(gaussian[2])
    sky_white_mu, sky_white_cov = get_mu_cov(gaussian[3])
    green_mu, green_cov = get_mu_cov(gaussian[4])
    red_mu, red_cov = get_mu_cov(gaussian[5])
    non_barrel_blue_mu, non_barrel_blue_cov = get_mu_cov(gaussian[6])

    HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # reshape the image to improve performance
    # change each pixel to double instead of int to avoid round off.
    HSV_image = HSV_image.reshape(-1, 3)
    HSV_image = HSV_image.astype(np.double)

    # the image was normalized from 0 to 1.
    # max h 179, max sv 255
    # use float .0
    HSV_image[:, 0] = HSV_image[:, 0] / 179.0
    HSV_image[:, 1] = HSV_image[:, 1] / 255.0
    HSV_image[:, 2] = HSV_image[:, 2] / 255.0

    # datum = np.empty([len(img), len(img[0]), 3])

    # g_green = Gaussian(green_mu, green_cov)

    mask = np.zeros((image.shape[0], image.shape[1]))
    datum = HSV_image.shape[0]

    g_barrel_blue = Gaussian(barrel_blue_mu, barrel_blue_cov)
    g_brown = Gaussian(brown_mu, brown_cov)
    g_sky_white = Gaussian(sky_white_mu, sky_white_cov)
    g_green = Gaussian(green_mu, green_cov)
    g_red = Gaussian(red_mu, red_cov)
    g_non_barrel_blue = Gaussian(non_barrel_blue_mu, non_barrel_blue_cov)
    g_sky_black = Gaussian(sky_black_mu, sky_black_cov)

    for i in range(datum):
        pixel = HSV_image[i]

        pdf_barrel_blue = g_barrel_blue.pdf(pixel)
        pdf_brown = g_brown.pdf(pixel)
        pdf_sky_white = g_sky_white.pdf(pixel)
        pdf_green = g_green.pdf(pixel)
        pdf_red = g_red.pdf(pixel)
        pdf_non_barrel_blue = g_non_barrel_blue.pdf(pixel)
        pdf_sky_black = g_sky_black.pdf(pixel)

        max_prob = max(pdf_brown, pdf_sky_white, pdf_sky_black, pdf_barrel_blue, pdf_red, pdf_green, pdf_non_barrel_blue)
        # if (HSV_image[i][j][0] > 100 and HSV_image[i][j][0] < 140 and max_prob == pdf_barrel_blue ):
        if max_prob == pdf_barrel_blue:
            # if (max_prob == pdf_barrel_blue or max_prob == pdf_dark_barrel_blue):
            # mask[i][j] = [255, 255, 255]
            x = i // 1200
            y = i % 1200
            mask[x][y] = 1

    mask = (mask.astype(np.uint8)) * 255
    # mask_image = label(mask)
    # for region in regionprops(mask_image):
    #     # take regions with large enough areas
    #     if region.area > 500:
    #         # draw rectangle around segmented coins
    #         minr, minc, maxr, maxc = region.bbox
    #         height = maxr - minr
    #         width = maxc - minc
    #         ratio = height / width
    #         if ratio < 1.6 or ratio > 2.2:
    #             for i in range(minr, maxr):
    #                 for j in range(minc, maxc):
    #                     mask[i][j] = 0
    #     if region.area < 500:
    #         minr, minc, maxr, maxc = region.bbox
    #         for i in range(minr, maxr):
    #             for j in range(minc, maxc):
    #                 mask[i][j] = 0
    # mask = np.uint8(mask)
    # #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(np.copy(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # area = [0]

    # initialize a flag
    # barrel_found = False

    # loop through possible blue barrels to pass tests.
    # final_width = []
    # final_height = []
    # regions = []
    for cont in contours:
        rect = cv2.minAreaRect(cont)
        region = np.int0(cv2.boxPoints(rect))

        x1, y1 = (region[0, 0], region[0, 1])
        x2, y2 = (region[3, 0], region[3, 1])

        # print(x1, y1, x2, y2)

        width = get_distance(region[0, 0], region[0, 1], region[1, 0], region[1, 1])
        height = get_distance(region[1, 0], region[1, 1], region[2, 0], region[2, 1])

        # print(width)
        # print(height)
        if (box_filter_noise(width, height) and box_filter_shape(width, height)):
            # is a barrel
            continue
        else:
            for i in range(min(x1, x2), max(x1, x2)):
                for j in range(min(y1, y2), max(y1, y2)):
                    mask[j][i] = 0
        # if (box_filter_noise(width, height)):
        # if true means its a barrel:
        # final_height.append(height)
        # final_width.append(width)

        # barrel_found = True
        # regions.append(region)
        # cv2.drawContours(original_image, [region], 0, (127, 255, 127), 3)
    # dilating the mask
    img_dilated = cv2.dilate(mask, np.ones((8, 8), np.uint8), iterations=1)
    img_erode = cv2.erode(img_dilated, np.ones((8, 8), np.uint8), iterations=1)
    return img_erode


def get_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def box_filter_noise(width, height):
    #if (width < 50 or height < 50):
    if (width * height < 1500 or width < 10 or height < 10):
        # means its not a barrel
        return False
    else:
        # it is a barrel:
        return True


def box_filter_shape(width, height):
    ratio = height / width
    if ((1 / ratio < 2.2 and 1 / ratio > 1.6) or (ratio > 1.6 and ratio < 2.2)):
        # or 1 / ratio > 0.43 and 1 / ratio < 0.5
        return True
    else:
        return False
# mask = get_mask()
# # vget_mask = np.vectorize(get_mask)
# # mask = vget_mask(image)
# mask = np.uint8(mask)
# mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
#
# # dilating the mask
# img_dilated = cv2.dilate(thresh, np.ones((15, 15), np.uint8), iterations=1)
# img_erode = cv2.erode(img_dilated, np.ones((15, 15), np.uint8), iterations=1)
# cv2.imwrite('mask5.png', img_erode)
# cv2.imshow('dilated', img_erode)
# cv2.waitKey(10)
# cv2.destroyAllWindows()
# # cv2.waitKey(100)


# kernel = np.ones((3, 3), np.uint8)
# # closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
# #
# # # Background area using Dialation
# # bg = cv2.dilate(closing, kernel, iterations=1)
# #
# # # Finding foreground area
# # dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)
# # ret, fg = cv2.threshold(dist_transform, 0.02 * dist_transform.max(), 255, 0)


# #print(mu)
# #print(sigma)
# print(pdf_barrel_blue)
# p_barrel_blue =
# p_nonbarrel_blue =
# p_brown =
# p_sky_black =
# p_sky_white =
