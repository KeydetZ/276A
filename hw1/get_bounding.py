import numpy as np
import cv2
import matplotlib
import glob
import os
import pylab as plt
import math
import csv
from skimage.measure import label, regionprops

# img = cv2.imread('mask5.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# original_image = cv2.imread('../hw1_starter_code/train/non_barrel_blue/5.png')
#plt.imshow(img)
#plt.show()


# def get_distance(x1, y1, x2, y2):
#     return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
#
# def box_filter_noise(width, height):
#     if (width * height < 500 or width < 10 or height < 10):
#         # means its not a barrel
#         return False
#     else:
#         # it is a barrel:
#         return True
#
# def box_filter_shape(width, height):
#     ratio = height / width
#     if ((1/ratio < 2.2 and 1/ratio > 1.6) or (ratio > 1.6 and ratio < 2.2)):
#         #or 1 / ratio > 0.43 and 1 / ratio < 0.5
#         return True
#     else:
#         return False

def get_bounding(img, original_image):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     contours, hierarchy = cv2.findContours(np.copy(img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     area = [0]
#
#     # initialize a flag
#     barrel_found = False
#
#     # loop through possible blue barrels to pass tests.
#     final_width = []
#     final_height = []
#     regions = []
#     for cont in contours:
#         rect = cv2.minAreaRect(cont)
#         region = np.int0(cv2.boxPoints(rect))
#
#
#         width = get_distance(region[0, 0], region[0, 1], region[1, 0], region[1, 1])
#         height = get_distance(region[1, 0], region[1, 1], region[2, 0], region[2, 1])
#
#         distances = []
#         for point in region:
#             distance = get_distance(point[0], point[1], 0, 0)
#             distances.append(distance)
#
#         top_left_index = np.argmin(distances)
#         top_left = region[top_left_index]
#         btm_right = region[(top_left_index + 2)%4]
#
#         # i=2
#         # list[i:]+[0:i-1]
#
#
#         # print('top_left', top_left)
#         # print('bottom_right', btm_right)
#         #print(top_left)
#         new_region = np.array([top_left, btm_right])
#         new_region = new_region.flatten()
#         #print(new_region.shape)
#
#         #print('region', new_region)
#         # print(width)
#         # print(height)
#         if (box_filter_noise(width, height) and box_filter_shape(width, height)):
#         #if (box_filter_shape(width, height)):
#         #if (box_filter_noise(width, height)):
#             #if (box_filter_noise(width, height)):
#             # if true means its a barrel:
#             # final_height.append(height)
#             # final_width.append(width)
#
#             #barrel_found = True
# #            print(new_region.shape)
# #            print(new_region)
#             regions.append(new_region)

    # Initialize the boxes list
    regions = []
    label_image = label(img)
    for region in regionprops(label_image):

        # this is actually box_filter_noise:
        if (region.area >= 1500):
            # get four points of each box:
            left_row, btm_col, right_row, top_col = region.bbox

            # get height:
            height = right_row - left_row

            # get width:
            width = top_col - btm_col

            # get ratio to filter out wrong shape:
            ratio = height / width

            # this is a potential barrel:
            if (ratio > 1 and ratio < 2.2):
                box = np.array([btm_col, left_row, top_col, right_row])
                regions.append(box)
            # rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
            # ax.add_patch(rect)
        # ax.set_axis_off()
        # plt.tight_layout()
        # plt.show()
    return regions
#cv2.drawContours(original_image, [region], 0, (127, 255, 127), 3)
#print(width)
#print(height)
#print(height/width)
    #return regions
# print(final_width)
# print(final_height)
