import csv
import glob

from roipoly import RoiPoly
from matplotlib import pyplot as plt
import numpy as np
import cv2

# image = cv2.imread('1.png')
# #print(image.shape)
# #RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = image[:,:,0]
# plt.imshow(image)
#
# my_roi = RoiPoly(color='r')
# # my_roi.display_roi()
# #print(np.shape(image))
# mask = my_roi.get_mask(image)
# plt.imshow(mask) # show the binary signal mask
# plt.show()

for image in glob.glob('../hw1_starter_code/train/non_barrel_blue/28.png'):
    img = cv2.imread(image)

    # get the RGB image in order to show the original image.
    RGB_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    HSV_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # show RGB image
    plt.imshow(RGB_image)
    #plt.imshow(HSV_image)
    #plt.show()

    my_roi = RoiPoly(color='r')

    # create an area which has the same dimension as the original image
    area = np.zeros((HSV_image.shape[0], HSV_image.shape[1]))
    mask = my_roi.get_mask(area)

    plt.imshow(mask)
    #plt.show()

    mask = np.array(mask, dtype=np.uint8)
    #print(mask.shape)

    area_of_interest = cv2.bitwise_and(HSV_image, HSV_image, mask = mask)
    #plt.imshow(area_of_interest)
    #plt.show()
    #print(area_of_interest.shape)

    #print pixel locations of interested area where the image is not black.
    locations = np.where(area_of_interest != [0, 0, 0])
    #plt.imshow(locations)
    #plt.show()
    #print(locations[0].size)
    collection = [0, 0, 0]

    for i in range(locations[0].size):
        # the x-location of the image
        x = locations[0][i]
        # the y-location of the image
        y = locations[1][i]
        # dont care about the z
        #print(x)
        #print(y)

        # stack vertically using vstack
        # the collection of pixel color info (HSV)
        collection = np.vstack((collection, area_of_interest[x, y, :]))
        #print(collection.shape)

    #convert to nparray
    # print(collection.shape)
    # print(collection)
    collection = np.array(collection)
    #print(collection.shape)

    #remove the first row [0, 0, 0]
    collection = np.delete(collection, (0), axis = 0)
    #print(collection)

    with open('non_barrel_blue.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerows(collection)


