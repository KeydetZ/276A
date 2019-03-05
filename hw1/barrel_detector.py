'''
    ECE276A WI19 HW1
    Blue Barrel Detector
    '''
import numpy as np
import os, cv2
from skimage.measure import label, regionprops
#from GaussianND import Gaussian
from get_mask import *
# from Covariance import *
from get_bounding import *


class BarrelDetector():
    def __init__(self):
        '''
            Initilize your blue barrel detector with the attributes you need
            eg. parameters of your classifier
            '''
        self.gaussian = ['new_gaussian_barrel_blue.csv', 'new_gaussian_brown.csv', 'new_gaussian_sky_black.csv', \
                         'new_gaussian_sky_white.csv', \
                          'new_gaussian_green.csv', 'new_gaussian_red.csv', 'new_gaussian_non_barrel_blue.csv']
    
    def segment_image(self, img):
        '''
            Calculate the segmented image using a classifier
            eg. Single Gaussian, Gaussian Mixture, or Logistic Regression
            call other functions in this class if needed
            
            Inputs:
            img - original image
            Outputs:
            mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
            '''
        
        mask_img = get_mask(img, self.gaussian)
        
        #return self.mask
        return mask_img
    
    def get_bounding_box(self, img):
        '''
            Find the bounding box of the blue barrel
            call other functions in this class if needed
            
            Inputs:
            img - original image
            Outputs:
            boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2]
            where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
            is from left to right in the image.
            
            Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
            '''
        #boxes = get_bounding(self.mask, img)
        boxes = get_bounding(self.segment_image(img), img)
        return boxes


# if __name__ == '__main__':
#     folder = r'../hw1_starter_code/train/barrel_blue'
#     img = cv2.imread(os.path.join(folder, '17.png'))
#     my_detector = BarrelDetector(img)
#     #for filename in os.listdir(folder):
#     # read one test image
#     #img = cv2.imread(os.path.join(folder, '17.png'))
#     # cv2.imshow('image', img)
#     # cv2.waitKey(10)
#     # cv2.destroyAllWindows()
#     mask_img = my_detector.segment_image(img)
#     cv2.imwrite('mask17.png', mask_img)
#     # cv2.imshow('mask', mask_img)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     boxes = my_detector.get_bounding_box(img)
#     print(boxes)
#
#     cv2.imshow('Bounding Box(es)', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

if __name__ == '__main__':
    folder = "trainset"
    my_detector = BarrelDetector()
    for filename in os.listdir(folder):
        # read one test image
        img = cv2.imread(os.path.join(folder, filename))


# Display results:
# (1) Segmented images
#     mask_img = my_detector.segment_image(img)
# (2) Barrel bounding box
#    boxes = my_detector.get_bounding_box(img)
# The autograder checks your answers to the functions segment_image() and get_bounding_box()
# Make sure your code runs as expected on the testset before submitting to Gradescope
