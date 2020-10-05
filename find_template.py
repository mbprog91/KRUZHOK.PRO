import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys
import math
from binarization import binar_image, resize_image


def matches(image1, image2):
    # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY).astype('uint8')
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
    # print(descriptors1, descriptors2)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    return len(matches)


image_to_be_found = cv2.imread(r'KRUZHOK.PRO/logo_kd.jpg')
image = cv2.imread(r'KRUZHOK.PRO/test.jpg')

# img_process = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
data_binar = binar_image(image)
for key in data_binar:
    image_key = data_binar[key]
    # cv2.imshow('image key ' + key, image_key)
    matches_found = matches(image_to_be_found, image_key)
    print('matches found == ', matches_found)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

    matches_string = 'Matches:'
    matches_string += str(matches_found)

    if matches_found < 200:
        print('net kruzhka')
    else:
        print('kruzhok')


