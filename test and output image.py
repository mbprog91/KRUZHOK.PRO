import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys
import math
from binarization import binar_image, resize_image

img1 = cv2.imread(r'KRUZHOK.PRO/logo_kd.jpg', 0)          # queryImage
img2 = cv2.imread(r'KRUZHOK.PRO/test.jpg', 0)             # trainImage

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
print(des1, des2)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1, des2)

# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, img2, flags=2)
cv2.imwrite(r'KRUZHOK.PRO/output.png', img3)