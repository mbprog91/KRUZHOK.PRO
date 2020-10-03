import cv2 
import numpy as np 
image = cv2.imread('doubletriple.jpg') 
img_process = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
def resize_image(img):
    scale_percent = 50
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    output = cv2.resize(img, (width, height))
    return output
smth, thresh1 = cv2.threshold(img_process, 120, 255, cv2.THRESH_BINARY)
smth, thresh2 = cv2.threshold(img_process, 120, 255, cv2.THRESH_BINARY_INV)
smth, thresh3 = cv2.threshold(img_process, 120, 255, cv2.THRESH_TRUNC)
smth, thresh4 = cv2.threshold(img_process, 120, 255, cv2.THRESH_TOZERO)
smth, thresh5 = cv2.threshold(img_process, 120, 255, cv2.THRESH_TOZERO_INV)
th2 = cv2.adaptiveThreshold(img_process, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img_process, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
smth, thotsu = cv2.threshold(img_process, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
gauss_process = cv2.GaussianBlur(img_process, (5, 5), 0)
smth, thotsu2 = cv2.threshold(gauss_process, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('1', resize_image(thresh1))
cv2.imshow('2', resize_image(thresh2))
cv2.imshow('3', resize_image(thresh3))
cv2.imshow('4', resize_image(thresh4))
cv2.imshow('5', resize_image(thresh5))
cv2.imshow('6', resize_image(th2))
cv2.imshow('7', resize_image(th3))
cv2.imshow('8', resize_image(thotsu))
cv2.imshow('9', resize_image(thotsu2))
if cv2.waitKey(0) & 0xff == 27: 
    cv2.destroyAllWindows() 