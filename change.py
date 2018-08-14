import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('school_test_2.jpg')

def move(img):
    ret, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return img[y:y+h, x:x+w]

def judge(img):
    # areas in source img
    # pt1 = np.float32([[164, 44], [1566, 125], [1566, 522], [82, 288]])
    pt1 = np.float32([[148, 45], [1197, 147], [1197, 682], [19, 571]])
    # fix the new image size as (x, y)
    x = 1200
    y = 600
    # points in new image
    pt2 = np.float32([[0, 0], [x, 0], [x, y], [0, y]])
    # get the transform matrix
    T = cv2.getPerspectiveTransform(pt1, pt2)
    # transform
    new_img = cv2.warpPerspective(img, T, (x, y))
    print(new_img.shape)
    # Show the new image.
    # cv2.imshow('newImage', new_img)
    # cv2.imwrite('school_test_2_edit.jpg', new_img)
    # cv2.imshow('newImage2', new_img[33:350,1:1276])
    # cv2.imshow('newImage2', new_img[65:425, :])
    # cv2.imshow('newImage2', new_img[1:568, 1:1184])
    # cv2.imwrite('use_test_3_edit_2.jpg', new_img[33:350,1:1276])
    # cv2.imwrite('school_test_edit_2.jpg', new_img[65:425, :])
    # cv2.imwrite('school_test_2_edit_2.jpg', new_img[1:568, 1:1184])
    # cv2.waitKey(0)
    return (new_img, new_img[1:568, 1:1184])

img = move(img)
# cv2.imshow('test',img[0:295,0:302])
print(img.shape)
judge(img)
# cv2.imshow('test',img)
cv2.waitKey(0)

# school
# [164,44]
# [1566,125]
# [1566,522]
# [82,288]

# garden
# [93,39]
# [1135,131]
# [1135,464]
# [35, 258]