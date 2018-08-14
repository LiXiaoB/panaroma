import numpy as np
import cv2
import os
import math
from matplotlib import pyplot as plt

# DIRECTORY = 'school_test'
# DIRECTORY = 'school_test_2'
# DIRECTORY = 'garden_test'
# DIRECTORY = 'final'
DIRECTORY = 'final_2'
# DIRECTORY = '33'
# DIRECTORY = 'pano_test_3'
# DIRECTORY = 'Xue-Mountain-Enterance'

def load_images_from_folder(folder):
    images = []
    for file in os.listdir(folder):
        if file.endswith('.jpg'):
            f = os.path.join(folder, file)
            img = cv2.imread(f)
            if img is not None:
                images.append(img)
    return images


def cylinderical_warping(img,f):
    h, w = img.shape[0], img.shape[1]
    warped_image = np.zeros((h, w, 3), np.uint8)
    xc = w//2
    yc = h//2

    # for y in range(h):
    #     for x in range(w):
    #         xp = int(f * np.arctan((x -xc )/f) + xc)
    #         yp = int(f * (y-yc) / (np.sqrt((x-xc) ** 2 + f ** 2)) + yc)
    #         warped_image[yp,xp] = img[y,x]

    for row in range(1,h):
        for col in range(1,w):
            x = col - xc
            y = row - yc

            theta = math.atan(x / f)
            h = y / math.sqrt(x ** 2 + f ** 2)

            xcap = int(f * theta + xc)
            ycap = int(f * h + yc)

            warped_image[ycap, xcap] = img[row, col]

    # crop black edge
    # ref: http://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
    ret, thresh = cv2.threshold(cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return warped_image[y:y+h, x:x+w]


def get_homography(img_1, img_2):
    sift = cv2.xfeatures2d.SIFT_create()
    MIN_MATCH_COUNT = 5

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img_1, None)
    kp2, des2 = sift.detectAndCompute(img_2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    # print('good',len(good))
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        # matchesMask = mask.ravel().tolist()
        #
        # h, w, n = img_1.shape
        # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # img_3 = cv2.warpPerspective(img_2,M)

        return M

    return None

# def image_warp(img_1, img_2, homography):
#     result = cv2.warpPerspective(img_2, homography, (img_1.shape[1] + img_2.shape[1], img_1.shape[0]))
#     cv2.imshow('test',result)
#     cv2.waitKey(0)
#     # result[0:img_2.shape[0], 0:img_1_croped.shape[1]] = img_1_croped
#     return result
#
# def image_align(img_1, img_2):
#     M = get_homography(img_2, img_1)
#     result = cv2.warpPerspective(img_2, M, (img_1.shape[1], img_1.shape[0]))
#     cv2.imshow('test',result)
#     cv2.waitKey(0)


def get_stitched_image(img1, img2, M):
    # Get width and height of input images
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    # Get the canvas dimesions
    img1_dims = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    img2_dims_temp = np.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)

    # Get relative perspective of second image
    img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

    # Resulting dimensions
    result_dims = np.concatenate((img1_dims, img2_dims), axis=0)

    # Getting images together
    # Calculate dimensions of match points
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    # Create output array after affine transformation
    transform_dist = [-x_min, -y_min]
    transform_array = np.array([[1, 0, transform_dist[0]],
                                [0, 1, transform_dist[1]],
                                [0, 0, 1]])

    # Warp images to get the resulting image
    result_img = cv2.warpPerspective(img2, transform_array.dot(M),
                                     (x_max - x_min, y_max - y_min))
    result_img[transform_dist[1]:w1 + transform_dist[1],
    transform_dist[0]:h1 + transform_dist[0]] = img1

    # Return the result
    return result_img

if __name__ == "__main__":

    #focal length in pixels = (image width in pixels) * (focal length in mm) / (CCD width in mm)
    focal = 16
    width = 600
    ccd = 23.5
    f = width * focal / ccd
    # f = 704

    images = load_images_from_folder(DIRECTORY)
    for i in range(len(images)):
        images[i] = cylinderical_warping(images[i],f)
    cv2.imshow('temp', images[0])
    cv2.waitKey(0)
    stitched_image = images[0].copy()
    for i in range(1,len(images)):
        print(i)
        if i <8:
            M = get_homography(images[i], stitched_image)
            stitched_image = get_stitched_image(images[i], stitched_image, M)
            if i ==1:
                cv2.imshow('test', stitched_image)
                cv2.waitKey(0)
        # M = get_homography(images[i], stitched_image)
        # stitched_image = get_stitched_image(images[i], stitched_image, M)

    # stitched_image_2 = images[3].copy()
    # for i in range(4, 6):
    #     M = get_homography(stitched_image_2,images[i])
    #     stitched_image_2 = get_stitched_image(stitched_image_2, images[i], M)
    #
    # M = get_homography(stitched_image, stitched_image_2)
    # final = get_stitched_image(stitched_image, stitched_image_2, M)
    cv2.imshow('test', stitched_image)
    # cv2.imwrite('final_2_test.jpg',stitched_image)
    cv2.waitKey(0)
    # plt.imshow(stitched_image)
    # plt.show()






