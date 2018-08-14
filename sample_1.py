import panorama
import cv2
import change

DIRECTORY = 'school_test_2'
focal = 16
width = 600
ccd = 23.5
f = width * focal / ccd


images = panorama.load_images_from_folder(DIRECTORY)
images = panorama.load_images_from_folder(DIRECTORY)
for i in range(len(images)):
    images[i] = panorama.cylinderical_warping(images[i],f)
stitched_image = images[0].copy()
for i in range(1,len(images)):
    print(i)
    if i <6:
        M = panorama.get_homography(images[i], stitched_image)
        stitched_image = panorama.get_stitched_image(images[i], stitched_image, M)


# cv2.imwrite('school.jpg',stitched_image)
stitched_image = change.move(stitched_image)
(img_1,img_2) = change.judge(stitched_image)
cv2.imshow('school_pre', img_1)
cv2.imshow('school', img_2)
# cv2.imwrite('school_pano.jpg',img_2)
cv2.waitKey(0)