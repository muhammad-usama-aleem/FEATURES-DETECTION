import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

os.chdir("C:/Users/abdul/OneDrive/Pictures/New folder")
img = cv2.imread("remote.jpg")
img2 = cv2.imread("match.jpg")

orb = cv2.ORB_create(2000)
kp, des = orb.detectAndCompute(img, None)
orb = cv2.ORB_create(500)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

match = bf.match(des, des2)
print(len(match))
match = sorted(match, key=lambda x: x.distance)

draw = cv2.drawMatches(img, kp, img2, kp2, match[:100], None)
cv2.imshow("size",draw)
cv2.waitKey(0)
cv2.destroyAllWindows()
