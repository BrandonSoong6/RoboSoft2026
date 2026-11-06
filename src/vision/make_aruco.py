import cv2
import cv2.aruco as aruco
import numpy as np

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
img = aruco.generateImageMarker(aruco_dict, id=33, sidePixels=200)
cv2.imwrite(f"marker33.png", img)
# for i in range(1,250):
#     img = aruco.generateImageMarker(aruco_dict, id=i, sidePixels=200)
#     cv2.imwrite(f"marker{i}.png", img)
