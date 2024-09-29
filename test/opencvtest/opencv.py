"""
python 打开raw图像并显示
@zhou 2020/1/8
"""
import numpy as np
import cv2
img = np.fromfile('raw8_image.raw',dtype=np.int8)
img=img.reshape((3264,2464))
cv2.namedWindow("zhou",0)
cv2.imshow("zhou",img)
cv2.waitKey(0)