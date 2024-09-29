import cv2  #OpenCV包
import numpy as np

# 首先确定原图片的基本信息：数据格式，行数列数，通道数
rows=2464#图像的行数
cols=3264#图像的列数
channels =1# 图像的通道数，灰度图为1
path = "raw8_image.raw"
# 利用numpy的fromfile函数读取raw文件，并指定数据格式
img=np.fromfile(path, dtype='uint8')
# 利用numpy中array的reshape函数将读取到的数据进行重新排列。
img=img.reshape(rows, cols, channels)

# 展示图像
cv2.imshow('Infared image',img)
# 如果是uint16的数据请先转成uint8。不然的话，显示会出现问题。
cv2.waitKey()
cv2.destroyAllWindows()
print('ok')