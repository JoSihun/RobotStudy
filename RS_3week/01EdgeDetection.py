import cv2 as cv
import numpy as np

Path = './Data/'
Name = 'img.jpg'
FullName = Path+Name

img = cv.imread(FullName)
# img = cv.imread(FullName, cv.IMREAD_GRAYSCALE)


# ********* 기본 1X2, 2X1 필터 *********
# X축 Filter
kernel1 = np.array([[-1, 1]])
dst1 = cv.filter2D(img, -1, kernel1)

# Y축 Filter
kernel2 = np.array([[-1], [1]])
dst2 = cv.filter2D(img, -1, kernel2)

cv.imshow('X', dst1)
cv.imshow('Y', dst2)
cv.imshow('Sum', dst1 + dst2)


# # ************* Roberts Cross Filter *********
# # X축 Filter
# roberts_x = np.array()
# RobertsX = cv.filter2D(img, -1, roberts_x)
#
# # Y축 Filter
# roberts_y = np.array()
# RobertsY = cv.filter2D(img, -1, roberts_y)
#
# cv.imshow('robertsX', RobertsX)
# cv.imshow('robertsY', RobertsY)
# cv.imshow('Roberts', RobertsX + RobertsY)
#
#
# # ************ Prewitt Filter *********
# # X축 Filter
# prewitt_x = np.array()
# PrewittX = cv.filter2D(img, -1, prewitt_x)
#
# # Y축 Filter
# prewitt_y = np.array()
# PrewittY = cv.filter2D(img, -1, prewitt_y)
#
# cv.imshow('prewittX', PrewittX)
# cv.imshow('prewittY', PrewittY)
# cv.imshow('Prewitt', PrewittX + PrewittY)
#
#
# # ************ Scharr Filter *********
# # X축 Filter
# scharr_x = np.array()
# ScharrX = cv.filter2D(img, -1, scharr_x)
#
# # Y축 Filter
# scharr_y = np.array()
# ScharrY = cv.filter2D(img, -1, scharr_y)
#
# cv.imshow('scharrX', ScharrX)
# cv.imshow('scharrY', ScharrY)
# cv.imshow('Scharr', ScharrX + ScharrY)
#
#
# # ************ Laplacian Filter ***********
# # X축 Filter
# laplacian = np.array()
# Laplacian = cv.filter2D(img, -1, laplacian)
#
# cv.imshow('Laplacian', Laplacian)
#
#
# cv.imshow('Original', img)
cv.waitKey(0)
cv.destroyAllWindows()