import cv2 as cv
import numpy as np

Path = 'Data/'
Name = 'lenna.tif'
src = Path + Name

color_img = cv.imread(src)
img = cv.imread(src, cv.IMREAD_GRAYSCALE)

cv.imshow('Original', img)


# =============== 산술 연산 1 ===============
# arith_img = img.copy()
# # **** 실습 1) (복습) for문을 이용하여 덧셈 산술연산을 구현하기 (+100) ****
# for y in range(arith_img.shape[0]):
#     for x in range(arith_img.shape[1]):
#         arith_img[y][x] += 100
#         if img[y][x] + 100 > 255:
#             arith_img[y][x] = 255
# cv.imshow('Arithmetic Operation 1', arith_img)
# cv.waitKey()
# exit()
# ==================================================


# =============== 산술 연산 2 ===============
# arith_img2 = img.copy()
#
# arith_img2 = cv.add(arith_img2, 100)
#
# cv.imshow('Arithmetic Operation 2', arith_img2)
#
# cv.waitKey()
# exit()
# ==================================================


# # =================== 논리 연산 ===================
# img = img / 255
# mask = np.zeros(img.shape, np.float64)
# start_pos = 150
# end_pos = 400
#
# for y in range(start_pos, end_pos):
#     for x in range(start_pos, end_pos):
#         mask[y][x] = 1
#
# # 슬라이싱 기법 사용
# # mask[start_pos:end_pos, start_pos:end_pos] = 1
#
# logic_img = img * mask
#
# cv.imshow('mask', mask)
# cv.imshow('Logical Operation', logic_img)
#
# cv.waitKey()
# exit()
# ==================================================


# =================== 그레이스케일 ===================
# img = color_img.copy()
# img = img[:, :, 2] / 3 + img[:, :, 1] / 3 + img[:, :, 0] / 3
#
# cv.imshow('Original', color_img)
# cv.imshow('grayscale', img.astype('uint8'))
#
# cv.waitKey()
# exit()
# ==================================================

