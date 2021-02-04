import cv2 as cv
import numpy as np

Path = 'Data/'
Name1 = 'lenna.tif'
Name2 = '4color_100.png'
LennaName = Path + Name1
FourColorName = Path + Name2

# ==================== 이미지 읽기 ====================
img = cv.imread(LennaName)
img_4color = cv.imread(FourColorName)

print("컬러 이미지 ", img.shape)
print("----------------------------------------")


# ==================== ROI 설정 ====================

# ROI = np.zeros(img.shape, np.uint8)     # gray 이미지 틀에 0으로 초기화된 2중 리스트
# for y in range(img.shape[0]):
#     for x in range(img.shape[1]):
#         if y < 300:                     # ******* 1) y 가 300 미만일 때 ***********:
#             ROI[y][x] = img[y][x]
#         else:
#             ROI[y][x] = 0
#
# cv.imshow("origin", img)
# cv.imshow("ROI", ROI)


# ==================== resize =========================
#  cv.resize(img, dsize, fx, fy, interpolation)
#   img:            이미지
#   dsize:          출력 사이즈 (가로,세로) 튜플 형태
#  *fx:             가로 사이즈 배수   ex) fx=2
#  *fy:             세로 사이즈 배수   ex) fy=4
#  *interpolation:  보간법
#
# resize1 = cv.resize(img, (150, 150))
# resize2 = cv.resize(img, (0, 0), fx=0.5, fy=0.6)
# #resize2 = cv.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.6)))
# #resize2 = ******** 2) 가로 0.5배, 세로 0.6배로 리사이즈 하기 hint: dsize (0, 0)으로 초기화 해야함)************
#
# cv.imshow("origin", img)
# cv.imshow("resize1", resize1)
# cv.imshow("resize2", resize2)


# ================== 이미지 합치기 =========================
# img는 512x512
# img_4color는 100x100
# ******* 3) 두 이미지의 좌측 상단 모서리가 닿도록 이미지 합치기 ***********:
# result = np.zeros(img.shape, np.uint8)
# for y in range(img.shape[0]):
#     for x in range(img.shape[1]):
#         result[y][x] = img[y][x]
#         if x < 100 and y < 100:
#             result[y][x] = img_4color[y][x]

result = np.copy(img)
for y in range(img_4color.shape[0]):
    for x in range(img_4color.shape[1]):
        result[y][x] = img_4color[y][x]

cv.imshow("result", result)
cv.waitKey()