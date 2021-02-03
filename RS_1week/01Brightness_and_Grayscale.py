import cv2 as cv
import numpy as np

Path = 'Data/'
Name = 'lenna.tif'
FullName = Path + Name

# ==================== 이미지 읽기 ====================
img = cv.imread(FullName, 1)   # BGR 컬러로 읽기
gray1 = cv.imread(FullName, 0)  # 흑백으로 읽기

print("그레이 이미지", gray1.shape)
print("컬러 이미지 ", img.shape)
print("----------------------------------------")


# ==================== 밝기 조절1 (픽셀 단위: for문 이용) ====================
# 그레이 사진으로 진행
# 왜냐하면 1채널 그레이 이미지는 픽셀값이 밝기를 나타내기 때문
# 3채널 컬러 이미지는 픽셀값이 각 채널의 색상 값을 나타냄
# modulo 연산 / numpy 연산

# bright = np.zeros(gray1.shape)      # gray 이미지 틀에 0으로 초기화된 2중 리스트
# for y in range(gray1.shape[0]):
#     for x in range(gray1.shape[1]):
#         #******** 1) 그레이 이미지에 밝기 +50한 것을 bright에 넣기 ********#

#
#
# dark = np.zeros(gray1.shape)
# for y in range(gray1.shape[0]):
#     for x in range(gray1.shape[1]):
#         #******** 2) 그레이 이미지에 밝기 -50한 것을 dark에 넣기 ********#

#
#
# print("*************** gray ***************\n원소 데이터 타입:", type(gray1[0][0]), "\n", gray1)
# print("\n************** bright **************\n원소 데이터 타입:", type(bright[0][0]), "\n", bright)
# print("\n*************** dark ***************\n원소 데이터 타입:", type(dark[0][0]), "\n", dark)


# ==================== 밝기 조절2 (이미지 단위) ====================
# saturation 연산 / openCV 사용

# bright = cv.add(gray1, 100)
# dark = cv.add(gray1, -100)
#
#
# cv.imshow('gray', gray1)
# cv.imshow('bright', bright)
# cv.imshow('dark', dark)



# ========================= Gray Scale =========================
# 공식 사용
# 1. gray = (R + G + B) / 3
# 2. gray = R * 0.2126 + G * 0.7152 + B * 0.0722

# gray2 = # ************* 3) 1번 공식 사용하여 회색조 이미지 만들기 ***********
# gray3 = img[:, :, 2] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 0] * 0.0722
#
# cv.imshow("color", img)
# cv.imshow("gray2", gray2 / 255)
# cv.imshow("gray3", gray3 / 255)
#
# # #OpenCV 함수 사용
# gray4 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow("gray4", gray4)

cv.waitKey()