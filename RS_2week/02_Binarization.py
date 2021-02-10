import cv2 as cv

Path = 'Data/'
Name = 'adaptive_thresholdiry.jpg'
Name = 'lenna.tif'
src = Path + Name

img = cv.imread(src, cv.IMREAD_GRAYSCALE)
cv.imshow('Original Image', img)


# =============== 전역 이진화 (고정 임계 값) ===============
# binimg = img.copy()
# threshold = 127
# # **** 실습 2) for문을 이용하여 threshold 값보다 작은 픽셀 값은 0, 큰 픽셀 값은 255를 할당하여 영상 이진화하기 ****
# for y in range(binimg.shape[0]):
#     for x in range(binimg.shape[1]):
#         if binimg[y][x] < threshold:
#             binimg[y][x] = 0
#         else:
#             binimg[y][x] = 255
#
# cv.imshow('binarization 1', binimg)
# cv.waitKey()
# exit()
# ==================================================


# ==== 전역 이진화 (고정 임계 값, 조건문을 이용한 인덱싱) ====
# binimg = img.copy()
# threshold = 127
#
# binimg[binimg <= threshold] = 0
# binimg[binimg > threshold] = 255
#
# cv.imshow('binarization 2', binimg)
# cv.waitKey()
# exit()
# ==================================================


# =========== 전역 이진화 (고정 임계값, 함수 사용) ===========
# cv2.threshold(src, thresh, maxval, type)
#   src : 입력 영상
#   thresh : 임계값
#   maxval : 임계값보다 클 때 적용되는 최대값
#   type : 임계값 적용 방법
#     cv2.THRESH_BINARY : 픽셀 값이 thresh보다 크면 maxval, 작으면 0으로 할당
# ==================================================
# bin_img = img.copy()
#
# _, bin_img = cv.threshold(bin_img, 100, 255, cv.THRESH_BINARY)
#
# cv.imshow('binarization 3', bin_img)
# cv.waitKey()
# ==================================================


# =========== 지역 이진화 (가변 임계값, 함수 사용) ===========
# cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
#   src : 입력 영상
#   maxValue : 임계값보다 클 때 적용되는 최대값
#   adaptiveMethod : 적용할 지역 이진화 알고리즘
#     cv2.ADAPTIVE_THRESH_MEAN_C : 주변 영역의 평균값으로 결정
#   thresholdType : 임계값 적용 방법
#     cv2.THRESH_BINARY : 픽셀 값이 thresh보다 크면 maxval, 작으면 0으로 할당
#   blockSize : 지역 이진화를 할 때 고려할 주변 픽셀 크기 (블록 크기)
#   C : 블록 내 평균값에서 뺄 값 (임계값을 결정하는 파라미터)
# ==================================================
# bin_img = img.copy()
#
# bin_img = cv.adaptiveThreshold(bin_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,  15, 2)
#
# cv.imshow('binarization 4', bin_img)
# cv.waitKey()
# ==================================================