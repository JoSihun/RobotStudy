# 로버츠, 프리윗, 소벨 필터 적용
import cv2
import numpy as np

PATH = './Data/'
NAME = 'lenna.tif'
NAME = 'test.jpg'
FILENAME = PATH + NAME

src = cv2.imread(FILENAME, cv2.IMREAD_GRAYSCALE)

#######################################################################################################################
# Roberts Cross Filter
# 기본 미분 필터를 개선, 대각선 방향으로 +-1을 배치
# 사선 경계 검출 효과, 노이즈에 민감

# 로버츠 크로스 커널 생성
roberts_kernelX = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])
roberts_kernelY = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])
roberts_kernelXY = roberts_kernelX + roberts_kernelY

# 로버츠 크로스 커널 필터 적용
robertsX = cv2.filter2D(src, -1, roberts_kernelX)
robertsY = cv2.filter2D(src, -1, roberts_kernelY)
robertsXY = cv2.filter2D(src, -1, roberts_kernelXY)

# 결과 출력
dst1 = np.hstack((src, robertsX, robertsY, robertsX + robertsY))
dst2 = np.hstack((src, robertsX, robertsY, robertsXY))
dst = np.vstack((dst1, dst2))
cv2.imshow('Roberts Cross Filter', dst)

#######################################################################################################################
# Prewitt Filter
# x, y축 각 방향으로 차분을 3번 계산
# 상하좌우 경계 검출 효과, 대각선 검출 약함

# 프리윗 커널 생성
prewitt_kernelX = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_kernelY = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
prewitt_kernelXY = prewitt_kernelX + prewitt_kernelY

# 프리윗 커널 필터 적용
prewittX = cv2.filter2D(src, -1, prewitt_kernelX)
prewittY = cv2.filter2D(src, -1, prewitt_kernelY)
prewittXY = cv2.filter2D(src, -1, prewitt_kernelXY)

# 결과 출력
dst1 = np.hstack((src, prewittX, prewittY, prewittX + prewittY))
dst2 = np.hstack((src, prewittX, prewittY, prewittXY))
dst = np.vstack((dst1, dst2))
cv2.imshow('Prewitt Filter', dst)

#######################################################################################################################
# Sobel Filter1
# 중심 픽셀의 차분 비중을 2배로 줌
# x, y축 대각선 방향의 경계 검출에 모두 강함

# 소벨 커널 생성 (직접 생성 방식)
sobel_kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
sobel_kernelXY = sobel_kernelX + sobel_kernelY

# 소벨 커널 필터 적용
sobelX = cv2.filter2D(src, -1, sobel_kernelX)
sobelY = cv2.filter2D(src, -1, sobel_kernelY)
sobelXY = cv2.filter2D(src, -1, sobel_kernelXY)

# 결과 출력
dst1 = np.hstack((src, sobelX, sobelY, sobelX + sobelY))
dst2 = np.hstack((src, sobelX, sobelY, sobelXY))
dst = np.vstack((dst1, dst2))
cv2.imshow('Sobel Filter1', dst)

#######################################################################################################################
# Sobel Filter2
# 로버츠와 프리윗은 현재 잘 쓰이지 않음
# 소벨은 실무적으로도 쓰이므로 OpenCV에서 별도의 함수를 제공함

# 소벨 커널 필터 적용2 (OpenCV 내장 함수 사용)
sobelX = cv2.Sobel(src, -1, 1, 0, ksize=3)
sobelY = cv2.Sobel(src, -1, 0, 1, ksize=3)
sobelXY = sobelX + sobelY

# 결과 출력
dst = np.hstack((src, sobelX, sobelY, sobelXY))
cv2.imshow('Sobel Filter2', dst)

cv2.waitKey()
cv2.destroyAllWindows()