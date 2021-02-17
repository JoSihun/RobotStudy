import cv2
import numpy as np

Path = './Data/'
Name = 'homework.jpg'
FullName = Path + Name
src = cv2.imread(FullName)     # image size : (429, 697)

# 1) 주어진 이미지 Gray Scale 변환
img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# 2) 이미지 Blurring 처리
gauss_filter = cv2.getGaussianKernel(5, 3)      # Blur 처리방법1: 커널 구하기
img1 = cv2.filter2D(img, -1, gauss_filter)      # Blur 처리방법1: 공간필터링 적용
img2 = cv2.GaussianBlur(img, (5, 5), 3)         # Blur 처리방법2: 내장함수로 한번에 적용

cv2.imshow('Gaussian Filter', img1)             # 공간필터를 이용한 Blur 결과
cv2.imshow('Gaussian Blur', img2)               # 내장함수를 이용한 Blur 결과

# 3) Canny Edge로 Edge만 검출
img1 = cv2.Canny(img1, 50, 200)                 # minVal: 50 | maxVal: 200
img2 = cv2.Canny(img2, 50, 200)                 # minVal: 50 | maxVal: 200

# 4) ROI로 차선 부분만 추출: 직사각형이 아닌 사다리꼴 등의 다각형으로 ROI 할 때 사용
def region_of_interest(src, vertices, color=(255, 255, 255)):
    if len(src.shape) < 3:                  # 1 Channel = Gray Scale:
        color = 255                         # Gray Scale Color Default 흰색 설정
    mask = np.zeros_like(src)               # src와 같은 크기의 빈 이미지
    cv2.fillPoly(mask, vertices, color)     # vertices 좌표로 구성된 다각형 범위내부를 color로 채움
    dst = cv2.bitwise_and(src, mask)        # src & ROI 이미지 합침
    return dst

# 5) Vertices Point Setting:
# np.array([[Top Left], [Top Right], [Bottom Right], [Bottom Left]])
# 수평을 기준으로 아래쪽 절반 선택
height, width = img.shape[:2]
point = np.array([[0, height // 2], [width, height // 2], [width, height], [0, height]])
roi1 = region_of_interest(img1, [point])
roi2 = region_of_interest(img2, [point])

cv2.imshow('ROI1', roi1)
cv2.imshow('ROI2', roi2)
cv2.waitKey()
cv2.destroyAllWindows()