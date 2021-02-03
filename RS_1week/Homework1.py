import cv2 as cv
import numpy as np

# 마우스 콜백 함수(마우스 이벤트 처리 함수)
def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:   # 마우스 왼쪽 버튼 클릭
        B, G, R = cv.split(param)
        print(f'B: {B}\nC: {G}\nR: {R}')
        print('B: ', param[y][x][0], '\nG: ', param[y][x][1], '\nR: ', param[y][x][2])
        print('=================================')



# 이미지 경로 설정
Path = f'./Data/'
Name1 = f'rabong.jpg'
Name2 = f'rabong2.jpg'
FullName1 = Path + Name1
FullName2 = Path + Name2

# 이미지 읽기
img1 = cv.imread(FullName1)
img2 = cv.imread(FullName2)

# 이미지 출력
cv.imshow(f'Result1', img1)
cv.imshow(f'Result2', img2)

# cv2.waitKey(time)
# time(ms)시간 동안 대기, 0은 무한대기
# 키 입력 대기, 대기하는 동안 return 값은 -1
while cv.waitKey(33) <= 0:  # 키가 입력되지 않은 동안
    cv.setMouseCallback('result', mouse_callback, img2)  # imshow에 쓴 winName이랑 같은 창에 마우스이벤트 적용

cv.waitKey(0)

# 한라봉의 위치(사분면)와 이미지 크기(height, width, channel) 출력
