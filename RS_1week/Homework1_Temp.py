import cv2
import numpy as np

# 마우스 콜백 함수(마우스 이벤트 처리 함수)
def mouse_callback(event, x, y, flags, param):
    B, G, R = cv2.split(dst)                                # B, G, R 채널 분리
    if event == cv2.EVENT_LBUTTONDOWN:                      # 마우스 왼쪽 버튼 클릭
        print(f'B: {B[y][x]}\nG: {G[y][x]}\nR: {R[y][x]}')  # 클릭 좌표 픽셀의 BGR값 출력
        print('=========================================')

# 객체 검출 함수(메소드)
def object_detect(src, color):
    B, G, R = cv2.split(src)                                # B, G, R 채널 분리
    count = 0; quadrant = 0                                 # 객체에 해당하는 모든픽셀의 개수, 객체가 위치하는 사분면
    quadrant_row = 0                                        # 객체에 해당하는 모든픽셀의 height 합
    quadrant_col = 0                                        # 객체에 해당하는 모든픽셀의 width 합

    # 객체 검출 및 사분면 계산을 위한 변수 계산
    rows, cols = src.shape[:2]                              # 이미지 height, width
    for row in range(rows):
        for col in range(cols):
            if G[row][col] > 100 and R[row][col] > 150:     # 노란색인 픽셀이면
                dst[row, col, ...] = color                  # 해당 픽셀을 지정한 컬러로 변경
                quadrant_row += row                         # 해당 픽셀의 height 합산
                quadrant_col += col                         # 해당 픽셀의 width 합산
                count += 1                                  # 해당 픽셀을 count

    # 사분면 계산
    if count > 0:                                                                   # 객체가 검출 되었다면
        quadrant_row /= count                                                       # 객체가 검출된 모든픽셀의 height 합의 평균
        quadrant_col /= count                                                       # 객체가 검출된 모든픽셀의 width 합의 평균
        if quadrant_row < rows // 2 and quadrant_col > cols // 2: quadrant = 1      # 제 1사분면
        elif quadrant_row < rows // 2 and quadrant_col < cols // 2: quadrant = 2    # 제 2사분면
        elif quadrant_row > rows // 2 and quadrant_col < cols // 2: quadrant = 3    # 제 3사분면
        elif quadrant_row > rows // 2 and quadrant_col > cols // 2: quadrant = 4    # 제 4사분면, 가독성을 위해 else 미사용

    dst[rows//2, :, :] = COLOR_BLUE                         # 가로 파란 선
    dst[:, cols//2, :] = COLOR_BLUE                         # 세로 파란 선
    print(f'위치: 제{quadrant}사분면')                        # 객체가 위치하는 사분면 출력
    print(f'크기: {src.shape}')                              # 이미지 크기 출력
    return dst

# 이미지 경로 설정
Path = f'./Data/'
Name1 = f'rabong.jpg'
Name2 = f'rabong2.jpg'
FullName = Path + Name1

# 이미지 읽기
src = cv2.imread(FullName)      # 소스 이미지
dst = np.copy(src)              # 결과 이미지, 최초에는 소스 이미지 사본으로 사용

COLOR_RED = (0, 0, 255)         # RED Color 정의
COLOR_BLUE = (255, 0, 0)        # BLUE Color 정의

cv2.namedWindow('Result')                       # 출력 윈도우 Title 정의
cv2.setMouseCallback('Result', mouse_callback)  # 출력 윈도우 Title에 마우스 콜백함수 호출
while True:
    cv2.imshow('Result', dst)                   # Title이 Result인 윈도우에 매 프레임마다 영상(이미지) 출력
    dst = object_detect(src, COLOR_RED)         # 객체 검출, 검출한 객체는 RED Color로 칠하기

    # cv2.waitKey(time): time(ms)시간 동안 키 입력 대기, 0은 무한대기, 대기하는 동안 return 값은 -1
    key = cv2.waitKey(1)                        # 1ms 동안 입력 대기 = 매 1ms 마다 이미지 갱신
    if key == 27 or key == ord('q'): break      # ESC 또는 q를 입력하면 while loop 탈출

