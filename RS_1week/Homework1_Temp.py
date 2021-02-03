# 한라봉의 위치(사분면)와 이미지 크기(height, width, channel) 출력
#======================================================================================================================
import cv2
import numpy as np

# 마우스 콜백 함수(마우스 이벤트 처리 함수)
def mouse_callback(event, x, y, flags, param):
    B, G, R = cv2.split(dst)
    if event == cv2.EVENT_LBUTTONDOWN:                          # 마우스 왼쪽 버튼 클릭
        print(f'B: {B[y][x]}\nG: {G[y][x]}\nR: {R[y][x]}')
        print('=================================')

# 이미지 경로 설정
Path = f'./Data/'
Name1 = f'rabong.jpg'
Name2 = f'rabong2.jpg'
FullName = Path + Name1

# 이미지 읽기
src = cv2.imread(FullName)
dst = np.copy(src)
# cv22.waitKey(time)
# time(ms)시간 동안 대기, 0은 무한대기
# 키 입력 대기, 대기하는 동안 return 값은 -1
cv2.namedWindow('Result')
cv2.setMouseCallback('Result', mouse_callback)  # imshow에 쓴 winName이랑 같은 창에 마우스이벤트 적용

def object_detect(src, color):
    count = 0; quadrant = 0
    quadrant_row = 0
    quadrant_col = 0
    B, G, R = cv2.split(dst)
    rows, cols = src.shape[:2]
    for row in range(rows):
        for col in range(cols):
            if G[row][col] > 100 and R[row][col] > 150:
                dst[row, col, ...] = color
                quadrant_row += row
                quadrant_col += col
                count += 1
    if count > 0:
        quadrant_row /= count
        quadrant_col /= count
        if quadrant_row < rows // 2 and quadrant_col > cols // 2: quadrant = 1
        elif quadrant_row < rows // 2 and quadrant_col < cols // 2: quadrant = 2
        elif quadrant_row > rows // 2 and quadrant_col < cols // 2: quadrant = 3
        else: quadrant = 4

    dst[rows//2, :, :] = COLOR_BLUE
    dst[:, cols//2, :] = COLOR_BLUE
    return dst, quadrant

COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
dst, quadrant = object_detect(src, (0, 0, 255))
print(f'위치: 제{quadrant}사분면')
print(f'크기: {src.shape}')
while True:
    cv2.imshow('Result', dst)        # 매 프레임마다 영상(이미지) 출력
    dst, quadrant = object_detect(src, COLOR_RED)

    key = cv2.waitKey(1)                         # 1ms 동안 입력 대기
    if key == 27 or key == ord('q'):            # ESC 또는 q를 입력하면
        break                                   # while loop 탈출