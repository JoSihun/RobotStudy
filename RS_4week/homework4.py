import cv2
import numpy as np

# =============================== 데이터 불러오기 ====================================#
Path = './Data/'
Name = 'drive.mp4'
FileName = Path + Name

# ================================== 영상 처리 =====================================#
# 4) ROI로 차선 부분만 추출: 직사각형이 아닌 사다리꼴 등의 다각형으로 ROI 할 때 사용
def region_of_interest(src, vertices, color=(255, 255, 255)):
    if len(src.shape) < 3:                  # 1 Channel = Gray Scale:
        color = 255                         # Gray Scale Color Default 흰색 설정
    mask = np.zeros_like(src)               # src와 같은 크기의 빈 이미지
    cv2.fillPoly(mask, vertices, color)     # vertices 좌표로 구성된 다각형 범위내부를 color로 채움
    dst = cv2.bitwise_and(src, mask)        # src & ROI 이미지 합침
    return dst
# ================================== 메인 루틴 =====================================#

# lines = cv2.HoughLines(img, rho, theta, threshold, lines, srn=0, stn=0, min_theta, max_theta)
# img: 입력 이미지, 1 채널 바이너리 스케일
# rho: 거리 측정 해상도, 0~1
# theta: 각도, 라디안 단위 (np.pi/0~180)
# threshold: 직선으로 판단할 최소한의 동일 개수 (작은 값: 정확도 감소, 검출 개수 증가 / 큰 값: 정확도 증가, 검출 개수 감소)
# lines: 검출 결과, N x 1 x 2 배열 (r, Θ)
# srn, stn: 멀티 스케일 허프 변환에 사용, 선 검출에서는 사용 안 함
# min_theta, max_theta: 검출을 위해 사용할 최대, 최소 각도

# Nframe = 0                                          # frame 수
# cap = cv2.VideoCapture(FileName)                    # VideoCapture
# while cap.isOpened():                               # 1회 재생
#     ret, frame = cap.read()                         # read() 성공여부, 프레임
#     frame = cv2.resize(frame, (1000, 562))          # cap.isOpened()로 이미 체크했으므로 if ret 생략
#
#     Nframe += 1
#     dst = np.copy(frame)
#     #dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
#     #dst = cv2.GaussianBlur(dst, (5, 5), 3)          # Blur 처리방법2: 내장함수로 한번에 적용
#     dst = cv2.Canny(dst, 50, 200)
#     height, width = dst.shape[:2]
#     lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
#     dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
#     for line in lines:
#         rho, theta = line[0]
#         a, b = np.cos(theta), np.sin(theta)
#         x0, y0 = a * rho, b * rho
#         x1, y1 = int(x0 + width * (-b)), int(y0 + height * a)
#         x2, y2 = int(x0 - width * (-b)), int(y0 - height * a)
#         cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
#
#     #dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
#     merged = np.hstack((frame, dst))
#     cv2.imshow('Hough Convert', merged)
#
#     if cv2.waitKey(1) & 0xff == ord('q'):  # 'q'누르면 영상 종료
#         break
#
# print("Number of Frame: ", Nframe)  # 영상의 frame 수 출력
#
# cap.release()
# cv2.destroyAllWindows()
#######################################################################################################################
import cv2
import numpy as np

cap = cv2.VideoCapture(FileName)                    # VideoCapture
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1000, 562))          # cap.isOpened()로 이미 체크했으므로 if ret 생략
    dst = cv2.Canny(frame, 50, 200, None, 3)        # return GrayScale
    dst1 = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)    # GRAY2BGR 변환
    dst2 = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)    # GRAY2BGR 변환

    # Vertices Point Setting:
    # 수평을 기준으로 아래쪽 절반 선택
    # np.array([[Top Left], [Top Right], [Bottom Right], [Bottom Left]])
    height, width = dst.shape[:2]; middle = height // 2
    point = np.array([[0, middle], [width, middle], [width, height], [0, height]])
    dst = region_of_interest(dst, [point])

    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)    # 허프변환
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * a)
            x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * a)
            cv2.line(dst1, (x1, y1), (x2, y2), (0, 255, 0), 3, cv2.LINE_AA)

    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    if linesP is not None:
        for line in linesP:
            points = line[0]
            x1, y1 = points[0], points[1]
            x2, y2 = points[2], points[3]
            cv2.line(dst2, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("Source", frame)
    cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", dst1)
    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", dst2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
