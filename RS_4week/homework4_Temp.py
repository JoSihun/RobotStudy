# Python OpenCV 강좌 : 제 28강 - 직선 검출
# https://076923.github.io/posts/Python-opencv-28/

# 표준 허프 변환(Standard Hough Transform) & 멀티 스케일 허프 변환(Multi-Scale Hough Transform)
# 점진성 확률적 허프 변환(Progressive Probabilistic Hough Transform)
# cv2.HoughLines(img, rho, theta, threshold, lines, srn=0, stn=0, min_theta, max_theta)
# cv2.HoughLinesP(img, rho, theta, threshold, lines, srn=0, stn=0, min_theta, max_theta)
#       img: 입력 이미지, 1 채널 바이너리 스케일
#       rho: 거리 측정 해상도, 0~1
#       theta: 각도, 라디안 단위 (np.pi/0~180)
#       threshold: 직선으로 판단할 최소한의 동일 개수 (작은 값: 정확도 감소, 검출 개수 증가 / 큰 값: 정확도 증가, 검출 개수 감소)
#       lines: 검출 결과, N x 1 x 2 배열 (r, Θ)
#       srn, stn: 멀티 스케일 허프 변환에 사용, 선 검출에서는 사용 안 함
#       min_theta, max_theta: 검출을 위해 사용할 최대, 최소 각도
########################################################################################################################
import cv2
import numpy as np

########################################################################################################################
# Define Data Path
Path = './Data/'
Name = 'drive.mp4'
FileName = Path + Name

########################################################################################################################
# Define Color
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

########################################################################################################################
# Define Image Processing Method
# ROI: 다각형 마스킹, 사다리꼴 등의 다각형으로 마스킹 할 때 사용
def region_of_interest(src, vertices, color=BLACK):
    if len(src.shape) < 3:                  # 1 Channel = Gray Scale:
        color = 255                         # Gray Scale Color Default 흰색 설정
    mask = np.zeros_like(src)               # src 와 같은 크기의 빈 이미지
    cv2.fillPoly(mask, vertices, color)     # vertices 좌표로 구성된 다각형 범위내부를 color로 채움
    dst = cv2.bitwise_and(src, mask)        # src & ROI 이미지 합침
    return dst

########################################################################################################################
# Main Routine
Nframe = 0                                                  # Frame 수
scale = 1500                                                # Scale for Multi-Scale Hough Transform
capture = cv2.VideoCapture(FileName)                        # VideoCapture
while True:
    ret, frame = capture.read()                             # Video Load 성공하면 True, Frame 반환
    if ret == True:                                         # Video Load 성공했다면
        Nframe += 1                                         # Increase Frame Count
        frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)   # Frame Resize 0.4
        dst = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)       # BGR2GRAY 변환
        dst = cv2.GaussianBlur(dst, (5, 5), 3)              # Gaussian Blurring, 내장함수로 한번에 적용
        dst = cv2.Canny(frame, 50, 200, None, 3)            # Canny Edge 검출, 3Channel 도 GrayScale 로 반환
        dst1 = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)        # 결과비교 출력을 위한 GRAY2BGR 변환, hstack/vstack 사용을 위함
        dst2 = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)        # 결과비교 출력을 위한 GRAY2BGR 변환, hstack/vstack 사용을 위함

        # Vertices Point Setting, 수평을 기준으로 아래쪽 절반 선택
        # np.array([[Top Left], [Top Right], [Bottom Right], [Bottom Left]])
        height, width = dst.shape[:2]; middle = height // 2
        point = np.array([[0, middle], [width, middle], [width, height], [0, height]])
        roi = region_of_interest(dst, [point])

        # 표준 허프 변환(Standard Hough Transform) & 멀티 스케일 허프 변환(Multi-Scale Hough Transform)
        lines = cv2.HoughLines(roi, 1, np.pi / 180, 150, None, 0, 0)        # 표준 허프 변환
        if lines is not None:                                               # 직선이 검출 되었다면
            for line in lines:                                              # 각 직선에 대해
                rho, theta = line[0]                                        # 극좌표
                a, b = np.cos(theta), np.sin(theta)                         # xy좌표 변환 과정
                x0, y0 = a * rho, b * rho                                   # xy좌표 변환 결과
                x1, y1 = int(x0 + scale * (-b)), int(y0 + scale * a)        # 직선의 스케일 평행이동을 통한 시작점 좌표계산
                x2, y2 = int(x0 - scale * (-b)), int(y0 - scale * a)        # 직선의 스케일 평행이동을 통한 종료점 좌표계산
                cv2.line(frame, (x1, y1), (x2, y2), GREEN, 3, cv2.LINE_AA)  # 결과이미지에 선그리기
                cv2.line(dst1, (x1, y1), (x2, y2), GREEN, 3, cv2.LINE_AA)   # 결과이미지에 선그리기

        # 점진성 확률적 허프 변환(Progressive Probabilistic Hough Transform)
        linesP = cv2.HoughLinesP(roi, 1, np.pi / 180, 50, None, 50, 10)     # 확률적 허프 변환
        if linesP is not None:                                              # 직선이 검출 되었다면
            for line in linesP:                                             # 각 직선에 대해
                points = line[0]                                            # 직선의 좌표모음
                x1, y1 = points[0], points[1]                               # 직선의 시작점 좌표
                x2, y2 = points[2], points[3]                               # 직선의 종료점 좌표
                cv2.line(frame, (x1, y1), (x2, y2), RED, 3, cv2.LINE_AA)    # 결과이미지에 선그리기
                cv2.line(dst2, (x1, y1), (x2, y2), RED, 3, cv2.LINE_AA)     # 결과이미지에 선그리기

        # 결과 프레임 별 텍스트 추가
        text1 = 'Source Video'
        text2 = 'ROI Video'
        text3 = 'Standard Hough Line Transform'
        text4 = 'Probabilistic Line Transform'
        cv2.putText(frame, text1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK, 2, cv2.LINE_AA)
        cv2.putText(roi, text2, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2, cv2.LINE_AA)
        cv2.putText(dst1, text3, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2, cv2.LINE_AA)
        cv2.putText(dst2, text4, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2, cv2.LINE_AA)

        # 결과출력
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)     # 결과비교 출력을 위한 GRAY2BGR 변환, hstack/vstack 사용을 위함
        merged1 = np.hstack((frame, roi))               # 수평합병, Source Video + ROI Video
        merged2 = np.hstack((dst1, dst2))               # 수평합병, Standard + Probabilistic
        merged = np.vstack((merged1, merged2))          # 수직합병
        cv2.imshow('Hough Convert', merged)             # 결과출력

    key = cv2.waitKey(1)                    # 키보드 입력대기
    if key == 27 or key == ord('q'):        # ESC, q 를 입력하면
        break                               # 종료

print(f'Number of Frame: {Nframe}')         # 영상의 frame 수 출력
capture.release()                           # Video Release
cv2.destroyAllWindows()                     # Destroy All Windows