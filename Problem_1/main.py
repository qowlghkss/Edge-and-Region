"""
=============================================================================
Problem 1: 소벨(Sobel) 에지 검출 및 결과 시각화
=============================================================================
이 코드는 OpenCV와 Matplotlib을 사용하여 이미지에서 소벨 에지를 검출하고,
원본 이미지와 에지 강도 이미지를 나란히 시각화합니다.

담
과목: 컴퓨터비전 - L03 Edge and Region
=============================================================================
"""

# OpenCV 라이브러리를 임포트: 이미지 처리의 핵심 라이브러리
import cv2 as cv

# NumPy 라이브러리를 임포트: 수치 연산 및 배열 처리에 사용
import numpy as np

# Matplotlib의 pyplot 모듈 임포트: 이미지 시각화에 사용
import matplotlib.pyplot as plt

# os 모듈 임포트: 파일 경로 처리에 사용
import os

# ─────────────────────────────────────────────────────────────────────────────
# 한글 폰트 설정 (matplotlib에서 한글 제목/라벨이 깨지지 않도록)
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib
# 시스템에 설치된 한글 폰트 목록 확인 후 사용 가능한 폰트 자동 선택
from matplotlib import font_manager

# 한글 지원 폰트 후보 목록 (설치된 경우 우선순위 순으로 시도)
_korean_fonts = ['NanumGothic', 'NanumBarunGothic', 'Malgun Gothic', 'AppleGothic', 'UnDotum']
# 시스템에 설치된 폰트 이름 목록 가져오기
_available_fonts = {f.name for f in font_manager.fontManager.ttflist}
# 후보 중 설치된 첫 번째 한글 폰트를 matplotlib 기본 폰트로 설정
for _font in _korean_fonts:
    if _font in _available_fonts:
        matplotlib.rcParams['font.family'] = _font  # 기본 폰트 설정
        break
# 마이너스 기호가 깨지는 문제 방지
matplotlib.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────────────────────────────────────
# 1단계: 이미지 불러오기
# ─────────────────────────────────────────────────────────────────────────────

# 현재 스크립트 파일이 위치한 디렉토리를 기준으로 이미지 경로를 동적으로 설정
# -> 어떤 환경에서 실행하더라도 경로 오류 방지
script_dir = os.path.dirname(os.path.abspath(__file__))

# edgeDetectionImage.jpg 파일의 절대 경로 생성
image_path = os.path.join(script_dir, "images", "edgeDetectionImage.jpg")

# cv.imread(): 지정된 경로의 이미지를 BGR 형식으로 읽어 NumPy 배열로 반환
# BGR = Blue, Green, Red 채널 순서 (OpenCV의 기본 채널 순서)
img_bgr = cv.imread(image_path)

# 이미지가 정상적으로 불러와졌는지 확인
# imread() 실패 시 None을 반환하므로 예외 처리 필요
if img_bgr is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

# 터미널에 이미지 불러오기 성공 메시지 출력
print(f"[완료] 이미지 불러오기 성공: {image_path}")
# 이미지의 크기(높이, 너비, 채널 수) 출력
print(f"[정보] 이미지 크기: {img_bgr.shape[1]}x{img_bgr.shape[0]} (너비x높이), 채널: {img_bgr.shape[2]}")

# ─────────────────────────────────────────────────────────────────────────────
# 2단계: 그레이스케일 변환
# ─────────────────────────────────────────────────────────────────────────────

# cv.cvtColor(): 이미지의 색상 공간을 변환하는 함수
# cv.COLOR_BGR2GRAY: BGR 3채널 컬러 이미지를 1채널 그레이스케일로 변환
# 소벨 필터는 단일 채널(그레이스케일) 이미지에 적용하기 때문에 변환이 필요
img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

# 그레이스케일 변환 완료 메시지 출력
print(f"[완료] 그레이스케일 변환 완료: 이미지 채널 수 -> {len(img_gray.shape)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3단계: 소벨 필터를 이용한 에지 검출 (X축 방향과 Y축 방향 각각 적용)
# ─────────────────────────────────────────────────────────────────────────────

# cv.Sobel(): 소벨 미분 필터를 이미지에 적용하여 에지를 검출하는 함수
# 소벨 필터란? 이미지의 픽셀 강도 변화(그래디언트)를 계산하여 에지를 찾는 1차 미분 필터
#
# 파라미터 설명:
#   - src: 입력 이미지 (그레이스케일)
#   - ddepth: 출력 이미지의 비트 깊이
#     * cv.CV_64F = 64비트 부동소수점 (float64)
#     * 음수(어두운->밝은 방향의 에지)까지 정확히 저장하기 위해 사용
#     * uint8(8비트)을 사용하면 음수 값이 잘려나가 에지 누락 발생
#   - dx: X 방향 미분 차수 (1이면 x 방향 에지 검출, 0이면 미적용)
#   - dy: Y 방향 미분 차수 (1이면 y 방향 에지 검출, 0이면 미적용)
#   - ksize: 소벨 커널의 크기 (3x3 또는 5x5), 클수록 에지가 부드러워짐

# X축 방향 에지 검출: 수직 에지(세로 경계선)를 찾음
# dx=1, dy=0 → 수평 방향 픽셀 변화율을 계산하여 수직 에지 검출
sobel_x = cv.Sobel(img_gray, cv.CV_64F, 1, 0, ksize=3)
print("[완료] X축 방향 소벨 에지 검출 완료 (수직 에지)")

# Y축 방향 에지 검출: 수평 에지(가로 경계선)를 찾음
# dx=0, dy=1 → 수직 방향 픽셀 변화율을 계산하여 수평 에지 검출
sobel_y = cv.Sobel(img_gray, cv.CV_64F, 0, 1, ksize=3)
print("[완료] Y축 방향 소벨 에지 검출 완료 (수평 에지)")

# ─────────────────────────────────────────────────────────────────────────────
# 4단계: 에지 강도(Magnitude) 계산
# ─────────────────────────────────────────────────────────────────────────────

# cv.magnitude(): 두 성분(x, y) 벡터의 크기(magnitude)를 계산하는 함수
# 에지 강도 공식: magnitude = sqrt(sobel_x^2 + sobel_y^2)
# → X와 Y 방향의 그래디언트를 합산하여 전체 에지 강도를 계산
# → 강도가 클수록 에지(경계선)가 강한 것을 의미
edge_magnitude = cv.magnitude(sobel_x, sobel_y)
print(f"[완료] 에지 강도 계산 완료 (최대값: {edge_magnitude.max():.2f}, 최소값: {edge_magnitude.min():.2f})")

# ─────────────────────────────────────────────────────────────────────────────
# 5단계: float64 -> uint8 변환 (시각화를 위한 자료형 변환)
# ─────────────────────────────────────────────────────────────────────────────

# cv.convertScaleAbs(): 부동소수점 배열을 절댓값 스케일 변환 후 uint8로 변환
# 왜 필요한가?
#   - 소벨 결과는 float64 타입으로, 값 범위가 0~수천에 달할 수 있음
#   - Matplotlib의 imshow()는 uint8(0~255) 또는 [0,1] float을 기대함
#   - 이 함수가 자동으로 값을 0~255 범위로 정규화 및 변환
edge_uint8 = cv.convertScaleAbs(edge_magnitude)
print(f"[완료] 에지 강도 이미지 uint8 변환 완료 (자료형: {edge_uint8.dtype})")

# ─────────────────────────────────────────────────────────────────────────────
# 6단계: 결과 저장 (이미지 파일로 저장)
# ─────────────────────────────────────────────────────────────────────────────

# 결과 이미지를 저장할 폴더 경로 설정 (없으면 생성)
result_dir = os.path.join(script_dir, "results")
os.makedirs(result_dir, exist_ok=True)  # exist_ok=True: 폴더가 이미 있어도 오류 발생 안 함

# 소벨 X 방향 결과를 이미지로 저장
sobel_x_abs = cv.convertScaleAbs(sobel_x)  # 절댓값 변환하여 시각화 가능하게
cv.imwrite(os.path.join(result_dir, "sobel_x.jpg"), sobel_x_abs)

# 소벨 Y 방향 결과를 이미지로 저장
sobel_y_abs = cv.convertScaleAbs(sobel_y)  # 절댓값 변환하여 시각화 가능하게
cv.imwrite(os.path.join(result_dir, "sobel_y.jpg"), sobel_y_abs)

# 에지 강도(최종 결과) 이미지를 저장
cv.imwrite(os.path.join(result_dir, "edge_magnitude.jpg"), edge_uint8)
print(f"[완료] 결과 이미지 저장 완료: {result_dir}/")

# ─────────────────────────────────────────────────────────────────────────────
# 7단계: Matplotlib을 사용한 시각화
# ─────────────────────────────────────────────────────────────────────────────

# OpenCV는 BGR 순서로 이미지를 읽으므로 Matplotlib 시각화 전에 RGB로 변환
# Matplotlib은 RGB 순서를 기대하기 때문에 채널 순서를 바꿔야 올바른 색상이 표시됨
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

# plt.figure(): 새 Figure(캔버스) 생성, figsize로 가로x세로 인치 크기 지정
# 2개의 이미지를 나란히 배치하므로 가로를 넉넉하게 설정
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 전체 Figure의 제목 설정
fig.suptitle("Problem 1: 소벨(Sobel) 에지 검출 결과", fontsize=16, fontweight='bold')

# ── 첫 번째 서브플롯: 원본 이미지 ──
axes[0].imshow(img_rgb)                       # RGB 원본 이미지 표시
axes[0].set_title("원본 이미지 (Original)", fontsize=13)  # 서브플롯 제목
axes[0].axis('off')                           # 축(눈금, 레이블) 숨김

# ── 두 번째 서브플롯: 에지 강도 이미지 ──
# cmap='gray': 단일 채널 이미지를 흑백(그레이스케일)으로 시각화
# 밝은 픽셀 = 에지(경계선)가 강한 부분, 어두운 픽셀 = 평탄한 부분
axes[1].imshow(edge_uint8, cmap='gray')        # 에지 강도 이미지 표시 (흑백)
axes[1].set_title("소벨 에지 강도 (Sobel Edge Magnitude)", fontsize=13)
axes[1].axis('off')                            # 축 숨김

# 서브플롯 사이 간격 자동 조정
plt.tight_layout()

# 시각화 결과를 파일로 저장
output_path = os.path.join(result_dir, "result_visualization.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')  # dpi=150: 해상도 설정
print(f"[완료] 시각화 결과 저장 완료: {output_path}")

# plt.show(): 화면에 그래프/이미지를 표시 (GUI 환경에서 동작)
plt.show()

# 최종 완료 메시지 출력
print("\n=== Problem 1 소벨 에지 검출 완료 ===")
print(f"   결과 파일 위치: {result_dir}/")
print("   생성된 파일: sobel_x.jpg, sobel_y.jpg, edge_magnitude.jpg, result_visualization.png")
