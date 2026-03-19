"""
=============================================================================
Problem 2: 캐니(Canny) 에지 및 허프(Hough) 변환을 이용한 직선 검출
=============================================================================
이 코드는 OpenCV를 사용하여 dabo 이미지에서 캐니 에지를 추출하고,
허프 확률적 변환(HoughLinesP)으로 직선을 검출하여 원본 이미지에 빨간색으로 표시합니다.


과목: 컴퓨터비전 - L03 Edge and Region
=============================================================================
"""

# OpenCV 라이브러리 임포트: 이미지 처리 및 컴퓨터비전의 핵심 라이브러리
import cv2 as cv

# NumPy 라이브러리 임포트: 배열 처리 및 수치 연산에 사용
import numpy as np

# Matplotlib pyplot 임포트: 결과 이미지 시각화에 사용
import matplotlib.pyplot as plt

# os 모듈 임포트: 파일 경로 및 디렉토리 처리에 사용
import os

# ─────────────────────────────────────────────────────────────────────────────
# 한글 폰트 설정 (matplotlib에서 한글 제목/라벨이 깨지지 않도록)
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib
from matplotlib import font_manager
_korean_fonts = ['NanumGothic', 'NanumBarunGothic', 'Malgun Gothic', 'AppleGothic', 'UnDotum']
_available_fonts = {f.name for f in font_manager.fontManager.ttflist}
for _font in _korean_fonts:
    if _font in _available_fonts:
        matplotlib.rcParams['font.family'] = _font
        break
matplotlib.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────────────────────────────────────
# 1단계: 이미지 불러오기
# ─────────────────────────────────────────────────────────────────────────────

# 현재 스크립트가 위치한 디렉토리를 기준으로 이미지 경로 설정
# os.path.abspath(__file__): 이 파이썬 파일의 절대 경로
# os.path.dirname(): 파일이 위치한 폴더 경로 추출
script_dir = os.path.dirname(os.path.abspath(__file__))

# dabo.jpg 이미지의 절대 경로 생성
image_path = os.path.join(script_dir, "images", "dabo.jpg")

# cv.imread(): 이미지 파일을 BGR 형식의 NumPy 배열로 불러옴
img_bgr = cv.imread(image_path)

# 이미지 불러오기 실패 시 예외 처리 (파일 없음 등)
if img_bgr is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

# 불러오기 성공 메시지 및 이미지 기본 정보 출력
print(f"[완료] 이미지 불러오기 성공: {image_path}")
print(f"[정보] 이미지 크기: {img_bgr.shape[1]}x{img_bgr.shape[0]} (너비x높이)")

# ─────────────────────────────────────────────────────────────────────────────
# 2단계: 그레이스케일 변환
# ─────────────────────────────────────────────────────────────────────────────

# 캐니 에지 검출은 단채널(그레이스케일) 이미지에 적용하므로 변환 필요
# cv.cvtColor(): 색상 공간 변환 함수 (BGR -> GRAY)
img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
print("[완료] 그레이스케일 변환 완료")

# ─────────────────────────────────────────────────────────────────────────────
# 3단계: 캐니(Canny) 에지 검출
# ─────────────────────────────────────────────────────────────────────────────

# cv.Canny(): 캐니 에지 검출 알고리즘 적용 함수
# 캐니 알고리즘 동작 원리:
#   1) 가우시안 블러를 적용하여 노이즈 제거
#   2) 소벨 필터로 그래디언트(기울기) 크기와 방향 계산
#   3) 비최대 억제(Non-Maximum Suppression)로 에지 얇게 만들기
#   4) 이중 임계값(Hysteresis Thresholding)으로 강한 에지 vs 약한 에지 분류
#
# 파라미터 설명:
#   - image: 입력 그레이스케일 이미지
#   - threshold1 (=100): 하위 임계값 - 이 값 미만의 그래디언트는 에지에서 제외
#   - threshold2 (=200): 상위 임계값 - 이 값 이상의 그래디언트는 강한 에지로 확정
#   사이값(100~200): 강한 에지와 연결된 경우에만 에지로 포함 (Hysteresis 처리)
# 결과: 에지 픽셀은 255(흰색), 비에지 픽셀은 0(검정)인 이진 이미지
edges = cv.Canny(img_gray, threshold1=100, threshold2=200)
print(f"[완료] 캐니 에지 검출 완료 (임계값: threshold1=100, threshold2=200)")
print(f"[정보] 에지 픽셀 수: {np.count_nonzero(edges)} 픽셀 / 전체 {edges.size} 픽셀")

# ─────────────────────────────────────────────────────────────────────────────
# 4단계: 허프 확률적 변환(HoughLinesP)을 이용한 직선 검출
# ─────────────────────────────────────────────────────────────────────────────

# cv.HoughLinesP(): 확률적 허프 변환(Probabilistic Hough Transform)으로 직선 검출
# 일반 허프 변환(HoughLines)과의 차이:
#   - HoughLines: 무한히 긴 직선을 검출 (rho, theta 반환)
#   - HoughLinesP: 유한한 선분(두 점의 좌표)을 검출 → 시각화에 더 편리
#
# 파라미터 설명:
#   - image: 에지 이미지 (캐니 결과)
#   - rho (=1): 허프 공간에서 거리 해상도 (픽셀 단위), 작을수록 정밀하지만 느림
#   - theta (=np.pi/180): 각도 해상도 (라디안 단위), π/180 = 1도
#   - threshold (=80): 직선으로 판단하기 위한 최소 투표 수
#                      (에지 공간에서 해당 직선을 지지하는 점의 최소 개수)
#                      → 값이 작을수록 더 많은(짧은) 직선 검출, 클수록 긴 직선만 검출
#   - minLineLength (=50): 직선으로 인정하는 최소 길이 (픽셀 단위)
#                          이 값보다 짧은 선분은 무시
#   - maxLineGap (=10): 두 선분 사이의 최대 허용 간격 (픽셀 단위)
#                       이 값 이하의 간격이면 같은 직선으로 연결
lines = cv.HoughLinesP(
    edges,              # 캐니 에지 이미지 입력
    rho=1,              # 거리 해상도 (픽셀)
    theta=np.pi / 180,  # 각도 해상도 (1도)
    threshold=80,       # 최소 투표 수 (직선 판정 기준)
    minLineLength=50,   # 검출할 최소 직선 길이 (픽셀)
    maxLineGap=10       # 선분 연결 허용 최대 간격 (픽셀)
)

# 검출된 직선 수 출력 (lines가 None이면 검출 실패)
if lines is not None:
    print(f"[완료] 허프 변환 직선 검출 완료: {len(lines)}개의 직선 검출")
else:
    print("[경고] 직선이 검출되지 않았습니다. 파라미터를 조정해 보세요.")
    lines = []  # 빈 리스트로 초기화하여 이후 코드 오류 방지

# ─────────────────────────────────────────────────────────────────────────────
# 5단계: 검출된 직선을 원본 이미지에 그리기
# ─────────────────────────────────────────────────────────────────────────────

# 원본 이미지를 복사하여 직선을 그릴 이미지 생성
# .copy()를 사용하는 이유: 원본 이미지(img_bgr)를 수정하지 않고 별도 복사본에 그림
img_with_lines = img_bgr.copy()

# 검출된 각 직선에 대해 반복하여 이미지에 그리기
for line in lines:
    # line 배열의 구조: [[x1, y1, x2, y2]]
    # [0]을 통해 내부 배열에서 좌표값 추출
    x1, y1, x2, y2 = line[0]

    # cv.line(): 이미지에 직선을 그리는 함수
    # 파라미터:
    #   - img: 직선을 그릴 이미지
    #   - pt1: 시작점 (x1, y1)
    #   - pt2: 끝점 (x2, y2)
    #   - color: 직선 색상 (B, G, R) = (0, 0, 255) = 빨간색
    #   - thickness: 직선 두께 (픽셀 단위)
    cv.line(img_with_lines, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

print(f"[완료] 원본 이미지에 {len(lines)}개의 빨간색 직선 그리기 완료")

# ─────────────────────────────────────────────────────────────────────────────
# 6단계: 결과 이미지 저장
# ─────────────────────────────────────────────────────────────────────────────

# 결과 저장 폴더 생성 (존재하면 무시)
result_dir = os.path.join(script_dir, "results")
os.makedirs(result_dir, exist_ok=True)

# 캐니 에지 이미지 저장
cv.imwrite(os.path.join(result_dir, "canny_edges.jpg"), edges)

# 직선이 그려진 이미지 저장 (BGR 형식 그대로 저장)
cv.imwrite(os.path.join(result_dir, "lines_detected.jpg"), img_with_lines)

print(f"[완료] 결과 이미지 저장 완료: {result_dir}/")

# ─────────────────────────────────────────────────────────────────────────────
# 7단계: Matplotlib을 사용한 시각화
# ─────────────────────────────────────────────────────────────────────────────

# Matplotlib은 RGB 순서를 사용하므로, BGR 이미지를 RGB로 변환
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)              # 원본 이미지 RGB 변환
img_lines_rgb = cv.cvtColor(img_with_lines, cv.COLOR_BGR2RGB)  # 직선 이미지 RGB 변환

# 1x2 서브플롯 생성 (원본, 직선 검출 결과를 나란히 배치)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Figure 전체 제목 설정
fig.suptitle("Problem 2: 캐니 에지 & 허프 변환 직선 검출", fontsize=16, fontweight='bold')

# ── 첫 번째 서브플롯: 원본 이미지 ──
axes[0].imshow(img_rgb)                              # 원본 이미지 시각화
axes[0].set_title("원본 이미지 (Original)", fontsize=13)
axes[0].axis('off')                                  # 축 눈금 제거

# ── 두 번째 서브플롯: 허프 변환으로 검출된 직선 ──
axes[1].imshow(img_lines_rgb)                        # 직선이 그려진 이미지 시각화
axes[1].set_title(f"직선 검출 결과 ({len(lines)}개 검출, 빨간색으로 표시)", fontsize=13)
axes[1].axis('off')                                  # 축 눈금 제거

# 서브플롯 간격 자동 조정
plt.tight_layout()

# 시각화 결과 파일로 저장 (dpi=150: 적당한 해상도로 저장)
output_path = os.path.join(result_dir, "result_visualization.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"[완료] 시각화 결과 저장 완료: {output_path}")

# 화면에 결과 창 표시
plt.show()

# 최종 완료 메시지 출력
print("\n=== Problem 2 캐니 에지 및 허프 변환 직선 검출 완료 ===")
print(f"   결과 파일 위치: {result_dir}/")
print("   생성된 파일: canny_edges.jpg, lines_detected.jpg, result_visualization.png")
