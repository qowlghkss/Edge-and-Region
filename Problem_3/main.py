"""
=============================================================================
Problem 3: GrabCut을 이용한 대화식 영역 분할 및 객체 추출
=============================================================================
이 코드는 OpenCV의 GrabCut 알고리즘을 사용하여 coffee cup 이미지에서
사용자가 지정한 사각형 영역을 기반으로 객체(컵)를 배경으로부터 자동 분리합니다.

과목: 컴퓨터비전 - L03 Edge and Region
=============================================================================
"""

# OpenCV 라이브러리 임포트: 이미지 처리 및 GrabCut 알고리즘 사용
import cv2 as cv

# NumPy 라이브러리 임포트: 배열 생성 및 마스크 처리에 사용
import numpy as np

# Matplotlib pyplot 임포트: 결과 이미지 3개를 나란히 시각화
import matplotlib.pyplot as plt

# os 모듈 임포트: 파일 경로 및 디렉토리 조작에 사용
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

# 현재 스크립트 파일의 디렉토리를 기준으로 이미지 경로 구성
script_dir = os.path.dirname(os.path.abspath(__file__))

# coffee cup 이미지 경로 (공백 포함 파일명을 coffee_cup.jpg로 복사해서 사용)
image_path = os.path.join(script_dir, "images", "coffee_cup.jpg")

# 이미지를 BGR 형식으로 불러옴
img_bgr = cv.imread(image_path)

# 이미지 불러오기 실패 시 예외 처리
if img_bgr is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

# 이미지 정보 출력
print(f"[완료] 이미지 불러오기 성공: {image_path}")
print(f"[정보] 이미지 크기: {img_bgr.shape[1]}x{img_bgr.shape[0]} (너비x높이)")

# ─────────────────────────────────────────────────────────────────────────────
# 2단계: GrabCut 초기 모델 및 마스크 설정
# ─────────────────────────────────────────────────────────────────────────────

# GrabCut 알고리즘 초기화를 위한 마스크 생성
# np.zeros(): 모든 값이 0인 배열 생성
# 마스크 크기: 이미지와 동일한 크기 (높이, 너비), 자료형: np.uint8 (0~255)
# 초기값 0은 cv.GC_BGD (확실한 배경)을 의미
mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
print(f"[완료] GrabCut 마스크 초기화 완료: shape={mask.shape}, dtype={mask.dtype}")

# GrabCut 내부 가우시안 혼합 모델(GMM) 초기화
# bgdModel: 배경 GMM 모델 파라미터 저장 배열 (1x65, float64로 초기화)
# fgdModel: 전경 GMM 모델 파라미터 저장 배열 (1x65, float64로 초기화)
# 65 = 가우시안 성분의 파라미터 수 (평균, 공분산, 가중치 등)
# 이 배열은 GrabCut이 내부적으로 학습하면서 업데이트하는 모델 파라미터
bgdModel = np.zeros((1, 65), np.float64)  # 배경 모델 파라미터 초기화
fgdModel = np.zeros((1, 65), np.float64)  # 전경 모델 파라미터 초기화

# ─────────────────────────────────────────────────────────────────────────────
# 3단계: 초기 사각형 영역 설정 (관심 영역 지정)
# ─────────────────────────────────────────────────────────────────────────────

# GrabCut에 전달할 초기 ROI(Region of Interest) 직사각형 설정
# 형식: (x, y, width, height)
#   x, y: 사각형의 왼쪽 상단 꼭짓점 좌표
#   width: 사각형의 너비 (픽셀)
#   height: 사각형의 높이 (픽셀)
#
# 이미지 크기에서 여백(margin)을 주어 객체가 포함되도록 설정
# 이미지 크기 기반 동적 사각형 설정
h, w = img_bgr.shape[:2]     # 이미지 높이, 너비
margin = 20                   # 이미지 가장자리에서 안쪽으로 20픽셀 여백

# 전체 이미지에서 가장자리를 제외한 영역을 사각형으로 지정
rect = (margin, margin, w - 2 * margin, h - 2 * margin)
print(f"[완료] 초기 사각형 설정 완료: (x={rect[0]}, y={rect[1]}, w={rect[2]}, h={rect[3]})")
print(f"[정보] 이미지 내 사각형 비율: {rect[2]/w*100:.1f}% x {rect[3]/h*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 4단계: GrabCut 알고리즘 실행
# ─────────────────────────────────────────────────────────────────────────────

# cv.grabCut(): GrabCut 분할 알고리즘 실행 함수
# GrabCut 알고리즘 원리:
#   1) 사각형 외부 → 확실한 배경(GC_BGD)으로 초기화
#   2) 사각형 내부 → GMM(가우시안 혼합 모델)로 전경/배경 분리 시작
#   3) 그래프 컷(Graph Cut) 최적화로 전경/배경 분류 반복 개선
#
# 파라미터 설명:
#   - img: 입력 컬러 이미지 (BGR)
#   - mask: 초기 마스크 (0: 배경, 1: 전경, 2: 가능배경, 3: 가능전경)
#   - rect: 초기 사각형 좌표 (x, y, w, h)
#   - bgdModel: 배경 GMM 모델 (내부적으로 업데이트됨)
#   - fgdModel: 전경 GMM 모델 (내부적으로 업데이트됨)
#   - iterCount: 알고리즘 반복 횟수 (많을수록 정확하지만 느림)
#   - mode: 초기화 방법 (cv.GC_INIT_WITH_RECT: 사각형으로 초기화)
cv.grabCut(
    img_bgr,            # 입력 이미지 (BGR)
    mask,               # 마스크 배열 (알고리즘이 업데이트함)
    rect,               # 초기 사각형 영역
    bgdModel,           # 배경 GMM 모델 (출력: 학습된 파라미터)
    fgdModel,           # 전경 GMM 모델 (출력: 학습된 파라미터)
    iterCount=10,       # 반복 횟수 (10번 반복으로 분류 정확도 향상)
    mode=cv.GC_INIT_WITH_RECT  # 사각형 기반 초기화 모드
)
print("[완료] GrabCut 알고리즘 실행 완료 (10회 반복 최적화)")

# ─────────────────────────────────────────────────────────────────────────────
# 5단계: 마스크 처리 - 전경/배경 분리
# ─────────────────────────────────────────────────────────────────────────────

# GrabCut 실행 후 마스크 값 설명:
#   cv.GC_BGD (=0): 확실한 배경   → 최종 배경으로 처리
#   cv.GC_FGD (=1): 확실한 전경   → 최종 전경으로 처리
#   cv.GC_PR_BGD (=2): 배경 가능성 → 배경으로 처리
#   cv.GC_PR_FGD (=3): 전경 가능성 → 전경으로 처리 (GrabCut의 추정값)
#
# np.where(): 조건에 따라 값을 선택하는 함수
# 조건: 마스크 값이 2(배경가능성) 또는 0(확실배경)이면 0, 그 외는 1
# → 전경(1)과 가능전경(3)을 1로, 배경(0)과 가능배경(2)을 0으로 변환
# np.uint8로 변환하여 이미지 곱셈에 사용 가능하게 함
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype(np.uint8)

# 마스크 통계 출력 (전경/배경 픽셀 수 확인)
foreground_pixels = np.sum(mask2 == 1)   # 전경으로 분류된 픽셀 수
background_pixels = np.sum(mask2 == 0)  # 배경으로 분류된 픽셀 수
print(f"[완료] 마스크 생성 완료 - 전경: {foreground_pixels}픽셀, 배경: {background_pixels}픽셀")
print(f"[정보] 전경 비율: {foreground_pixels/(foreground_pixels+background_pixels)*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 6단계: 배경 제거 - 객체(전경)만 추출
# ─────────────────────────────────────────────────────────────────────────────

# 마스크를 3채널로 확장하여 컬러 이미지에 적용 가능하게 변환
# mask2[:, :, np.newaxis]: (H, W) -> (H, W, 1) 형태로 차원 추가
# * img_bgr: 브로드캐스팅으로 3채널 이미지 각각에 마스크 곱셈 적용
#   → 전경(mask=1): 원본 픽셀 값 유지 (1을 곱하므로)
#   → 배경(mask=0): 픽셀 값이 0이 됨 = 검정색(완전 배경 제거)
img_foreground = img_bgr * mask2[:, :, np.newaxis]
print("[완료] 배경 제거 완료 - 전경 객체만 추출된 이미지 생성")

# ─────────────────────────────────────────────────────────────────────────────
# 7단계: 결과 이미지 저장
# ─────────────────────────────────────────────────────────────────────────────

# 결과 저장 폴더 생성
result_dir = os.path.join(script_dir, "results")
os.makedirs(result_dir, exist_ok=True)  # 폴더 없으면 자동 생성

# 마스크 이미지 저장 (0 또는 1의 이진 마스크를 0 또는 255로 스케일링하여 가시화)
mask_visual = mask2 * 255  # 1 → 255(흰색), 0 → 0(검정색)으로 스케일링
cv.imwrite(os.path.join(result_dir, "grabcut_mask.jpg"), mask_visual)

# 배경 제거된 전경 이미지 저장
cv.imwrite(os.path.join(result_dir, "foreground_extracted.jpg"), img_foreground)

print(f"[완료] 결과 이미지 저장 완료: {result_dir}/")

# ─────────────────────────────────────────────────────────────────────────────
# 8단계: Matplotlib을 사용한 3개 이미지 나란히 시각화
# ─────────────────────────────────────────────────────────────────────────────

# BGR -> RGB 변환 (Matplotlib은 RGB 채널 순서를 사용)
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)              # 원본 이미지
img_fg_rgb = cv.cvtColor(img_foreground, cv.COLOR_BGR2RGB)    # 배경 제거 이미지

# 1x3 서브플롯 생성 (3개 이미지를 가로로 나란히)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 전체 Figure 제목
fig.suptitle("Problem 3: GrabCut 대화식 영역 분할 및 객체 추출", fontsize=16, fontweight='bold')

# ── 첫 번째: 원본 이미지 ──
axes[0].imshow(img_rgb)                              # 원본 컬러 이미지 시각화
axes[0].set_title("원본 이미지 (Original)", fontsize=12)
axes[0].axis('off')                                  # 축 눈금 제거

# ── 두 번째: GrabCut 마스크 이미지 ──
# cmap='gray': 이진 마스크를 흑백으로 시각화 (흰색=전경, 검정=배경)
axes[1].imshow(mask2, cmap='gray')                   # 마스크 시각화
axes[1].set_title("GrabCut 마스크\n(흰색=전경, 검정=배경)", fontsize=12)
axes[1].axis('off')                                  # 축 눈금 제거

# ── 세 번째: 배경 제거 이미지 ──
axes[2].imshow(img_fg_rgb)                           # 전경(객체)만 남은 이미지
axes[2].set_title("배경 제거 결과\n(전경 객체만 추출)", fontsize=12)
axes[2].axis('off')                                  # 축 눈금 제거

# 서브플롯 사이 여백 자동 조정
plt.tight_layout()

# 시각화 결과 파일 저장
output_path = os.path.join(result_dir, "result_visualization.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"[완료] 시각화 결과 저장 완료: {output_path}")

# 화면에 결과 표시
plt.show()

# 최종 완료 메시지
print("\n=== Problem 3 GrabCut 객체 추출 완료 ===")
print(f"   결과 파일 위치: {result_dir}/")
print("   생성된 파일: grabcut_mask.jpg, foreground_extracted.jpg, result_visualization.png")
