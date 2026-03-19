# L03 Edge and Region - Homework

> **과목**: 컴퓨터비전 (Computer Vision)  
> **교수**: 서정일 | **대학**: 동아대학교 컴퓨터AI공학부  
> **주차**: L03 Edge and Region

---

## 📁 레포지토리 구조 (Repository Structure)

```
3week/
├── README.md                         ← 전체 과제 개요 (이 파일)
├── requirements.txt                  ← 전체 공통 패키지 목록
│
├── base/                             ← 원본 이미지 및 PDF (제공 파일)
│   ├── L03_Edge_and_Region.pdf
│   ├── edgeDetectionImage.jpg
│   ├── dabo.jpg
│   └── coffee cup.JPG
│
├── Problem_1/                        ← 문제 1: 소벨 에지 검출
│   ├── main.py                       ← 소벨 에지 검출 코드 (주석 포함)
│   ├── README.md                     ← 문제 1 상세 설명
│   ├── requirements.txt              ← 문제 1 패키지 목록
│   ├── images/
│   │   └── edgeDetectionImage.jpg    ← 입력 이미지
│   └── results/                      ← 실행 후 자동 생성
│       ├── sobel_x.jpg
│       ├── sobel_y.jpg
│       ├── edge_magnitude.jpg
│       └── result_visualization.png
│
├── Problem_2/                        ← 문제 2: 캐니 에지 + 허프 직선 검출
│   ├── main.py                       ← 캐니+허프 변환 코드 (주석 포함)
│   ├── README.md                     ← 문제 2 상세 설명
│   ├── requirements.txt              ← 문제 2 패키지 목록
│   ├── images/
│   │   └── dabo.jpg                  ← 입력 이미지
│   └── results/                      ← 실행 후 자동 생성
│       ├── canny_edges.jpg
│       ├── lines_detected.jpg
│       └── result_visualization.png
│
└── Problem_3/                        ← 문제 3: GrabCut 객체 추출
    ├── main.py                       ← GrabCut 코드 (주석 포함)
    ├── README.md                     ← 문제 3 상세 설명
    ├── requirements.txt              ← 문제 3 패키지 목록
    ├── images/
    │   └── coffee_cup.jpg            ← 입력 이미지
    └── results/                      ← 실행 후 자동 생성
        ├── grabcut_mask.jpg
        ├── foreground_extracted.jpg
        └── result_visualization.png
```

> **가상환경 폴더**: 각 Problem 폴더 또는 루트에 `.venv/` 또는 `env/`로 생성됩니다.
> `.gitignore`에 추가하여 GitHub에는 업로드하지 않는 것을 권장합니다.

---

## 🧾 문제 요약

| 문제 | 제목 | 핵심 기술 | 사용 함수 |
|------|------|-----------|-----------|
| **Problem 1** | 소벨 에지 검출 및 시각화 | Sobel Filter | `cv.Sobel`, `cv.magnitude` |
| **Problem 2** | 캐니 에지 + 허프 직선 검출 | Canny + HoughLinesP | `cv.Canny`, `cv.HoughLinesP` |
| **Problem 3** | GrabCut 대화식 객체 추출 | Graph Cut Segmentation | `cv.grabCut`, `np.where` |

---

## ⚡ 빠른 시작 (Quick Start)

### 전체 패키지 설치 (공통 가상환경 사용)

```bash
# 1. 루트 폴더로 이동
cd /home/ji/Desktop/homework/3week

# 2. 가상환경 생성 (Python 3.10 이상 권장)
python3 -m venv .venv

# 3. 가상환경 활성화
source .venv/bin/activate

# 4. 공통 패키지 설치
pip install -r requirements.txt

# 5. 각 문제 실행
python Problem_1/main.py
python Problem_2/main.py
python Problem_3/main.py
```

### Conda 가상환경 사용 시

```bash
conda create -n cv_homework python=3.10 -y
conda activate cv_homework
pip install -r requirements.txt

python Problem_1/main.py
python Problem_2/main.py
python Problem_3/main.py
```

---

## 📦 공통 의존성 (Dependencies)

```
opencv-python >= 4.8.0   # 이미지 처리, 에지 검출, GrabCut
numpy >= 1.24.0          # 배열 연산, 마스크 처리
matplotlib >= 3.7.0      # 결과 시각화
```

---

## 📝 .gitignore 권장 설정

```
# 가상환경 폴더
.venv/
env/
venv/

# Python 캐시
__pycache__/
*.pyc

# 결과 이미지 (선택적)
# Problem_*/results/
```
