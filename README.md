# 🚶 Seulsekwon Analysis (슬세권 분석)

**"Slippers + Area" Analysis: Finding the best living areas for single-person households in Seoul.**

## 📌 Project Overview (프로젝트 개요)
이 프로젝트는 **'슬세권(슬리퍼를 신고 편의시설을 이용할 수 있는 권역)'** 지수를 산출하고 시각화하는 파이썬 분석 도구입니다.
단순히 시설의 개수를 세는 것이 아니라, 1인 가구에게 필수적인 **카페, 운동시설, 편의점**과의 **거리 붕괴 함수(Distance Decay Function)**를 적용하여 실제 생활 편의성을 정량적으로 평가합니다.

![Visualization Preview](preview.png)
*(실제 실행 시 seulsekwon_map.html 파일로 인터랙티브 지도가 생성됩니다)*

## 🎯 Key Features (핵심 기능)
1.  **Multi-Factor Scoring**: 카페, 헬스장, 편의점의 복합적인 접근성을 분석합니다.
2.  **Distance Decay Algorithm**: 거리가 멀어질수록 점수가 선형적으로 감소하는 정교한 채점 방식을 사용합니다. (100m 이내 10점 ~ 1km 이상 0점)
3.  **Fast Spatial Indexing**: `KDTree` 알고리즘을 사용하여 대용량 위치 데이터의 거리를 수 초 내에 계산합니다.
4.  **Interactive Visualization**:
    *   **Heatmap**: 편의시설 밀집도를 붉은색 히트맵으로 시각화.
    *   **Marker Clustering**: 개별 시설물(상호명 포함)을 종류별 아이콘으로 클러스터링하여 표시.

## 🛠️ Technology Stack
- **Language**: Python 3.8+
- **Data Processing**: Pandas, GeoPandas, NumPy, Shapely
- **Visualization**: Folium (Leaflet.js based)
- **Algorithm**: Scipy (Spatial KDTree)

## 🚀 How to Run (실행 방법)

### 1. Prerequisites (준비물)
파이썬 환경이 필요합니다. 아래 명령어로 의존성 패키지를 설치하세요.
```bash
pip install -r requirements.txt
```

### 2. Run Analysis (실행)
메인 스크립트를 실행하면 데이터 생성(Mock Data 사용 가능), 점수 계산, 지도 생성이 자동으로 진행됩니다.
```bash
python main.py
```

### 3. Check Results (결과 확인)
생성된 `seulsekwon_map.html` 파일을 웹 브라우저로 엽니다.

## 📂 File Structure
```
seulsekwon_analysis/
├── algorithm.py        # KDTree 기반 거리 계산 및 점수 산출 로직
├── data_processor.py   # 데이터 로딩, 전처리, Mock 데이터 생성
├── visualization.py    # Folium 지도 시각화 및 마커 클러스터링
├── main.py             # 전체 워크플로우 실행
├── requirements.txt    # 필요 라이브러리 목록
└── seulsekwon_map.html # 결과물 (생성됨)
```

## 📊 Logic Details
**Scoring Formula:**
$$S(d) = \begin{cases} 10 & d \le 100m \\ \text{Linear Decay} & 100m < d < 1000m \\ 0 & d \ge 1000m \end{cases}$$
- 각 격자점(Grid)에서 주변 시설물까지의 점수를 합산하여 최종 '슬세권 지수'를 도출합니다.

---
*Created by Antigravity*
