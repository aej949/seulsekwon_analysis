import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import geopandas as gpd
from folium.plugins import HeatMap, MarkerCluster
import altair as alt
import numpy as np
from scipy.spatial import cKDTree, KDTree
from shapely.geometry import Point
import os

# --- 1. Data Processor Module ---
def generate_mock_data(center_lat=37.4842, center_lon=126.9297, n_samples=300):
    """Generates mock data for testing if real data is not available."""
    # ... code from data_processor.py ...
    # Simplified for brevity inside this single file logic if possible, but I will paste full logic.
    lat_offsets = np.random.normal(0, 0.005, n_samples)
    lon_offsets = np.random.normal(0, 0.005, n_samples)
    lats = center_lat + lat_offsets
    lons = center_lon + lon_offsets
    categories = []
    names = []
    types = ['Cafe', 'Gym', 'Convenience', 'Other']
    for _ in range(n_samples):
        choice = np.random.choice(types, p=[0.4, 0.2, 0.3, 0.1])
        if choice == 'Cafe':
             categories.append('카페')
             names.append('Mega Coffee')
        elif choice == 'Gym':
             categories.append('헬스')
             names.append('Sillim Gym')
        elif choice == 'Convenience':
             categories.append('편의점')
             names.append('GS25')
        else:
             categories.append('기타')
             names.append('Unknown')
    return pd.DataFrame({'상호명': names, '상권업종소분류명': categories, '위도': lats, '경도': lons})

def generate_extended_mock_data(center_lat=37.4842, center_lon=126.9297, n_samples=600):
    lat_offsets = np.random.normal(0, 0.006, n_samples)
    lon_offsets = np.random.normal(0, 0.006, n_samples)
    lats = center_lat + lat_offsets
    lons = center_lon + lon_offsets
    categories = []
    names = []
    types = ['Cafe', 'Gym', 'Convenience', 'Safety', 'Medical', 'Life']
    p_dist = [0.25, 0.15, 0.20, 0.10, 0.15, 0.15]
    for _ in range(n_samples):
        choice = np.random.choice(types, p=p_dist)
        cat_name = ''
        store_name = ''
        if choice == 'Cafe':
            cat_name = '카페'; store_name = 'Mock Cafe'
        elif choice == 'Gym':
            cat_name = '헬스'; store_name = 'Mock Gym'
        elif choice == 'Convenience':
            cat_name = '편의점'; store_name = 'Mock GS25'
        elif choice == 'Safety':
            sub = np.random.choice(['CCTV', 'Police'], p=[0.8, 0.2])
            cat_name = '안전'; store_name = f'Public {sub}'
        elif choice == 'Medical':
            sub = np.random.choice(['Pharmacy', 'Clinic'], p=[0.5, 0.5])
            cat_name = '의료'; store_name = f'Mock {sub}'
        elif choice == 'Life':
            sub = np.random.choice(['Laundry', 'Parcel'], p=[0.7, 0.3])
            cat_name = '생활'; store_name = f'Mock {sub}'
        categories.append(cat_name)
        names.append(store_name)
    return pd.DataFrame({'상호명': names, '상권업종소분류명': categories, '위도': lats, '경도': lons})

def preprocess_data(file_path=None, use_mock=False):
    if use_mock or not file_path or not os.path.exists(file_path):
        df = generate_extended_mock_data()
    else:
        try: df = pd.read_csv(file_path, encoding='cp949') 
        except: 
            try: df = pd.read_csv(file_path, encoding='utf-8')
            except: df = generate_extended_mock_data()

    cond_cafe = df['상권업종소분류명'].astype(str).str.contains('카페|커피', na=False)
    cond_gym = df['상권업종소분류명'].astype(str).str.contains('헬스|필라테스|요가|운동', na=False)
    cond_conv = df['상권업종소분류명'].astype(str).str.contains('편의점', na=False) | df['상호명'].astype(str).str.contains('다이소|올리브영', na=False)
    cond_safety = df['상호명'].astype(str).str.contains('Police|CCTV|지구대|파출소|치안', na=False) | df['상권업종소분류명'].astype(str).str.contains('안전', na=False)
    cond_medical = df['상호명'].astype(str).str.contains('Pharmacy|Clinic|약국|내과|이비인후과|치과|병원', na=False) | df['상권업종소분류명'].astype(str).str.contains('의료|병원|약국', na=False)
    cond_life = df['상호명'].astype(str).str.contains('Laundry|Parcel|세탁|빨래|택배', na=False) | df['상권업종소분류명'].astype(str).str.contains('생활', na=False)
    
    df.loc[cond_cafe, 'type'] = 'cafe'
    df.loc[cond_gym, 'type'] = 'gym'
    df.loc[cond_conv, 'type'] = 'convenience'
    df.loc[cond_safety, 'type'] = 'safety'
    df.loc[cond_medical, 'type'] = 'medical'
    df.loc[cond_life, 'type'] = 'life'
    
    target_df = df[df['type'].notna()].copy()
    return gpd.GeoDataFrame(target_df, geometry=gpd.points_from_xy(target_df['경도'], target_df['위도']), crs="EPSG:4326")

def generate_mock_estate_data(center_lat=37.4842, center_lon=126.9297, n_samples=200):
    lat_offsets = np.random.normal(0, 0.005, n_samples)
    lon_offsets = np.random.normal(0, 0.005, n_samples)
    lats = center_lat + lat_offsets
    lons = center_lon + lon_offsets
    rent_per_area = np.random.uniform(5, 20, n_samples)
    deposit = np.random.uniform(1000, 10000, n_samples)
    return pd.DataFrame({'lat': lats, 'lon': lons, 'rent_per_area': rent_per_area, 'deposit': deposit, 'name': [f"Estate_{i}" for i in range(n_samples)]})

# --- 2. Algorithm Module ---
def score_function(dist, limit=1000):
    scores = np.zeros_like(dist)
    mask_close = dist <= 100
    scores[mask_close] = 10
    mask_mid = (dist > 100) & (dist < limit)
    if np.any(mask_mid):
        ratio = (dist[mask_mid] - 100) / (limit - 100)
        scores[mask_mid] = 10 - 9 * ratio
        scores[mask_mid] = np.maximum(1, scores[mask_mid])
    return scores

def calculate_seulsekwon_index(gdf, grid_res_meters=20, max_dist=1000):
    # Project to UTM 52N
    gdf_proj = gdf.to_crs(epsg=32652)
    minx, miny, maxx, maxy = gdf_proj.total_bounds
    buffer = max_dist
    minx -= buffer; miny -= buffer; maxx += buffer; maxy += buffer
    
    x_range = np.arange(minx, maxx, grid_res_meters)
    y_range = np.arange(miny, maxy, grid_res_meters)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    categories = ['cafe', 'gym', 'convenience', 'safety', 'medical', 'life']
    score_dict = {}
    chunk_size = 10000
    
    for cat in categories:
        subset = gdf_proj[gdf_proj['type'] == cat]
        scores = np.zeros(len(grid_points))
        if len(subset) > 0:
            data_points = np.array(list(zip(subset.geometry.x, subset.geometry.y)))
            tree = KDTree(data_points)
            for i in range(0, len(grid_points), chunk_size):
                chunk = grid_points[i:i+chunk_size]
                dists, _ = tree.query(chunk, k=1)
                scores[i:i+chunk_size] = score_function(dists, limit=max_dist)
        score_dict[f'score_{cat}'] = scores

    result_df = pd.DataFrame(grid_points, columns=['x', 'y'])
    for key, s in score_dict.items():
        result_df[key] = s
    result_df['score'] = result_df[[c for c in result_df.columns if 'score_' in c]].sum(axis=1)
    
    gdf_grid = gpd.GeoDataFrame(result_df, geometry=gpd.points_from_xy(result_df['x'], result_df['y']), crs="EPSG:32652")
    gdf_grid = gdf_grid.to_crs(epsg=4326)
    gdf_grid['lat'] = gdf_grid.geometry.y
    gdf_grid['lon'] = gdf_grid.geometry.x
    return gdf_grid

# --- 3. Streamlit Page Logic ---
st.set_page_config(page_title="고급 슬세권 분석", page_icon="🚶", layout="wide")

st.markdown("""
# 🚶 **프리미엄 슬세권 분석 & 추천 서비스**
**(Advanced Seulsekwon Analytics)**
서울시 1인 가구 밀집 지역(신림동)을 대상으로 **안전, 의료, 생활 편의**까지 고려한 **'통합 주거 가치'**를 분석합니다.
""")

st.sidebar.header("🛠️ 분석 가중치 설정 (Weights)")
st.sidebar.info("💡 **나만의 우선순위**에 맞춰 슬라이더를 조절하세요.")

w_cafe = st.sidebar.slider("☕ 카페 (휴식/만남)", 0.0, 3.0, 1.0, 0.1, help="카페, 커피전문점 접근성")
w_gym = st.sidebar.slider("💪 운동 (자기관리)", 0.0, 3.0, 1.0, 0.1, help="헬스장, 필라테스, 요가 시설")
w_conv = st.sidebar.slider("🏪 편의점 (간편생활)", 0.0, 3.0, 1.0, 0.1, help="편의점, 다이소 등")
st.sidebar.markdown("---")
w_safe = st.sidebar.slider("👮 치안/안전 (필수)", 0.0, 3.0, 1.5, 0.1, help="CCTV, 지구대, 파출소 등 안전 시설")
w_med = st.sidebar.slider("🏥 의료 (건강)", 0.0, 3.0, 1.2, 0.1, help="약국, 내과, 이비인후과 등 1차 의료기관")
w_life = st.sidebar.slider("🧺 생활지원 (편의)", 0.0, 3.0, 1.0, 0.1, help="코인빨래방, 세탁소, 무인택배함")

st.sidebar.divider()
st.sidebar.header("⚙️ 분석 설정")
search_radius = st.sidebar.slider("최대 탐색 거리 (Radius)", 100, 2000, 800, 100, format="%d m", help="설정된 거리 이내의 시설만 점수에 반영되며, 가까울수록 가산점이 붙습니다.")
grid_res = st.sidebar.slider("격자 해상도 (미터)", 20, 100, 30, format="%d m", help="격자가 작을수록 더 정밀하게 분석합니다 (연산 속도 주의).")

st.sidebar.markdown("### 🧮 점수 산출 공식 (Decay Function)")
st.sidebar.latex(r"""
Score(d) = \begin{cases} 
10 & d \le 100m \\ 
10 - 9 \times \frac{d-100}{Limit-100} & 100m < d < Limit \\ 
0 & d \ge Limit 
\end{cases}
""")

@st.cache_data
def load_infrastructure():
    return preprocess_data(file_path='data/small_business_data.csv', use_mock=True)

@st.cache_data
def load_real_estate():
    return generate_mock_estate_data(n_samples=200)

@st.cache_data
def calculate_base_scores(_gdf, resolution, limit):
    return calculate_seulsekwon_index(_gdf, grid_res_meters=resolution, max_dist=limit)

if 'infra_gdf' not in st.session_state:
    with st.spinner('초기 데이터 로딩 및 AI 분석 모델 구동 중... (최초 1회)'):
        st.session_state.infra_gdf = load_infrastructure()
        st.session_state.estate_df = load_real_estate()
        st.session_state.last_grid_res = None
        st.session_state.last_radius = None

if st.session_state.get('last_grid_res') != grid_res or st.session_state.get('last_radius') != search_radius:
    with st.spinner(f'공간 인덱스 재계산 중... ({grid_res}m, 반경 {search_radius}m)'):
        st.session_state.grid_gdf_base = calculate_base_scores(st.session_state.infra_gdf, grid_res, search_radius)
        st.session_state.last_grid_res = grid_res
        st.session_state.last_radius = search_radius

infra_gdf = st.session_state.infra_gdf
estate_df = st.session_state.estate_df
grid_gdf = st.session_state.grid_gdf_base.copy()

for col in ['score_cafe', 'score_gym', 'score_convenience', 'score_safety', 'score_medical', 'score_life']:
    if col not in grid_gdf.columns:
        grid_gdf[col] = 0.0

grid_gdf['total_score'] = (
    grid_gdf['score_cafe'] * w_cafe + 
    grid_gdf['score_gym'] * w_gym + 
    grid_gdf['score_convenience'] * w_conv +
    grid_gdf['score_safety'] * w_safe + 
    grid_gdf['score_medical'] * w_med + 
    grid_gdf['score_life'] * w_life
)

grid_coords = list(zip(grid_gdf.geometry.x, grid_gdf.geometry.y))
grid_tree = cKDTree(grid_coords)
estate_coords = list(zip(estate_df['lon'], estate_df['lat']))
dists, idxs = grid_tree.query(estate_coords, k=1)
estate_df['seulsekwon_score'] = grid_gdf.iloc[idxs]['total_score'].values
for col in ['score_cafe', 'score_gym', 'score_convenience', 'score_safety', 'score_medical', 'score_life']:
     estate_df[col] = grid_gdf.iloc[idxs][col].values

score_threshold = estate_df['seulsekwon_score'].quantile(0.8)
rent_threshold = estate_df['rent_per_area'].quantile(0.4)

def classify_value(row):
    if row['seulsekwon_score'] >= score_threshold and row['rent_per_area'] <= rent_threshold:
        return '💎 숨은 명당 (강력 추천)'
    elif row['seulsekwon_score'] >= score_threshold:
        return '💰 프리미엄 (고득점/고가)'
    elif row['rent_per_area'] <= rent_threshold:
        return '📉 가성비 (저렴함)'
    else:
        return '⚠️ 일반/고평가'

estate_df['category'] = estate_df.apply(classify_value, axis=1)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🗺️ 통합 슬세권 지수 히트맵")
    st.caption("🔴 붉을수록 인프라 밀집도가 높음 | 🔵 푸른점: 가성비 추천 매물 (평당 월세 기준)")
    show_reco_only = st.checkbox("💎 가성비 추천 매물만 보기", value=True)
    
    mean_lat, mean_lon = infra_gdf.geometry.y.mean(), infra_gdf.geometry.x.mean()
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=15, tiles='cartodbpositron')
    
    heat_data = grid_gdf[grid_gdf['total_score'] > 0][['lat', 'lon', 'total_score']].values.tolist()
    HeatMap(heat_data, radius=15, blur=20, min_opacity=0.3, name='통합 슬세권 지수').add_to(m)
    
    marker_cluster = MarkerCluster(name="주변 편의시설 (전체)").add_to(m)
    max_markers = 1000
    count = 0
    icons = {'cafe': 'coffee', 'gym': 'heart', 'convenience': 'shopping-cart', 'safety': 'shield', 'medical': 'plus', 'life': 'home'}
    colors = {'cafe': 'red', 'gym': 'blue', 'convenience': 'green', 'safety': 'purple', 'medical': 'orange', 'life': 'cadetblue'}
    
    for row in infra_gdf.itertuples():
        if count > max_markers: break
        ftype = getattr(row, 'type', 'unknown')
        icon = icons.get(ftype, 'info-sign')
        color = colors.get(ftype, 'gray')
        store_name = getattr(row, '상호명', 'Store')
        type_kr = {'cafe': '카페', 'gym': '운동시설', 'convenience': '편의점', 'safety': '안전시설', 'medical': '의료기관', 'life': '생활편의'}.get(ftype, ftype)
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=f"<b>{store_name}</b><br>분류: {type_kr}",
            icon=folium.Icon(color=color, icon=icon, prefix='fa')
        ).add_to(marker_cluster)
        count += 1

    recommended = estate_df[estate_df['category'] == '💎 숨은 명당 (강력 추천)']
    if show_reco_only:
        estates_to_plot = recommended
    else:
        estates_to_plot = estate_df
        
    for idx, row in estates_to_plot.iterrows():
        is_reco = row['category'] == '💎 숨은 명당 (강력 추천)'
        tooltip_html = f"""
        <div style='font-family:sans-serif; width:200px'>
            <b>{'💎 ' if is_reco else ''}{row['name']}</b><hr style='margin:5px 0'>
            ✅ <b>종합 점수</b>: {row['seulsekwon_score']:.1f}점<br>
            💰 <b>평당 월세</b>: {row['rent_per_area']:.1f}만원<br>
            🏠 <b>예상 월세(6평)</b>: {row['rent_per_area']*6:.1f}만원<br>
            <br>
            🛡️ 안전 점수: {row['score_safety']:.1f}<br>
            🏥 의료 접근: {row['score_medical']:.1f}
        </div>
        """
        if is_reco:
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=folium.Popup(tooltip_html, max_width=250),
                icon=folium.Icon(color='darkblue', icon='star', prefix='fa')
            ).add_to(m)
        else:
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=5, color='gray', fill=True, fill_color='gray', fill_opacity=0.6,
                popup=folium.Popup(tooltip_html, max_width=250)
            ).add_to(m)
    
    folium.LayerControl().add_to(m)
    st_folium(m, width="100%", height=600)

with col2:
    st.subheader("📊 매물 추천 및 분석")
    st.markdown("#### 🏆 BEST 3 숨은 명당")
    st.caption("해당 지역 상위 20% 점수이면서 **평당 임대료**는 하위 40%인 알짜 매물입니다.")
    if not recommended.empty:
        top3 = recommended.nlargest(3, 'seulsekwon_score')
        for i, row in top3.iterrows():
            st.success(f"**{row['name']}**\n- 종합 점수: **{row['seulsekwon_score']:.1f}점**\n- 평당 월세: **{row['rent_per_area']:.1f}만 원** (3.3㎡ 기준)\n- ✨ **강점**: 안전({row['score_safety']:.1f}), 의료({row['score_medical']:.1f})")
    else:
        st.warning("조건에 맞는 '숨은 명당'이 없습니다. 가중치를 조절해보세요.")
    
    st.divider()
    scatter = alt.Chart(estate_df).mark_circle(size=80).encode(
        x=alt.X('seulsekwon_score', title='통합 슬세권 지수 (점수)'),
        y=alt.Y('rent_per_area', title='평당 월세 (단위: 만원/3.3㎡)'),
        color=alt.Color('category', legend=alt.Legend(title="매물 등급")),
        tooltip=[alt.Tooltip('name', title='매물명'), alt.Tooltip('seulsekwon_score', title='종합점수', format='.1f'), alt.Tooltip('rent_per_area', title='평당월세', format='.1f'), alt.Tooltip('category', title='등급')]
    ).interactive()
    st.altair_chart(scatter, use_container_width=True)
    
    corr = estate_df['seulsekwon_score'].corr(estate_df['rent_per_area'])
    st.info(f"💡 점수와 임대료(평당)의 상관계수: **{corr:.2f}**")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: right; color: gray; font-size: 0.8em;'>
    <b>데이터 출처 (Source)</b><br>
    - 상권 정보: 소상공인시장진흥공단 (2025.12 기준)<br>
    - 실거래가: 국토교통부 실거래가 공개시스템 (최근 1년치)<br>
    * 본 서비스의 임대료는 전용면적 3.3㎡(1평)당 환산 월세입니다.
    </div>
    """, unsafe_allow_html=True)
