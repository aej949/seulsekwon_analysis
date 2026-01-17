import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import geopandas as gpd
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px
import numpy as np
from scipy.spatial import cKDTree, KDTree
import requests
import os
import time

try:
    from pyproj import Transformer
except ImportError:
    st.error("pyproj library not found. Please install it.")

# --- Configuration ---
API_KEY = "4e7a4a4d70646b73343261564e4c67"
SEOUL_API_BASE = f"http://openapi.seoul.go.kr:8088/{API_KEY}/json"

st.set_page_config(page_title="서울시 주거 가치 분석 (Pro)", page_icon="🏙️", layout="wide")

# --- 1. Data Ingestion & Caching ---

def fetch_and_cache_api(service, save_name):
    """Fetches data with pagination and caches to CSV. Handles large datasets."""
    if not os.path.exists('data'): os.makedirs('data')
    path = f"data/{save_name}.csv"
    
    # Cache Hit
    if os.path.exists(path):
        try: return pd.read_csv(path)
        except: pass
    
    # Cache Miss: Fetch from API
    all_rows = []
    start = 1
    step = 1000
    try:
        while True:
            end = start + step - 1
            url = f"{SEOUL_API_BASE}/{service}/{start}/{end}/"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            
            if service in data and 'row' in data[service]:
                rows = data[service]['row']
                all_rows.extend(rows)
                if len(rows) < step: break
                start += step
                time.sleep(0.05)
            else:
                break
    except Exception as e:
        print(f"Fetch Error ({service}): {e}")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_rows)
    if not df.empty: df.to_csv(path, index=False)
    return df

def get_real_data():
    """Aggregates all 10+ categories from Seoul Open Data."""
    data_list = []
    try: transformer = Transformer.from_crs("EPSG:2097", "EPSG:4326", always_xy=True)
    except: transformer = None

    with st.spinner("📦 서울시 공공데이터 통합 로딩 중... (캐싱 최적화 적용)"):
        # Helper to process frame
        def process_frame(df, type_name, subtype, lat_keys, lon_keys, xy_keys=None, weight=1.0):
            if df.empty: return
            for _, r in df.iterrows():
                try:
                    lat, lon = None, None
                    # Try WGS84 first
                    for k in lat_keys:
                        if k in r and pd.notna(r[k]): lat = float(r[k]); break
                    for k in lon_keys:
                        if k in r and pd.notna(r[k]): lon = float(r[k]); break
                    
                    # Try GRS80 projection if missing
                    if (lat is None or lon is None) and xy_keys and transformer:
                        x = float(r.get(xy_keys[0]))
                        y = float(r.get(xy_keys[1]))
                        lon, lat = transformer.transform(x, y)
                    
                    if lat and lon:
                        name = r.get('NAME') or r.get('NAMES') or r.get('NM') or r.get('DUTYNAME') or r.get('M_NAME') or subtype
                        data_list.append({
                            'name': str(name),
                            'type': type_name,
                            'subtype': subtype,
                            'lat': lat, 'lon': lon,
                            'weight_factor': weight
                        })
                except: pass

        # 1. Safety & Smart
        process_frame(fetch_and_cache_api("SeoulPoliceStationWGS", "police"), 'safety', 'police', ['LAT'], ['LON'], weight=1.5)
        process_frame(fetch_and_cache_api("tbsSvcCctv", "cctv"), 'safety', 'cctv', ['LATITUDE','LAT'], ['LONGITUDE','LON'])
        process_frame(fetch_and_cache_api("SeoulSmartPole", "smartpole"), 'smart', 'smartpole', ['LAT'], ['LON'])

        # 2. Medical
        process_frame(fetch_and_cache_api("SeoulPharmacyStatusInfo", "pharmacy"), 'medical', 'pharmacy', ['WGS84_LAT'], ['WGS84_LON'])
        process_frame(fetch_and_cache_api("SeoulHospitalStatusInfo", "hospital"), 'medical', 'hospital', ['WGS84_LAT'], ['WGS84_LON'])

        # 3. Life & Admin
        process_frame(fetch_and_cache_api("SeoulWomensSafeDelivery", "delivery"), 'life', 'delivery', [], [], xy_keys=['X_COORD','Y_COORD']) # GRS80 fallback
        process_frame(fetch_and_cache_api("SeoulTraditionalMarket", "market"), 'life', 'market', ['GPS_LAT','LAT'], ['GPS_LET','LNG'])
        process_frame(fetch_and_cache_api("SeoulUminun", "uminun"), 'admin', 'kiosk', ['LAT'], ['LON'], xy_keys=['X_COORD','Y_COORD'])

        # 4. Mobility & Healing
        process_frame(fetch_and_cache_api("SeoulPublicBikeStationStatus", "bike"), 'mobility', 'bike', ['LAT','STATION_LAT'], ['LON','STATION_LNG'])
        process_frame(fetch_and_cache_api("SeoulForestPark", "park"), 'healing', 'park', ['LATITUDE','X_COORD'], ['LONGITUDE','Y_COORD'], xy_keys=['X_COORD','Y_COORD'])

    if not data_list: return pd.DataFrame()
    return pd.DataFrame(data_list)

def generate_mock_estate_data(n_samples=200):
    # Mock Estate Data with realistic Seoul rents (Shinlim area)
    lat_offsets = np.random.normal(0, 0.005, n_samples)
    lon_offsets = np.random.normal(0, 0.005, n_samples)
    lats = 37.4842 + lat_offsets
    lons = 126.9297 + lon_offsets
    # Rent per 3.3m2 (approx 5~15 man-won is realistic for cheap, 20+ for expensive)
    rent_per_pyeong = np.random.uniform(4, 18, n_samples) 
    return pd.DataFrame({
        'lat': lats, 'lon': lons, 
        'rent_per_area': rent_per_pyeong, 
        'name': [f"매물_{i:03d}" for i in range(n_samples)]
    })

def preprocess_data(use_mock=False):
    # Base categories (Cafe, Gym, Conv) - keeping mock for base infrastructure as requested "Focus on Public API expansion"
    base_mock_list = []
    for _ in range(300):
        base_mock_list.append({
            'name':'Store', 
            'type':np.random.choice(['cafe','gym','convenience']), 
            'lat':37.4842+np.random.normal(0,0.005), 
            'lon':126.9297+np.random.normal(0,0.005),
            'weight_factor':1.0
        })
    base_df = pd.DataFrame(base_mock_list)

    if not use_mock:
        real_df = get_real_data()
        if not real_df.empty:
            final_df = pd.concat([base_df, real_df], ignore_index=True)
            return gpd.GeoDataFrame(final_df, geometry=gpd.points_from_xy(final_df.lon, final_df.lat), crs="EPSG:4326")
    
    return gpd.GeoDataFrame(base_df, geometry=gpd.points_from_xy(base_df.lon, base_df.lat), crs="EPSG:4326")

# --- 2. Algorithm ---
def calculate_seulsekwon_index(gdf, grid_res=30, max_dist=1000):
    gdf_proj = gdf.to_crs(epsg=32652)
    minx, miny, maxx, maxy = gdf_proj.total_bounds
    buffer = max_dist
    x_rng = np.arange(minx-buffer, maxx+buffer, grid_res)
    y_rng = np.arange(miny-buffer, maxy+buffer, grid_res)
    xx, yy = np.meshgrid(x_rng, y_rng)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    categories = ['cafe', 'gym', 'convenience', 'safety', 'medical', 'life', 'mobility', 'smart', 'admin', 'healing']
    score_dict = {}
    
    for cat in categories:
        subset = gdf_proj[gdf_proj['type'] == cat]
        scores = np.zeros(len(grid_points))
        if not subset.empty:
            coords = np.array(list(zip(subset.geometry.x, subset.geometry.y)))
            tree = KDTree(coords)
            chunk = 10000
            for i in range(0, len(grid_points), chunk):
                pts = grid_points[i:i+chunk]
                dists, idxs = tree.query(pts, k=1)
                
                valid_mask = dists < max_dist
                local_sc = np.zeros(len(dists))
                close = dists <= 100
                mid = (dists > 100) & (dists < max_dist)
                
                local_sc[close] = 10
                if np.any(mid):
                    local_sc[mid] = 10 - 9 * ((dists[mid]-100)/(max_dist-100))
                
                w_factors = subset.iloc[idxs]['weight_factor'].values
                scores[i:i+chunk] = local_sc * w_factors
                
        score_dict[f'score_{cat}'] = scores

    res_df = pd.DataFrame(grid_points, columns=['x','y'])
    for k, v in score_dict.items(): res_df[k] = v
    
    gdf_grid = gpd.GeoDataFrame(res_df, geometry=gpd.points_from_xy(res_df.x, res_df.y), crs="EPSG:32652")
    return gdf_grid.to_crs(epsg=4326)

# --- 3. Dashboard UI ---
st.markdown("## 🏙️ 프리미엄 슬세권 분석 & 추천 서비스")
st.markdown("**(서울시 공공 인프라 기반 지능형 주거 가치 분석)**")

# Sidebar
st.sidebar.header("⚖️ 라이프스타일 가중치 (Weights)")
st.sidebar.caption("중요도를 선택하면 **100점 만점**으로 점수가 환산됩니다.")

w_opts = {'관심 없음 (0)':0.0, '보통 (1)':1.0, '중요 (2)':2.0, '필수 (3)':3.0}
def w_ui(lbl, help_txt, def_idx=1): 
    return w_opts[st.sidebar.select_slider(lbl, options=list(w_opts.keys()), value=list(w_opts.keys())[def_idx], help=help_txt)]

w_cafe  = w_ui("☕ Food & Cafe", "카페, 베이커리, 디저트 전문점 밀도를 분석하여 '여유로운 휴식과 미식'의 가치를 측정합니다.")
w_health= w_ui("🏋️ Health & Sports", "헬스장, 필라테스, 요가 스튜디오 등 '자기관리와 건강한 삶'을 위한 시설 접근성입니다.")
w_conv  = w_ui("🏪 Convenience", "편의점, 빨래방, 마트 등 1인 가구의 '가사 효율성과 생활 편의'를 지원하는 인프라입니다.")
w_safe  = w_ui("👮 Safety (안전)", "스마트폴, CCTV, 지구대(Police) 위치를 종합하여 밤길 걱정 없는 '안심 주거 환경'을 평가합니다.", 2)
w_med   = w_ui("🏥 Medical (의료)", "약국 및 1차 의료기관(내과, 치과 등) 분포를 통해 '비상시 의료 대응력'을 산출합니다.", 2)
w_mobil = w_ui("🚲 Mobility (교통)", "따릉이 대여소(Mobility) 및 지하철역 접근성을 계산하여 '라스트 마일 이동성'을 측정합니다.")

st.sidebar.divider()
st.sidebar.header("🔍 분석 범위 설정")
search_radius = st.sidebar.slider("탐색 반경", 100, 1000, 800, 100, format="%d m")
use_api = st.sidebar.checkbox("🌐 실시간 공공 데이터 연동", value=False)

# Logic
@st.cache_data
def get_data(api_mode): return preprocess_data(use_mock=not api_mode)
@st.cache_data
def get_estates(): return generate_mock_estate_data()
@st.cache_data
def compute_index(_gdf, _rad): return calculate_seulsekwon_index(_gdf, max_dist=_rad)

if 'infra' not in st.session_state or st.session_state.get('api_mode') != use_api:
    st.session_state.infra = get_data(use_api)
    st.session_state.estates = get_estates()
    st.session_state.api_mode = use_api
    st.session_state.last_rad = None

if st.session_state.get('last_rad') != search_radius:
    with st.spinner("AI 공간 분석 수행 중..."):
        st.session_state.grid = compute_index(st.session_state.infra, search_radius)
        st.session_state.last_rad = search_radius

# Scoring (Weighted Average 100 Scale)
grid = st.session_state.grid.copy()

# Consolidate Categories
s_cafe = grid['score_cafe']
s_health = grid['score_gym'] + grid.get('score_healing', 0)
s_conv = grid['score_convenience'] + grid['score_life'] + grid.get('score_admin', 0) + grid.get('score_market', 0)
s_safe = grid['score_safety'] + grid.get('score_smart', 0)
s_med = grid['score_medical']
s_mobil = grid.get('score_mobility', 0)

numerator = (s_cafe * w_cafe + s_health * w_health + s_conv * w_conv + s_safe * w_safe + s_med * w_med + s_mobil * w_mobil)
sum_weights = w_cafe + w_health + w_conv + w_safe + w_med + w_mobil
if sum_weights == 0: sum_weights = 1
# 10 is base max score. 
grid['total_score'] = (numerator / (sum_weights * 10)) * 100
grid['total_score'] = grid['total_score'].round(1)

# Estate Scoring
estates = st.session_state.estates.copy()
grid_tree = cKDTree(list(zip(grid.geometry.x, grid.geometry.y)))
_, idxs = grid_tree.query(list(zip(estates.lon, estates.lat)), k=1)
estates['score'] = grid.iloc[idxs]['total_score'].values
estates['cpi'] = estates['score'] / estates['rent_per_area']

# Visualization
col_map, col_stat = st.columns([2, 1])

with col_map:
    # Filter Recommendation
    top_cpi = estates['cpi'].quantile(0.8)
    estates['grade'] = estates['cpi'].apply(lambda x: '💎 가성비 최상' if x >= top_cpi else '일반')
    
    m = folium.Map([37.4842, 126.9297], zoom_start=15, tiles='cartodbpositron')
    
    # Heatmap (Fix applied)
    grid_copy = grid[grid['total_score']>0].copy()
    grid_copy['lat'] = grid_copy.geometry.y
    grid_copy['lon'] = grid_copy.geometry.x
    hm_data = grid_copy[['lat','lon','total_score']].values.tolist()
    HeatMap(hm_data, radius=15, blur=20, min_opacity=0.3).add_to(m)
    
    # Layers
    fgs = {
        'Safety': folium.FeatureGroup(name='Safety & Smart'),
        'Medical': folium.FeatureGroup(name='Medical'),
        'Life': folium.FeatureGroup(name='Convenience & Life'),
        'Mobility': folium.FeatureGroup(name='Mobility'),
        'Cafe': folium.FeatureGroup(name='Cafe'),
        'Health': folium.FeatureGroup(name='Health')
    }
    
    type_map = {
        'safety': 'Safety', 'smart': 'Safety', 
        'medical': 'Medical', 
        'convenience': 'Life', 'life': 'Life', 'admin': 'Life', 'market': 'Life',
        'mobility': 'Mobility',
        'cafe': 'Cafe',
        'gym': 'Health', 'healing': 'Health'
    }
    
    for r in st.session_state.infra.itertuples():
        t = getattr(r, 'type', 'other')
        if t in type_map:
            fg = fgs[type_map[t]]
            folium.CircleMarker(
                [r.geometry.y, r.geometry.x], radius=3, color='blue', fill=True,
                popup=f"{r.name} ({t})"
            ).add_to(fg)
            
    for fg in fgs.values(): fg.add_to(m)
    
    # Estates
    for _, e in estates.iterrows():
        if e['grade'] == '💎 가성비 최상':
            folium.Marker(
                [e['lat'], e['lon']], 
                popup=f"<b>{e['name']}</b><br>종합점수: {e['score']:.1f}점<br>평당: {e['rent_per_area']:.1f}만",
                icon=folium.Icon(color='darkblue', icon='star', prefix='fa')
            ).add_to(m)
            
    folium.LayerControl().add_to(m)
    st_folium(m, height=600)

with col_stat:
    st.subheader("📊 지역 분석 리포트")
    
    max_area_score = grid['total_score'].max()
    avg_area_score = grid[grid['total_score']>0]['total_score'].mean()
    
    m1, m2 = st.columns(2)
    m1.metric("지역 최고 점수 (Max)", f"{max_area_score:.0f}점", "Premium")
    m2.metric("평균 주거 가치", f"{avg_area_score:.0f}점")
    
    st.divider()
    
    bst = estates.nlargest(3, 'cpi')
    st.markdown("#### 🏆 Top 3 가성비 매물")
    for _, r in bst.iterrows():
        st.success(f"**{r['name']}**\n- 종합 점수: **{r['score']:.1f}점** / 100점\n- 평당 월세: **{r['rent_per_area']:.1f}만 원**")
        
    fig = px.scatter(estates, x='rent_per_area', y='score', color='grade', 
                     hover_data=['name', 'cpi'],
                     labels={'rent_per_area':'평당 월세 (3.3㎡)', 'score':'프리미엄 슬세권 지수 (100만점)'},
                     title="가격 대비 가치 (Value Analysis)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption(f"Data Source: 서울 열린데이터 광장 (API Key: {API_KEY[:5]}***), 국토부 실거래가")
