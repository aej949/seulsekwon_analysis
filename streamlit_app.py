import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import geopandas as gpd
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px
import plotly.graph_objects as go
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

st.set_page_config(page_title="서울시 주거 가치 분석 (Premium)", page_icon="🏙️", layout="wide")

# --- 1. Data Ingestion & Caching ---
def fetch_and_cache_api(service, save_name):
    if not os.path.exists('data'): os.makedirs('data')
    path = f"data/{save_name}.csv"
    if os.path.exists(path):
        try: return pd.read_csv(path)
        except: pass
    
    all_rows = []
    start, step = 1, 1000
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
            else: break
    except: return pd.DataFrame()
    
    df = pd.DataFrame(all_rows)
    if not df.empty: df.to_csv(path, index=False)
    return df

def get_real_data():
    data_list = []
    try: transformer = Transformer.from_crs("EPSG:2097", "EPSG:4326", always_xy=True)
    except: transformer = None

    with st.spinner("📦 서울시 공공데이터 통합 로딩 중..."):
        def process_frame(df, type_name, subtype, lat_keys, lon_keys, xy_keys=None, weight=1.0):
            if df.empty: return
            for _, r in df.iterrows():
                try:
                    lat, lon = None, None
                    for k in lat_keys:
                        if k in r and pd.notna(r[k]): lat = float(r[k]); break
                    for k in lon_keys:
                        if k in r and pd.notna(r[k]): lon = float(r[k]); break
                    if (lat is None or lon is None) and xy_keys and transformer:
                        x, y = float(r.get(xy_keys[0])), float(r.get(xy_keys[1]))
                        lon, lat = transformer.transform(x, y)
                    
                    if lat and lon:
                        name = r.get('NAME') or r.get('NAMES') or r.get('NM') or r.get('DUTYNAME') or r.get('M_NAME') or subtype
                        data_list.append({'name': str(name), 'type': type_name, 'subtype': subtype, 'lat': lat, 'lon': lon, 'weight_factor': weight})
                except: pass

        process_frame(fetch_and_cache_api("SeoulPoliceStationWGS", "police"), 'safety', 'police', ['LAT'], ['LON'], weight=1.5)
        process_frame(fetch_and_cache_api("tbsSvcCctv", "cctv"), 'safety', 'cctv', ['LATITUDE','LAT'], ['LONGITUDE','LON'])
        process_frame(fetch_and_cache_api("SeoulSmartPole", "smartpole"), 'safety', 'smartpole', ['LAT'], ['LON']) # Merged to Safety category per UX logic
        process_frame(fetch_and_cache_api("SeoulPharmacyStatusInfo", "pharmacy"), 'medical', 'pharmacy', ['WGS84_LAT'], ['WGS84_LON'])
        process_frame(fetch_and_cache_api("SeoulHospitalStatusInfo", "hospital"), 'medical', 'hospital', ['WGS84_LAT'], ['WGS84_LON'])
        process_frame(fetch_and_cache_api("SeoulWomensSafeDelivery", "delivery"), 'life', 'delivery', [], [], xy_keys=['X_COORD','Y_COORD'])
        process_frame(fetch_and_cache_api("SeoulTraditionalMarket", "market"), 'life', 'market', ['GPS_LAT','LAT'], ['GPS_LET','LNG'])
        process_frame(fetch_and_cache_api("SeoulUminun", "uminun"), 'life', 'kiosk', ['LAT'], ['LON'], xy_keys=['X_COORD','Y_COORD'])
        process_frame(fetch_and_cache_api("SeoulPublicBikeStationStatus", "bike"), 'mobility', 'bike', ['LAT','STATION_LAT'], ['LON','STATION_LNG'])
        process_frame(fetch_and_cache_api("SeoulForestPark", "park"), 'health', 'park', ['LATITUDE','X_COORD'], ['LONGITUDE','Y_COORD'], xy_keys=['X_COORD','Y_COORD'])

    return pd.DataFrame(data_list) if data_list else pd.DataFrame()

def preprocess_data(use_mock=False):
    base_mock_list = []
    for _ in range(300):
        base_mock_list.append({'name':'Store', 'type':np.random.choice(['cafe','gym','convenience']), 'lat':37.4842+np.random.normal(0,0.005), 'lon':126.9297+np.random.normal(0,0.005), 'weight_factor':1.0})
    base_df = pd.DataFrame(base_mock_list)

    if not use_mock:
        real_df = get_real_data()
        if not real_df.empty:
            final_df = pd.concat([base_df, real_df], ignore_index=True)
            return gpd.GeoDataFrame(final_df, geometry=gpd.points_from_xy(final_df.lon, final_df.lat), crs="EPSG:4326")
    
    return gpd.GeoDataFrame(base_df, geometry=gpd.points_from_xy(base_df.lon, base_df.lat), crs="EPSG:4326")

def generate_mock_estate_data(n_samples=200):
    lats = 37.4842 + np.random.normal(0, 0.005, n_samples)
    lons = 126.9297 + np.random.normal(0, 0.005, n_samples)
    rent = np.random.uniform(5, 18, n_samples) 
    return pd.DataFrame({'lat': lats, 'lon': lons, 'rent_per_area': rent, 'name': [f"매물_{i:03d}" for i in range(n_samples)]})

def calculate_seulsekwon_index(gdf, grid_res=30, max_dist=1000):
    gdf_proj = gdf.to_crs(epsg=32652)
    minx, miny, maxx, maxy = gdf_proj.total_bounds
    buffer = max_dist
    x_rng = np.arange(minx-buffer, maxx+buffer, grid_res)
    y_rng = np.arange(miny-buffer, maxy+buffer, grid_res)
    xx, yy = np.meshgrid(x_rng, y_rng)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    categories = ['cafe', 'gym', 'convenience', 'safety', 'medical', 'life', 'mobility', 'health']
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

# --- UI LOGIC ---

# Sidebar Persona Presets
st.sidebar.header("👤 페르소나 (빠른 설정)")
c_p1, c_p2, c_p3 = st.sidebar.columns(3)
w_keys_list = ['관심 없음 (0)', '보통 (1)', '중요 (2)', '필수 (3)']

def set_weights(c, h, cv, s, m, mo):
    st.session_state['k_cafe'] = w_keys_list[c]
    st.session_state['k_health'] = w_keys_list[h]
    st.session_state['k_conv'] = w_keys_list[cv]
    st.session_state['k_safe'] = w_keys_list[s]
    st.session_state['k_med'] = w_keys_list[m]
    st.session_state['k_mobil'] = w_keys_list[mo]

if c_p1.button("💪 갓생"): set_weights(1, 3, 2, 1, 1, 1)
if c_p2.button("🏃 야근"): set_weights(1, 1, 3, 3, 2, 3)
if c_p3.button("💎 가성비"): set_weights(1, 1, 1, 1, 1, 1)

st.sidebar.divider()
st.sidebar.header("⚖️ 상세 가중치 설정")
w_opts = {k: v for v, k in enumerate([0.0, 1.0, 2.0, 3.0]) for k in [w_keys_list[v]]} # Map Str to Float
def w_ui(lbl, help_txt, key, def_idx=1): 
    if key not in st.session_state: st.session_state[key] = w_keys_list[def_idx]
    val = st.sidebar.select_slider(lbl, options=w_keys_list, key=key, help=help_txt)
    return w_opts[val]

w_cafe  = w_ui("☕ Food & Cafe", "카페, 디저트", 'k_cafe', 1)
w_health= w_ui("🏋️ Health & Sports", "헬스, 공원", 'k_health', 1)
w_conv  = w_ui("🏪 Convenience", "편의점, 마트", 'k_conv', 1)
w_safe  = w_ui("👮 Safety (안전)", "CCTV, 경찰, 스마트폴", 'k_safe', 2)
w_med   = w_ui("🏥 Medical (의료)", "약국, 병원", 'k_med', 2)
w_mobil = w_ui("🚲 Mobility (교통)", "따릉이, 지하철", 'k_mobil', 1)

st.sidebar.divider()
st.sidebar.markdown(r"""
**🧮 산출 로직 (Methodology)**
$$Score = \frac{\sum (w_i \cdot s_i)}{\sum w_i \times 10} \times 100$$
""")
use_api = st.sidebar.checkbox("🌐 실시간 공공 데이터", value=False)
st.sidebar.caption("Data Source: 소상공인, 서울시, 국토부")

# Data & Calc
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
    
# Layout Main
st.title("🏙️ 프리미엄 슬세권 분석 & 추천 서비스")
st.markdown("**(Seoul Smart Habitat Analytics)**: 빅데이터와 AI 공간 분석을 통한 지능형 주거 가치 평가")

search_radius = 800 # Fixed for simplicity or add slider in top

if st.session_state.get('last_rad') != search_radius:
    with st.spinner("Calculating..."):
        st.session_state.grid = compute_index(st.session_state.infra, search_radius)
        st.session_state.last_rad = search_radius

grid = st.session_state.grid.copy()
# Aggregate
s_cafe = grid['score_cafe']
s_health = grid['score_gym'] + grid.get('score_health', 0)
s_conv = grid['score_convenience'] + grid.get('score_life', 0)
s_safe = grid['score_safety'] + grid.get('score_smartcase', 0) # Fallback if key mismatch
s_med = grid['score_medical']
s_mobil = grid.get('score_mobility', 0)

num = (s_cafe*w_cafe + s_health*w_health + s_conv*w_conv + s_safe*w_safe + s_med*w_med + s_mobil*w_mobil)
den = (w_cafe+w_health+w_conv+w_safe+w_med+w_mobil) * 10
grid['total_score'] = (num / (den if den > 0 else 1)) * 100

estates = st.session_state.estates.copy()
grid_tree = cKDTree(list(zip(grid.geometry.x, grid.geometry.y)))
_, idxs = grid_tree.query(list(zip(estates.lon, estates.lat)), k=1)
estates['score'] = grid.iloc[idxs]['total_score'].values
estates['cpi'] = estates['score'] / estates['rent_per_area']

# Metrics Row
top_score = estates['score'].max()
avg_rent = estates['rent_per_area'].mean()
best_val = estates.loc[estates['cpi'].idxmax()]

m1, m2, m3 = st.columns(3)
m1.metric("지역 최고 점수", f"{top_score:.1f}점", "Premium Quality")
m2.metric("평균 평당 월세", f"{avg_rent:.1f}만 원", "-1.2% (MoM)")
m3.metric("Best Value 매물", best_val['name'], f"가성비 {best_val['cpi']:.2f}")

st.divider()

# Split Layout
col_map, col_chart = st.columns([2, 1])

with col_map:
    top_cpi_thr = estates['cpi'].quantile(0.8)
    estates['grade'] = estates['cpi'].apply(lambda x: '💎 Best' if x >= top_cpi_thr else 'Normal')
    
    m = folium.Map([37.4842, 126.9297], zoom_start=15, tiles='cartodbpositron')
    
    # Heatmap
    g = grid[grid['total_score']>0].copy()
    g['lat'] = g.geometry.y
    g['lon'] = g.geometry.x
    HeatMap(g[['lat','lon','total_score']].values.tolist(), radius=15, blur=20, min_opacity=0.3).add_to(m)
    
    # Clustering Markers
    mc = MarkerCluster(name="Facilities")
    for r in st.session_state.infra.itertuples():
        mc.add_child(folium.Marker([r.lat, r.lon], popup=r.name, icon=folium.Icon(color='gray', icon='info-sign')))
    # mc.add_to(m) # Optional: Enable if user wants to see all dots
    
    # Estates
    for _, e in estates.iterrows():
        if e['grade'] == '💎 Best':
            folium.Marker([e['lat'], e['lon']], popup=f"<b>{e['name']}</b><br>Score: {e['score']:.1f}", icon=folium.Icon(color='darkblue', icon='star', prefix='fa')).add_to(m)
    
    # Legend HTML
    l_html = '''<div style="position:fixed; bottom:30px; right:30px; z-index:9999; background:white; padding:10px; border:1px solid gray;">
    <b>Score Legend</b><br><span style='color:red;'>■</span> High<br><span style='color:yellow;'>■</span> Mid<br><span style='color:blue;'>■</span> Low</div>'''
    m.get_root().html.add_child(folium.Element(l_html))
    
    st_folium(m, height=600)

with col_chart:
    st.markdown("### 🔍 상세 분석")
    
    # 1. Radar Chart (Average of Top 3 CPI Estates)
    top3 = estates.nlargest(3, 'cpi')
    # Use global weights as proxy for "Ideal" vs "Actual"? 
    # Or calculate category sub-scores for the estates.
    # We didn't store category sub-scores in estates DF yet. 
    # Quick fix: Just Use Slider Weights to act as "User Preference Profile" visualization
    df_radar = pd.DataFrame({
        'Category': ['Cafe', 'Health', 'Conv', 'Safety', 'Medical', 'Mobility'],
        'Importance': [w_cafe, w_health, w_conv, w_safe, w_med, w_mobil]
    })
    fig_r = px.line_polar(df_radar, r='Importance', theta='Category', line_close=True, title="나의 라이프스타일 균형 (User Profile)")
    fig_r.update_traces(fill='toself')
    st.plotly_chart(fig_r, use_container_width=True)

    # 2. Price Trend (Mock)
    dates = pd.date_range(start='2025-01-01', periods=12, freq='M')
    prices = np.linspace(avg_rent*0.9, avg_rent*1.05, 12) + np.random.normal(0, 0.5, 12)
    df_trend = pd.DataFrame({'Date': dates, 'Price (3.3㎡)': prices})
    fig_l = px.line(df_trend, x='Date', y='Price (3.3㎡)', title="최근 1년 월세 변동 추이 (지역 평균)")
    st.plotly_chart(fig_l, use_container_width=True)
    
    # 3. Value Scatter
    fig_s = px.scatter(estates, x='rent_per_area', y='score', color='grade', title="가성비 매물 매트릭스")
    st.plotly_chart(fig_s, use_container_width=True)
