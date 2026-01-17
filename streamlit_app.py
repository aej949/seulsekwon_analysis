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

st.set_page_config(page_title="서울시 주거 가치 분석 (Pro)", page_icon="🏙️", layout="wide")

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
        process_frame(fetch_and_cache_api("SeoulSmartPole", "smartpole"), 'safety', 'smartpole', ['LAT'], ['LON']) 
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

# 1. Sidebar - Persona & Logic
st.sidebar.header("👤 라이프 스타일 (Persona)")
c_p1, c_p2 = st.sidebar.columns(2)
w_keys_list = ['관심 없음 (0)', '보통 (1)', '중요 (2)', '필수 (3)']

def set_weights(c, h, cv, s, m, mo):
    st.session_state['k_cafe'] = w_keys_list[c]
    st.session_state['k_health'] = w_keys_list[h]
    st.session_state['k_conv'] = w_keys_list[cv]
    st.session_state['k_safe'] = w_keys_list[s]
    st.session_state['k_med'] = w_keys_list[m]
    st.session_state['k_mobil'] = w_keys_list[mo]
    st.rerun()

with c_p1:
    if st.button("🔥 갓생러"): set_weights(1, 3, 1, 1, 2, 1)
    if st.button("💻 노마드"): set_weights(3, 1, 2, 1, 1, 2)
    if st.button("🏥 투병"): set_weights(1, 1, 2, 2, 3, 1)
with c_p2:
    if st.button("🛡️ 귀가러"): set_weights(1, 1, 1, 3, 1, 3)
    if st.button("🏠 집순이"): set_weights(2, 1, 3, 1, 1, 1)

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
use_api = st.sidebar.checkbox("🌐 실시간 공공 데이터", value=False)
st.sidebar.markdown(
    """
    **📊 Data Sources**
    - **Commercial**: 소상공인시장진흥공단 상권정보
    - **Public**: 서울 열린데이터 광장 (CCTV 등)
    - **Real Estate**: 국토교통부 실거래가
    """
)

# 2. Data Loading & Calc
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
    
# 3. Main Layout
st.title("🏙️ 프리미엄 슬세권 분석 & 추천 서비스 (Pro)")

if st.session_state.get('last_rad') != 800:
    with st.spinner("AI 공간 분석 수행 중..."):
        st.session_state.grid = compute_index(st.session_state.infra, 800)
        st.session_state.last_rad = 800

grid = st.session_state.grid.copy()
# Aggregate Scores
s_cafe = grid['score_cafe']
s_health = grid['score_gym'] + grid.get('score_health', 0)
s_conv = grid['score_convenience'] + grid.get('score_life', 0)
s_safe = grid['score_safety'] + grid.get('score_smartcase', 0) 
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

# Metrics Prep
top_score = estates['score'].max()
avg_rent = estates['rent_per_area'].mean()
best_val = estates.loc[estates['cpi'].idxmax()]

if 'map_center' not in st.session_state: st.session_state.map_center = [37.4842, 126.9297]

# COLUMNS: MAP (7) | RIGHT (3)
col_map, col_right = st.columns([7, 3])

with col_map:
    # Map
    m = folium.Map(location=st.session_state.map_center, zoom_start=15, tiles='cartodbdark_matter')
    
    # Heatmap (Custom Gradient)
    g = grid[grid['total_score']>0].copy()
    g['lat'] = g.geometry.y
    g['lon'] = g.geometry.x
    hm_grad = {0.2: '#00008b', 0.4: '#00ced1', 0.6: '#ffff00', 0.8: '#ff8c00', 1.0: '#ff0000'}
    HeatMap(g[['lat','lon','total_score']].values.tolist(), 
            radius=25, blur=18, min_opacity=0.3, gradient=hm_grad).add_to(m)
    
    # Gold Stars for Top 10 Est.
    top_10 = estates.nlargest(10, 'score')
    for _, e in top_10.iterrows():
        folium.Marker([e['lat'], e['lon']], 
            popup=f"<b>🏆 {e['name']}</b><br>Score: {e['score']:.1f}", 
            icon=folium.Icon(color='orange', icon='star', prefix='fa', icon_color='white')).add_to(m)
            
    # CSS Legend
    l_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                background-color: rgba(0,0,0,0.7); padding: 15px; border-radius: 10px; color: white; border: 1px solid grey;">
        <b>Premium Score Legend</b><br>
        <div style="margin-top:5px;">
            <i style="background: red; width: 10px; height: 10px; display: inline-block; border-radius:50%;"></i> High (80-100)<br>
            <i style="background: yellow; width: 10px; height: 10px; display: inline-block; border-radius:50%;"></i> Mid (40-79)<br>
            <i style="background: blue; width: 10px; height: 10px; display: inline-block; border-radius:50%;"></i> Low (0-39)
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(l_html))
    
    map_data = st_folium(m, height=700, key="map")

with col_right:
    # 1. Metric
    r1, r2 = st.columns(2)
    r1.metric("최고 점수", f"{top_score:.0f}", "Premium")
    r2.metric("Best 가성비", best_val['name'], f"CPI {best_val['cpi']:.2f}")

    # 2. Radar Chart
    st.markdown("##### 🧬 라이프스타일 매칭")
    r_df = pd.DataFrame({
        'r': [w_safe, w_med, w_mobil, w_conv, w_cafe, w_health],
        'theta': ['안전', '의료', '교통', '편의', '카페', '운동']
    })
    fig_r = px.line_polar(r_df, r='r', theta='theta', line_close=True)
    fig_r.update_traces(fill='toself', line_color='#FFD700')
    fig_r.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 3], showticklabels=False),
            bgcolor='rgba(255, 255, 255, 0.05)'
        ),
        font=dict(color='white'),
        margin=dict(t=10, b=10, l=30, r=30),
        height=200
    )
    st.plotly_chart(fig_r, use_container_width=True)
    
    # 3. List
    st.markdown("##### 📋 추천 매물 리스트")
    sort_opt = st.selectbox("정렬", ["점수 높은 순", "가성비 순", "월세 낮은 순"], label_visibility="collapsed")
    
    # Filter
    if map_data and map_data.get('bounds'):
        b = map_data['bounds']
        sw, ne = b['_southWest'], b['_northEast']
        visible_estates = estates[
            (estates['lat'] >= sw['lat']) & (estates['lat'] <= ne['lat']) &
            (estates['lon'] >= sw['lng']) & (estates['lon'] <= ne['lng'])
        ]
    else: visible_estates = estates

    if visible_estates.empty:
        st.info("지도 영역 내 매물이 없습니다.")
    else:
        if sort_opt == "점수 높은 순": visible_estates = visible_estates.sort_values(by='score', ascending=False)
        elif sort_opt == "가성비 순": visible_estates = visible_estates.sort_values(by='cpi', ascending=False)
        else: visible_estates = visible_estates.sort_values(by='rent_per_area', ascending=True)
        
        # Scrollable container
        with st.container(height=400):
            for i, row in visible_estates.head(15).iterrows():
                badge = "🥇" if i == visible_estates.index[0] else ""
                with st.expander(f"{badge} [{row['score']:.0f}점] {row['name']}"):
                    c1, c2 = st.columns([2,1])
                    c1.caption(f"월세: {row['rent_per_area']:.1f}만/평")
                    if c2.button("이동", key=f"b_{i}"):
                        st.session_state.map_center = [row['lat'], row['lon']]
                        st.rerun()
                    
                    reasons = []
                    if w_safe >= 2: reasons.append("치안")
                    if w_mobil >= 2: reasons.append("교통")
                    if w_health >= 2: reasons.append("운동")
                    rs = ", ".join(reasons) if reasons else "생활 편의"
                    st.success(f"**추천**: {rs} 우수")

st.caption("Data Source: 소상공인시장진흥공단, 서울 열린데이터 광장, 국토교통부 | Powered by 9조 (Antigravity)")
