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
    
    # Tier Weights (Extreme Differentiation)
    rarity = {
        'cafe': 0.3, 'convenience': 0.3,  
        'life': 1.5, 'health': 2.0, 'mobility': 2.0,
        'medical': 6.0, 'safety': 6.0    
    }
    
    mapping = {
        'cafe': ['cafe'], 'convenience': ['convenience'],
        'health': ['gym', 'healing', 'park'],
        'mobility': ['mobility', 'bike'],
        'medical': ['medical', 'pharmacy', 'hospital'],
        'safety': ['safety', 'police', 'cctv', 'smartpole'],
        'life': ['life', 'market', 'admin']
    }
    
    score_dict = {}
    for cat_key, sub_types in mapping.items():
        subset = gdf_proj[gdf_proj['type'].isin(sub_types)]
        scores = np.zeros(len(grid_points))
        if not subset.empty:
            coords = np.array(list(zip(subset.geometry.x, subset.geometry.y)))
            tree = KDTree(coords)
            
            # Sharp Decay (150m lambda)
            lambda_k = 150.0 
            dists, _ = tree.query(grid_points, k=10)
            if dists.ndim == 1: dists = dists.reshape(-1, 1)
            
            # Exp Decay + Log Saturation
            raw_s = np.exp(-dists / lambda_k)
            cat_score = np.log1p(np.sum(raw_s, axis=1))
            
            # Normalize Category Score strictly to 0-1
            if cat_score.max() > cat_score.min():
                cat_score = (cat_score - cat_score.min()) / (cat_score.max() - cat_score.min())
            
            scores = cat_score * rarity.get(cat_key, 1.0)
                
        score_dict[f'score_{cat_key}'] = scores

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
st.sidebar.markdown("### ⚖️ 인프라 중요도 (Tier)")
w_opts = {k: v for v, k in enumerate([0.0, 1.0, 2.0, 3.0]) for k in [w_keys_list[v]]} 

def w_ui(lbl, help_txt, key, def_idx=1): 
    if key not in st.session_state: st.session_state[key] = w_keys_list[def_idx]
    val = st.sidebar.select_slider(lbl, options=w_keys_list, key=key, help=help_txt)
    return w_opts[val]

w_cafe  = w_ui("Tier 3: 편의/카페", "포화 상태 (0.5x) - 많아도 큰 이점 없음", 'k_cafe', 1)
w_conv  = w_ui("Tier 3: 마트/시장", "포화 상태 (0.5x)", 'k_conv', 1)
w_health= w_ui("Tier 2: 운동/산책", "삶의 질 (2.0x)", 'k_health', 1)
w_mobil = w_ui("Tier 2: 교통/이동", "삶의 질 (2.0x)", 'k_mobil', 2)
w_safe  = w_ui("Tier 1: 치안 (Safety)", "필수/희소 (5.0x) - 점수 결정적 요인", 'k_safe', 3)
w_med   = w_ui("Tier 1: 의료 (Medical)", "필수/희소 (5.0x) - 점수 결정적 요인", 'k_med', 3)

# Sidebar Formula & Legend
st.sidebar.divider()
st.sidebar.markdown("### 🧮 분석 방법론 (Exp Decay)")
st.sidebar.latex(r'''
S_{cat} = W_{tier} \cdot \ln\left(1 + \sum 10 \cdot e^{-\frac{dist}{200}}\right)
''')
st.sidebar.caption("지수 감쇠(300m↓) + Tier 가중치(5배) + 로그 포화")

# Debug Info
infra_count = len(st.session_state.infra) if 'infra' in st.session_state else 0
st.sidebar.caption(f"📡 불러온 인프라 데이터 수: {infra_count:,}개")

st.sidebar.markdown("### 🎨 범례 (Score Legend)")
st.sidebar.markdown(
    """
    <div style="background-color: rgba(255,255,255,0.05); padding: 10px; border-radius: 5px; border: 1px solid #444;">
        <div style="height: 15px; background: linear-gradient(to right, #edf8b1, #7fcdbb, #2c7fb8, #253494); border-radius: 3px; margin-bottom: 5px;"></div>
        <div style="display: flex; justify-content: space-between; font-size: 11px; color: #DDD;">
            <span>Basic (0)</span>
            <span>Premium (100)</span>
        </div>
        <div style="font-size: 10px; color: #888; margin-top:5px; text-align:center;">
           <span style="color:#edf8b1;">●</span> Low 
           <span style="color:#2c7fb8;">●</span> Good 
           <span style="color:#253494;">●</span> High
        </div>
    </div>
    """, unsafe_allow_html=True
)

st.sidebar.divider()
use_api = st.sidebar.checkbox("🌐 실시간 공공 데이터", value=False)
st.sidebar.markdown(
    """
    **📊 Data Sources**
    - **Commercial**: 소상공인시장진흥공단
    - **Public**: 서울시 열린광장 (CCTV, 스마트폴)
    - **Real Estate**: 국토교통부 실거래가
    """
)



# 2. Data Loading & Calc
@st.cache_data
def get_data(api_mode): return preprocess_data(use_mock=not api_mode)
@st.cache_data
def get_estates(): return generate_mock_estate_data()
@st.cache_data
def compute_index(_gdf, _rad, _version=1): return calculate_seulsekwon_index(_gdf, max_dist=_rad)

if 'infra' not in st.session_state or st.session_state.get('api_mode') != use_api:
    st.session_state.infra = get_data(use_api)
    st.session_state.estates = get_estates()
    st.session_state.api_mode = use_api
    st.session_state.last_rad = None
    
# 3. Main Layout
st.title("🏙️ 프리미엄 슬세권 분석 & 추천 서비스 (Pro)")

# Check if grid needs update (Radius change or Old Schema)
required_cols = ['score_health', 'score_mobility', 'score_life']
current_grid = st.session_state.get('grid', pd.DataFrame())
is_outdated = current_grid.empty or any(c not in current_grid.columns for c in required_cols)

if st.session_state.get('last_rad') != 800 or is_outdated:
    with st.spinner("AI 공간 분석 수행 중... (Algorithm Update v4)"):
        st.session_state.grid = compute_index(st.session_state.infra, 800, _version=4)
        st.session_state.last_rad = 800

# Aggregate Scores (Log-based + Rarity)
grid = st.session_state.grid.copy()
s_cafe = grid['score_cafe']
s_health = grid['score_health']
s_conv = grid['score_convenience']
s_life = grid['score_life']
s_safe = grid['score_safety']
s_med = grid['score_medical']
s_mobil = grid['score_mobility']

# Calculate Weighted Raw Score
raw_score = (
    s_cafe * w_cafe + 
    s_health * w_health + 
    (s_conv + s_life) * w_conv + 
    s_safe * w_safe + 
    s_med * w_med + 
    s_mobil * w_mobil
)

# Normalize to Percentile Rank (0-100)
grid['total_score'] = raw_score.rank(pct=True) * 100

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
    
    # Heatmap (Rank-based Gradient)
    g = grid[grid['total_score']>0].copy()
    g['lat'] = g.geometry.y
    g['lon'] = g.geometry.x
    hm_grad = {0.2: '#f7fbff', 0.5: '#abd9e9', 0.8: '#2c7fb8', 0.95: '#d73027'}
    HeatMap(g[['lat','lon','total_score']].values.tolist(), 
            radius=15, blur=20, min_opacity=0.2, max_zoom=13, gradient=hm_grad).add_to(m)

    # Helper
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000 
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    # Infra Markers
    mcs = {
        'Safety': MarkerCluster(name='Safety (안전)', overlay=True, control=True, show=True),
        'Medical': MarkerCluster(name='Medical (의료)', overlay=True, control=True, show=True),
        'Mobility': MarkerCluster(name='Mobility (교통)', overlay=True, control=True, show=True),
        'Life': MarkerCluster(name='Life (편의)', overlay=True, control=True, show=True),
        'Cafe': MarkerCluster(name='Cafe (카페)', overlay=True, control=True, show=True),
        'Health': MarkerCluster(name='Health (운동)', overlay=True, control=True, show=True)
    }
    
    cat_cfg = {
        'Cafe': {'icon':'coffee', 'color':'cadetblue'},
        'Health': {'icon':'heartbeat', 'color':'purple'},
        'Life': {'icon':'shopping-cart', 'color':'green'},
        'Safety': {'icon':'shield', 'color':'darkblue'},
        'Medical': {'icon':'medkit', 'color':'red'},
        'Mobility': {'icon':'bicycle', 'color':'orange'}
    }
    
    type_map = {
        'safety': 'Safety', 'smart': 'Safety', 'police':'Safety', 'cctv':'Safety', 'smartpole':'Safety',
        'medical': 'Medical', 'pharmacy':'Medical', 'hospital':'Medical',
        'mobility': 'Mobility', 'bike':'Mobility',
        'convenience': 'Life', 'life': 'Life', 'admin': 'Life', 'market': 'Life', 'delivery':'Life', 'kiosk':'Life',
        'cafe': 'Cafe',
        'gym': 'Health', 'healing': 'Health', 'park':'Health'
    }
    
    infra_df = st.session_state.infra
    
    # Robust KDTree
    e_coords = list(zip(estates.lat, estates.lon))
    e_tree = cKDTree(e_coords) if len(e_coords) > 0 else None
    
    for r in infra_df.itertuples():
        t = str(getattr(r, 'type', 'other')).lower()
        if t in type_map:
            try:
                if pd.isna(r.lat) or pd.isna(r.lon): continue
                
                cat = type_map[t]
                cfg = cat_cfg.get(cat, {'icon':'info-sign', 'color':'gray'})
                
                meta_txt = "반경 500m 추천 매물 없음"
                if e_tree:
                    dists, indices = e_tree.query([r.lat, r.lon], k=5)
                    if not isinstance(indices, (list, np.ndarray)): 
                        indices = [indices]
                        
                    candidates = []
                    for idx in indices:
                        if idx >= len(estates): continue
                        est = estates.iloc[idx]
                        d_m = haversine(est.lat, est.lon, r.lat, r.lon)
                        if d_m <= 500:
                            candidates.append((d_m, est['name'], max(1, int(d_m/80))))
                    
                    if candidates:
                        candidates.sort()
                        list_items = ""
                        # Renamed 'm' to 'mins' to avoid shadowing the Map object 'm'
                        for d, n, mins in candidates[:3]: # Top 3
                            list_items += f"<li style='margin-bottom:2px;'>{n}: {int(d)}m (약 {mins}분)</li>"
                        meta_txt = f"<b>🔎 주변 500m 연동:</b><ul style='margin:5px 0; padding-left:15px; font-size:11px;'>{list_items}</ul>"

                folium.Marker(
                    [r.lat, r.lon],
                    icon=folium.Icon(icon=cfg['icon'], color=cfg['color'], prefix='fa'),
                    popup=folium.Popup(f"""<div style="font-family:sans-serif; min-width:180px;"><b>{r.name}</b><br><span style='color:grey; font-size:12px;'>{cat}</span><hr style="margin:5px 0;">{meta_txt}</div>""", max_width=300)
                ).add_to(mcs[cat])
            except Exception:
                continue
            
    for mc in mcs.values(): mc.add_to(m)
    
    # Gold Stars
    top_10 = estates.nlargest(10, 'score')
    for _, e in top_10.iterrows():
        folium.Marker([e['lat'], e['lon']], 
            popup=f"<b>🏆 {e['name']}</b><br>Score: {e['score']:.1f}<br>CPI: {e['cpi']:.2f}", 
            icon=folium.Icon(color='orange', icon='star', prefix='fa', icon_color='white')).add_to(m)
            
    folium.LayerControl().add_to(m)
    
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
        'theta': ['👮 안전', '🏥 의료', '🚲 교통', '🏪 편의', '☕ 카페', '🏋️ 운동']
    })
    fig_r = px.line_polar(r_df, r='r', theta='theta', line_close=True)
    fig_r.update_traces(fill='toself', line_color='#FFD700')
    fig_r.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 3], showticklabels=True, tickfont=dict(color='gray', size=10)),
            bgcolor='rgba(255, 255, 255, 0.05)'
        ),
        font=dict(color='white', size=12),
        margin=dict(t=40, b=40, l=50, r=50),
        height=250
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
