import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import geopandas as gpd
from folium.plugins import HeatMap, MarkerCluster
import altair as alt
import numpy as np
from scipy.spatial import cKDTree

# Import local modules
from data_processor import preprocess_data, generate_mock_estate_data
from algorithm import calculate_seulsekwon_index

# Page Settings
st.set_page_config(page_title="Seulsekwon Analysis", page_icon="🚶", layout="wide")

st.markdown("""
# 🚶 **1인 가구 맞춤형 슬세권 분석 & 매물 가치 평가**
**"Slippers + Area" Analysis Dashboard**
서울시 1인 가구 밀집 지역(신림동)을 대상으로 **인프라 점수(슬세권 지수)**를 산출하고, 
부동산 실거래가와 결합하여 **'가성비 매물'**을 발굴합니다.
""")

# --- Sidebar Controls ---
st.sidebar.header("🛠️ Analysis Weights")
st.sidebar.caption("개인 선호도에 따라 가중치를 조절하세요.")

w_cafe = st.sidebar.slider("☕ Cafe Weight", 0.0, 3.0, 1.0, 0.1)
w_gym = st.sidebar.slider("💪 Fitness Weight", 0.0, 3.0, 1.0, 0.1)
w_conv = st.sidebar.slider("🏪 Convenience Weight", 0.0, 3.0, 1.0, 0.1)

st.sidebar.divider()
st.sidebar.header("⚙️ Grid Settings")
grid_res = st.sidebar.slider("Grid Resolution (m)", 20, 100, 30)

# --- Data Loading (Cached) ---
@st.cache_data
def load_infrastructure():
    return preprocess_data(file_path='data/small_business_data.csv', use_mock=True)

@st.cache_data
def load_real_estate():
    # Mock data for demonstration
    return generate_mock_estate_data(n_samples=150)

@st.cache_data
def calculate_base_scores(_gdf, resolution):
    # This is the heavy calculation (KDTree)
    # Returns DataFrame with 'score_cafe', 'score_gym', 'score_conv' separated
    return calculate_seulsekwon_index(_gdf, grid_res_meters=resolution)

# Load Data
with st.spinner('Loading Data & Calculating Base Scores...'):
    infra_gdf = load_infrastructure()
    estate_df = load_real_estate()
    grid_gdf = calculate_base_scores(infra_gdf, grid_res)

# --- Dynamic Scoring (Fast) ---
# Calculate Weighted Total Score
grid_gdf['total_score'] = (
    grid_gdf['score_cafe'] * w_cafe + 
    grid_gdf['score_gym'] * w_gym + 
    grid_gdf['score_convenience'] * w_conv
)

# --- Analysis: Assign Score to Real Estate Listings ---
# For each estate, find the score of the nearest grid point
# This is a quick lookup
grid_coords = list(zip(grid_gdf.geometry.x, grid_gdf.geometry.y))
grid_tree = cKDTree(grid_coords)

estate_coords = list(zip(estate_df['lon'], estate_df['lat']))
dists, idxs = grid_tree.query(estate_coords, k=1)

estate_df['seulsekwon_score'] = grid_gdf.iloc[idxs]['total_score'].values

# Identify "Undervalued" properties
# Simple logic: High Score, Low Rent
# We divide into quadrants based on Median
median_score = estate_df['seulsekwon_score'].median()
median_rent = estate_df['rent_per_area'].median()

def classify_value(row):
    # Avoid zero score which messes up logic
    if row['seulsekwon_score'] < 1:
        return 'No Data'
        
    if row['seulsekwon_score'] >= median_score and row['rent_per_area'] < median_rent:
        return '💎 Undervalued (Best Value)'
    elif row['seulsekwon_score'] >= median_score and row['rent_per_area'] >= median_rent:
        return '💰 High Value, High Price'
    elif row['seulsekwon_score'] < median_score and row['rent_per_area'] < median_rent:
        return '📉 Low Price'
    else:
        return '⚠️ Overpriced'

estate_df['category'] = estate_df.apply(classify_value, axis=1)

# --- Visualization ---

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("🗺️ Seulsekwon Heatmap & Listings")
    
    # Base Map
    mean_lat, mean_lon = infra_gdf.geometry.y.mean(), infra_gdf.geometry.x.mean()
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=15, tiles='cartodbpositron')
    
    # 1. Heatmap (Weighted Score)
    heat_data = grid_gdf[grid_gdf['total_score'] > 0][['lat', 'lon', 'total_score']].values.tolist()
    HeatMap(heat_data, radius=15, blur=20, min_opacity=0.3, max_zoom=1).add_to(m)
    
    # 2. Real Estate Markers
    # Color code by category
    color_map = {
        '💎 Undervalued (Best Value)': 'blue',
        '💰 High Value, High Price': 'orange',
        '📉 Low Price': 'gray',
        '⚠️ Overpriced': 'red',
        'No Data': 'black'
    }
    
    for idx, row in estate_df.iterrows():
        color = color_map.get(row['category'], 'black')
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=f"<b>{row['name']}</b><br>Score: {row['seulsekwon_score']:.1f}<br>Rent: {row['rent_per_area']:.1f}<br>{row['category']}"
        ).add_to(m)
        
    st_folium(m, width="100%", height=600)

with col2:
    st.subheader("📊 Business Insights")
    
    # correlation
    corr = estate_df['seulsekwon_score'].corr(estate_df['rent_per_area'])
    st.info(f"💡 Correlation between **Score** and **Rent**: **{corr:.2f}**")
    
    # 3. Scatter Plot (Interactive)
    # Using Altair
    scatter = alt.Chart(estate_df).mark_circle(size=80).encode(
        x=alt.X('seulsekwon_score', title='Seulsekwon Index (Score)'),
        y=alt.Y('rent_per_area', title='Rent per Area (Simulated)'),
        color=alt.Color('category', legend=alt.Legend(title="Evaluation")),
        tooltip=['name', 'seulsekwon_score', 'rent_per_area', 'category', 'deposit']
    ).interactive()
    
    st.altair_chart(scatter, use_container_width=True)
    
    # Radar Chart / Bar Chart for Infrastructure
    st.markdown("### 🕸️ Infrastructure Breakdown")
    st.caption("Balance of amenities for the top scoring location.")
    
    # Example: Top Score Listing
    if not estate_df.empty:
        top_listing = estate_df.loc[estate_df['seulsekwon_score'].idxmax()]
        
        # Radar Data
        categories = ['Cafe', 'Gym', 'Convenience']
        # Use Nearest Neighbor to find breakdown for this specific point
        dist, specific_idx = grid_tree.query([[top_listing['lon'], top_listing['lat']]])
        specific_grid_point = grid_gdf.iloc[specific_idx[0]]
        
        radar_data = pd.DataFrame({
            'Category': categories,
            'Score': [
                specific_grid_point['score_cafe'], 
                specific_grid_point['score_gym'], 
                specific_grid_point['score_convenience']
            ]
        })
        
        # Simple Bar Chart as Radar is tricky in pure Altair without polar
        bar = alt.Chart(radar_data).mark_bar().encode(
            x='Category',
            y='Score',
            color='Category'
        ).properties(title=f"Best Listing: {top_listing['name']}")
        st.altair_chart(bar, use_container_width=True)

st.success("Real-time Analysis Complete!")
