import streamlit as st
import folium
from streamlit_folium import st_folium
from data_processor import preprocess_data
from algorithm import calculate_seulsekwon_index
from visualization import create_map

# Page Config
st.set_page_config(page_title="Seulsekwon Analysis", page_icon="🚶", layout="wide")

st.title("🚶 Seulsekwon Analysis (슬세권 분석)")
st.markdown("""
**"Slippers + Area" Analysis**  
Finding the best living areas for single-person households in Seoul.
This dashboard visualizes the density of **Cafes, Gyms, and Convenience Stores**.
""")

# Sidebar for controls
st.sidebar.header("Settings")
grid_res = st.sidebar.slider("Grid Resolution (meters)", 10, 50, 20)
center_lat_default = 37.4842
center_lon_default = 126.9297

# Main Logic
@st.cache_data
def load_and_process_data():
    # In a real app, this might load from a file or cloud storage
    # For now, we use the existing logic which falls back to mock data if file not found
    return preprocess_data(file_path='data/small_business_data.csv', use_mock=True)

with st.spinner('Loading data and calculating scores...'):
    gdf = load_and_process_data()
    
    # Recalculate if grid changes (though for simplicity we might want to cache this too)
    # We'll re-run algorithm if parameters change
    result_grid = calculate_seulsekwon_index(gdf, grid_res_meters=grid_res)

# Visualization
# We need to adapt create_map slightly or just use the logic here.
# The existing create_map saves to file. We want the map object.
# Let's import the logic or modify create_map to return the map object.
# For now, I'll recreate the map logic here to keep it simple for Streamlit
# or better, I will modify visualization.py to return the map object.

# Let's modify visualization.py first to be more flexible?
# Actually, for speed, I will just replicate the map creation here or import folium and make it.
# The existing create_map function creates a map and saves it. 
# It's better to refactor visualization.py to return the map object. 
# But to avoid touching too many files, I will just build the map here using the same logic.

st.subheader(f"Analysis Result (Grid: {grid_res}m)")

# Filter data
data = result_grid[result_grid['score'] > 0][['lat', 'lon', 'score']].values.tolist()

m = folium.Map(location=[center_lat_default, center_lon_default], zoom_start=15, tiles='cartodbpositron')

# Heatmap
from folium.plugins import HeatMap, MarkerCluster
HeatMap(data, radius=15, blur=20, max_zoom=1, min_opacity=0.4, name='Seulsekwon Heatmap').add_to(m)

# Marker Cluster
marker_cluster = MarkerCluster(name='Facilities').add_to(m)
icons = {
    'cafe': {'color': 'red', 'icon': 'coffee'},
    'gym': {'color': 'blue', 'icon': 'heart'},
    'convenience': {'color': 'green', 'icon': 'shopping-cart'}
}

# Add markers (Limit to reasonable number if too many for browser perf in Streamlit)
# Streamlit can handle it, but let's be safe.
count = 0
MAX_MARKERS = 2000 
for row in gdf.itertuples():
    if count > MAX_MARKERS:
        break
    lat = row.geometry.y
    lon = row.geometry.x
    ftype = getattr(row, 'type', 'unknown')
    store_name = getattr(row, '상호명', 'Store')
    style = icons.get(ftype, {'color': 'gray', 'icon': 'info-sign'})
    
    folium.Marker(
        location=[lat, lon],
        popup=f"<b>{store_name}</b><br>Type: {ftype}",
        icon=folium.Icon(color=style['color'], icon=style['icon'], prefix='fa')
    ).add_to(marker_cluster)
    count += 1

if count >= MAX_MARKERS:
    st.caption(f"Note: Only first {MAX_MARKERS} facilities shown for performance.")

folium.LayerControl().add_to(m)

# Render map
st_folium(m, width="100%", height=600)

st.success("Analysis Complete!")
