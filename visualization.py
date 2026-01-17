import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd

def create_map(grid_gdf, facilities_gdf, center_lat=37.4842, center_lon=126.9297, output_file='seulsekwon_map.html'):
    """
    Generates a Folium map with a heatmap of the Seulsekwon Index and clustered facility markers.
    """
    print("Generating visualization...")
    
    # Filter out zero scores for better performance/visualization
    data = grid_gdf[grid_gdf['score'] > 0][['lat', 'lon', 'score']].values.tolist()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles='cartodbpositron')
    
    # 1. HeatMap (Density of preferred combination)
    # Radius: adjust based on grid density. 20m grid -> maybe 15-20 pixels?
    HeatMap(data, radius=15, blur=20, max_zoom=1, min_opacity=0.4, name='Seulsekwon Heatmap').add_to(m)
    
    # 2. Marker Clustering (Facilities)
    marker_cluster = MarkerCluster(name='Facilities').add_to(m)
    
    # Define icons/colors for categories
    icons = {
        'cafe': {'color': 'red', 'icon': 'coffee'},
        'gym': {'color': 'blue', 'icon': 'heart'}, # heart or star
        'convenience': {'color': 'green', 'icon': 'shopping-cart'} # shopping-cart
    }
    
    # We iterate over facilities_gdf to add them to the cluster
    # Using itertuples for speed
    for row in facilities_gdf.itertuples():
        # geometry in itertuples is row.geometry
        lat = row.geometry.y
        lon = row.geometry.x
        ftype = getattr(row, 'type', 'unknown') # safely get type
        store_name = getattr(row, '상호명', 'Store')
        
        style = icons.get(ftype, {'color': 'gray', 'icon': 'info-sign'})
        
        folium.Marker(
            location=[lat, lon],
            popup=f"<b>{store_name}</b><br>Type: {ftype}",
            icon=folium.Icon(color=style['color'], icon=style['icon'], prefix='fa')
        ).add_to(marker_cluster)
    
    # Add Layer Control to toggle layers
    folium.LayerControl().add_to(m)
    
    # Add a title (using standard HTML)
    title_html = '''
             <h3 align="center" style="font-size:16px"><b>Seulsekwon Analysis (Sillim-dong)</b></h3>
             '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    m.save(output_file)
    print(f"Map saved to {output_file}")
