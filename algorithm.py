import numpy as np
from scipy.spatial import KDTree
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd

def score_function(dist):
    """
    Apply the linear decay scoring function.
    dist: distance in meters
    """
    # S(d) logic
    # Vectorized implementation for numpy arrays
    scores = np.zeros_like(dist)
    
    # Condition 1: d <= 100 -> 10
    mask_close = dist <= 100
    scores[mask_close] = 10
    
    # Condition 2: 100 < d < 1000
    mask_mid = (dist > 100) & (dist < 1000)
    # 10 - 9 * (d - 100) / 900
    scores[mask_mid] = 10 - 9 * ((dist[mask_mid] - 100) / 900)
    scores[mask_mid] = np.maximum(1, scores[mask_mid])
    
    # Condition 3: d >= 1000 -> 0 (already 0 init)
    
    return scores

def calculate_seulsekwon_index(gdf, grid_res_meters=20):
    """
    Calculates the Seulsekwon Index for the area covered by gdf.
    """
    print("Projecting data to UTM Zone 52N (EPSG:32652) for meter-based calculation...")
    
    # Project to UTM 52N (Seoul/Korea area)
    gdf_proj = gdf.to_crs(epsg=32652)
    
    # Create Meshgrid
    minx, miny, maxx, maxy = gdf_proj.total_bounds
    # Add buffer
    buffer = 1000
    minx -= buffer
    miny -= buffer
    maxx += buffer
    maxy += buffer
    
    x_range = np.arange(minx, maxx, grid_res_meters)
    y_range = np.arange(miny, maxy, grid_res_meters)
    
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    print(f"Generated grid with {len(grid_points)} points.")
    
    # Categories for Advanced Analysis
    categories = ['cafe', 'gym', 'convenience', 'safety', 'medical', 'life']
    
    score_dict = {}
    chunk_size = 10000
    
    for cat in categories:
        subset = gdf_proj[gdf_proj['type'] == cat]
        scores = np.zeros(len(grid_points))
        
        if len(subset) > 0:
            print(f"Calculating scores for {cat}...")
            data_points = np.array(list(zip(subset.geometry.x, subset.geometry.y)))
            tree = KDTree(data_points)
            
            for i in range(0, len(grid_points), chunk_size):
                chunk = grid_points[i:i+chunk_size]
                dists, _ = tree.query(chunk, k=1) # Nearest neighbor
                scores[i:i+chunk_size] = score_function(dists)
        else:
            print(f"No data for {cat}, skipping...")
        
        score_dict[f'score_{cat}'] = scores

    # Combine into DF
    result_df = pd.DataFrame(grid_points, columns=['x', 'y'])
    for key, s in score_dict.items():
        result_df[key] = s
        
    # Calculate default total (equal weights)
    result_df['score'] = result_df[[c for c in result_df.columns if 'score_' in c]].sum(axis=1)
    
    # Convert back to Lat/Lon for visualization (EPSG:4326)
    gdf_grid = gpd.GeoDataFrame(
        result_df,
        geometry=gpd.points_from_xy(result_df['x'], result_df['y']),
        crs="EPSG:32652"
    )
    
    gdf_grid = gdf_grid.to_crs(epsg=4326)
    
    # Extract lat/lon for folium
    gdf_grid['lat'] = gdf_grid.geometry.y
    gdf_grid['lon'] = gdf_grid.geometry.x
    
    return gdf_grid
