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
    
    # Initialize total score
    total_scores = np.zeros(len(grid_points))
    
    # Categories
    categories = ['cafe', 'gym', 'convenience']
    
    for cat in categories:
        subset = gdf_proj[gdf_proj['type'] == cat]
        if len(subset) == 0:
            continue
            
        data_points = np.array(list(zip(subset.geometry.x, subset.geometry.y)))
        
        # Build KDTree
        tree = KDTree(data_points)
        
        # Query: We only care about points within 1000m
        # query_ball_point matches all points within radius, but for scoring "density",
        # usually we want to know ANY valid point nearby or sum of impacts?
        # The prompt implies: "Summing [Cafe Score + Gym Score + Conv Score]".
        # And "Points within radius".
        # If there are MULTIPLE cafes, do we sum them? 
        # "100m 이내는 10점" -> Usually implies distance to the NEAREST, or sum of influences?
        # "Weighted sum of distances" usually implies Gravity Model.
        # But here it says "Score function S(d)". S(d) usually applies to a single relation.
        # "KDTree... find points in radius... score function".
        # If I have 10 cafes within 100m, is my score 100? Or just 10 (availability)?
        # "Target: Cafe, Gym, Convenience".
        # Usually Seulsekwon is about "Availability". 
        # But "Scoring" often implies summing multiple amenities.
        # Let's assume we sum the scores of ALL amenities within range to show "density" and "convenience".
        # However, finding ALL neighbors for ALL grid points can be heavy if dense.
        # Let's try 'query' for k=nearest first to see standard 'access' score?
        # Re-reading: "Summing [Cafe Score + Gym Score + Conv Score]".
        # It likely means the score for the CATEGORY.
        # Typically: Score(Cafe) = Score(Nearest Cafe) OR Sum(Score(All Cafes)).
        # Given "Real estate value" and "Infrastructure amount", Sum(Score(All)) makes sense for density.
        # But standard walkability is usually "Distance to nearest".
        # Let's implement Sum of Scores from ALL amenities within 1km.
        # This highlights "cluster" areas better than just "nearest".
        
        # To do this efficiently with KDTree for all grid points:
        # tree.query_ball_point(grid_points, r=1000) returns indices.
        
        print(f"Calculating scores for {cat}...")
        
        # Batch processing to avoid memory issues
        chunk_size = 10000
        category_scores = np.zeros(len(grid_points))
        
        for i in range(0, len(grid_points), chunk_size):
            chunk = grid_points[i:i+chunk_size]
            
            # Hybrid approach for speed/demo:
            # We use nearest 1 neighbor for simplicity in this demo as per "S(d)" singular function implication
            # Ideally for density we might sum multiple, but let's stick to Distance Decay to nearest facility
            # as it aligns well with "How far is the nearest X?"
            
            dists, _ = tree.query(chunk, k=1) # Nearest neighbor
            
            # Calculate score for these distances
            s = score_function(dists)
            category_scores[i:i+chunk_size] = s
            
        # Store individual category scores
        total_scores = total_scores + category_scores # Keep total for backward compatibility if needed, or just new cols
        
        # We will add columns dynamically
        
    # Create result dataframe
    result_df = pd.DataFrame(grid_points, columns=['x', 'y'])
    
    # Rerun logic to attach columns properly (iterating again or storing in dict above would be cleaner but let's just do it sequentially or fix logic)
    # Refactoring slightly for cleanliness:
    
    score_dict = {}
    
    for cat in categories:
        subset = gdf_proj[gdf_proj['type'] == cat]
        scores = np.zeros(len(grid_points))
        
        if len(subset) > 0:
            data_points = np.array(list(zip(subset.geometry.x, subset.geometry.y)))
            tree = KDTree(data_points)
            
            for i in range(0, len(grid_points), chunk_size):
                chunk = grid_points[i:i+chunk_size]
                dists, _ = tree.query(chunk, k=1)
                scores[i:i+chunk_size] = score_function(dists)
        
        score_dict[f'score_{cat}'] = scores

    # Combine into DF
    result_df = pd.DataFrame(grid_points, columns=['x', 'y'])
    for cat, s in score_dict.items():
        result_df[cat] = s
        
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
