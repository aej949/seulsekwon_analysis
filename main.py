import os
from data_processor import preprocess_data
from algorithm import calculate_seulsekwon_index
from visualization import create_map

def main():
    print("Starting Seulsekwon Analysis Workflow...")
    
    # 1. Load and Preprocess Data
    # In a real scenario, user would provide a path. Here we default to mock if none.
    data_path = 'data/small_business_data.csv' # Placeholder path
    
    # Check if file exists, else use mock
    use_mock = not os.path.exists(data_path)
    
    gdf = preprocess_data(file_path=data_path, use_mock=use_mock)
    
    if gdf.empty:
        print("No data found. Exiting.")
        return

    # 2. Algorithm
    # Sillim-dong center (approx) used for map visualization later
    center_lat = gdf.geometry.y.mean()
    center_lon = gdf.geometry.x.mean()
    
    result_grid = calculate_seulsekwon_index(gdf)
    
    # 3. Visualization
    create_map(result_grid, gdf, center_lat, center_lon, output_file='seulsekwon_map.html')
    
    print("Process Complete!")

if __name__ == "__main__":
    main()
