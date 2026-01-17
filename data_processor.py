import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

def generate_mock_data(center_lat=37.4842, center_lon=126.9297, n_samples=300):
    """
    Generates mock data for testing if real data is not available.
    Sillim-dong approximate center: 37.4842, 126.9297
    """
    print("Generating mock data...")
    
    # Generate random offsets
    lat_offsets = np.random.normal(0, 0.005, n_samples)
    lon_offsets = np.random.normal(0, 0.005, n_samples)
    
    lats = center_lat + lat_offsets
    lons = center_lon + lon_offsets
    
    categories = []
    names = []
    
    # Randomly assign categories
    types = ['Cafe', 'Gym', 'Convenience', 'Other']
    for _ in range(n_samples):
        choice = np.random.choice(types, p=[0.4, 0.2, 0.3, 0.1])
        cat_name = ''
        store_name = ''
        
        if choice == 'Cafe':
            cat_name = '카페'
            store_name = 'Mega Coffee'
        elif choice == 'Gym':
            cat_name = '헬스'
            store_name = 'Sillim Gym'
        elif choice == 'Convenience':
            cat_name = '편의점'
            store_name = 'GS25'
        else:
            cat_name = '기타'
            store_name = 'Unknown'
            
        categories.append(cat_name)
        names.append(store_name)
        
    df = pd.DataFrame({
        '상호명': names,
        '상권업종소분류명': categories,
        '위도': lats,
        '경도': lons
    })
    
    return df

def preprocess_data(file_path=None, use_mock=False):
    """
    Loads and filters data.
    """
    if use_mock or not file_path:
        df = generate_mock_data()
    else:
        try:
            # Assuming CSV format common in public data
            df = pd.read_csv(file_path, encoding='cp949') # Common encoding for Korean Windows csv
        except:
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except FileNotFoundError:
                print(f"File not found: {file_path}. Switching to mock data.")
                df = generate_mock_data()

    # Filter targets
    # Cafe: '카페', '커피'
    cond_cafe = df['상권업종소분류명'].astype(str).str.contains('카페|커피', na=False)
    
    # Gym: '헬스', '필라테스', '요가'
    cond_gym = df['상권업종소분류명'].astype(str).str.contains('헬스|필라테스|요가', na=False)
    
    # Convenience: '편의점', '약국' in category OR '다이소', '올리브영' in name
    cond_conv_cat = df['상권업종소분류명'].astype(str).str.contains('편의점|약국', na=False)
    cond_conv_name = df['상호명'].astype(str).str.contains('다이소|올리브영', na=False)
    cond_conv = cond_conv_cat | cond_conv_name
    
    # Labeling
    df.loc[cond_cafe, 'type'] = 'cafe'
    df.loc[cond_gym, 'type'] = 'gym'
    df.loc[cond_conv, 'type'] = 'convenience'
    
    # Filter only selected
    target_df = df[df['type'].notna()].copy()
    
    print(f"Filtered {len(target_df)} target locations from {len(df)} total.")
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        target_df, 
        geometry=gpd.points_from_xy(target_df['경도'], target_df['위도']),
        crs="EPSG:4326"
    )
    
    return gdf

def generate_mock_estate_data(center_lat=37.4842, center_lon=126.9297, n_samples=100):
    """
    Generates mock real estate transaction data (Rent/Jeonse).
    """
    print("Generating mock real estate data...")
    
    lat_offsets = np.random.normal(0, 0.005, n_samples)
    lon_offsets = np.random.normal(0, 0.005, n_samples)
    
    lats = center_lat + lat_offsets
    lons = center_lon + lon_offsets
    
    # Mock Rent per Area (10,000 KRW / 3.3m2 approx or just arbitrary unit)
    # Let's say monthly rent per 10m2 ranging from 5 to 20 (unit: 10k KRW)
    rent_per_area = np.random.uniform(5, 20, n_samples)
    
    # Deposit (random)
    deposit = np.random.uniform(1000, 10000, n_samples)
    
    df = pd.DataFrame({
        'lat': lats,
        'lon': lons,
        'rent_per_area': rent_per_area, # 월세/면적
        'deposit': deposit,
        'name': [f"Estate_{i}" for i in range(n_samples)]
    })
    
    return df
