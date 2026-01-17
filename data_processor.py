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

def generate_extended_mock_data(center_lat=37.4842, center_lon=126.9297, n_samples=600):
    """
    Generates mock data including Safety, Medical, and Life Support.
    """
    print("Generating EXTENDED mock data...")
    
    # Generate random offsets
    lat_offsets = np.random.normal(0, 0.006, n_samples)
    lon_offsets = np.random.normal(0, 0.006, n_samples)
    lats = center_lat + lat_offsets
    lons = center_lon + lon_offsets
    
    categories = []
    names = []
    
    # Probabilities for categories
    # Cafe, Gym, Conv, Safety, Medical, Life
    types = ['Cafe', 'Gym', 'Convenience', 'Safety', 'Medical', 'Life']
    p_dist = [0.25, 0.15, 0.20, 0.10, 0.15, 0.15]
    
    for _ in range(n_samples):
        choice = np.random.choice(types, p=p_dist)
        cat_name = ''
        store_name = ''
        
        if choice == 'Cafe':
            cat_name = '카페'
            store_name = 'Mock Cafe'
        elif choice == 'Gym':
            cat_name = '헬스'
            store_name = 'Mock Gym'
        elif choice == 'Convenience':
            cat_name = '편의점'
            store_name = 'Mock GS25'
        elif choice == 'Safety':
            # Sub-types: CCTV vs Police
            sub = np.random.choice(['CCTV', 'Police'], p=[0.8, 0.2])
            cat_name = '안전' # Use generic or specific code
            store_name = f'Public {sub}'
        elif choice == 'Medical':
             # Sub-types: Pharmacy vs Clinic
            sub = np.random.choice(['Pharmacy', 'Clinic'], p=[0.5, 0.5])
            cat_name = '의료'
            store_name = f'Mock {sub}'
        elif choice == 'Life':
            # Laundry vs Parcel
            sub = np.random.choice(['Laundry', 'Parcel'], p=[0.7, 0.3])
            cat_name = '생활'
            store_name = f'Mock {sub}'
        
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
        # Use extended mock data for the requested upgrade
        df = generate_extended_mock_data()
    else:
        try:
            # Assuming CSV format common in public data
            df = pd.read_csv(file_path, encoding='cp949') 
        except:
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except FileNotFoundError:
                print(f"File not found: {file_path}. Switching to mock data.")
                df = generate_extended_mock_data()

    # Filter targets
    # Cafe: '카페', '커피'
    cond_cafe = df['상권업종소분류명'].astype(str).str.contains('카페|커피', na=False)
    
    # Gym: '헬스', '필라테스', '요가'
    cond_gym = df['상권업종소분류명'].astype(str).str.contains('헬스|필라테스|요가', na=False)
    
    # Convenience
    cond_conv_cat = df['상권업종소분류명'].astype(str).str.contains('편의점|약국', na=False) # '약국' moved to Medical? Let's check logic.
    # Instruction says Medical includes Pharmacy.
    # So we should be careful not to double count or prioritize.
    # Let's refine: Conv = Convenience Store only. Medical = Pharmacy + Clinic
    
    cond_conv = df['상권업종소분류명'].astype(str).str.contains('편의점', na=False) 
    # Or names '다이소', '올리브영' (Life Support or Convenience?)
    # Original logic had them in Conv. New logic separates Life Support.
    # New Prompt: "Life Support: Laundry, Parcel".
    # Convenience: "Convenience Store, Pharmacy(removed), Daiso(keep?), OliveYoung(keep?)"
    # Wait, prompt B says: Medical -> Pharmacy.
    # Prompt C says: Life -> Laundry, Parcel.
    # Prompt A: Conv -> "Convenience, Pharmacy(moved), Daiso, OliveYoung".
    # Let's keep Daiso/OliveYoung in Convenience for now unless specified otherwise.
    # Actually, let's strictly follow:
    # Medical: Pharmacy, Clinic.
    # Safety: CCTV, Police.
    # Life: Laundry, Parcel.
    # Conv: Convenience Store (gs25 etc).
    
    # Revised Logic:
    
    # 1. Cafe
    cond_cafe = df['상권업종소분류명'].astype(str).str.contains('카페|커피', na=False)
    
    # 2. Gym
    cond_gym = df['상권업종소분류명'].astype(str).str.contains('헬스|필라테스|요가|운동', na=False)
    
    # 3. Convenience (Convenience Store only or +Daiso/OliveYoung)
    # Let's include Daiso/OliveYoung here as typical "Conv"
    cond_conv_bg = df['상권업종소분류명'].astype(str).str.contains('편의점', na=False)
    cond_conv_name = df['상호명'].astype(str).str.contains('다이소|올리브영', na=False)
    cond_conv = cond_conv_bg | cond_conv_name
    
    # 4. Safety
    # In mock data, category is '안전', name contains 'Police'/'CCTV'
    # In real data, we would filter by keyword.
    # Here we rely on Mock '안전' or keywords.
    cond_safety = df['상호명'].astype(str).str.contains('Police|CCTV|지구대|파출소|치안', na=False) | \
                  df['상권업종소분류명'].astype(str).str.contains('안전', na=False)
    
    # 5. Medical
    cond_medical = df['상호명'].astype(str).str.contains('Pharmacy|Clinic|약국|내과|이비인후과|치과|병원', na=False) | \
                   df['상권업종소분류명'].astype(str).str.contains('의료|병원|약국', na=False)

    # 6. Life
    cond_life = df['상호명'].astype(str).str.contains('Laundry|Parcel|세탁|빨래|택배', na=False) | \
                df['상권업종소분류명'].astype(str).str.contains('생활', na=False)
    
    # Labeling
    df.loc[cond_cafe, 'type'] = 'cafe'
    df.loc[cond_gym, 'type'] = 'gym'
    df.loc[cond_conv, 'type'] = 'convenience'
    df.loc[cond_safety, 'type'] = 'safety'
    df.loc[cond_medical, 'type'] = 'medical'
    df.loc[cond_life, 'type'] = 'life'
    
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
