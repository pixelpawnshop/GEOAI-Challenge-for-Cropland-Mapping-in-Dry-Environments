import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from sklearn.neighbors import KDTree
from scipy.stats import linregress
import os

def aggregate_features(df, id_col='ID'):
    return df.drop(columns=['translated_lat', 'translated_lon'], errors='ignore') \
             .groupby(id_col).agg(['mean', 'std', 'min', 'max']).reset_index()

def add_indices_and_ratios(s1, s2):
    s2 = s2.copy()
    s2['NDVI'] = (s2['B8'] - s2['B4']) / (s2['B8'] + s2['B4'] + 1e-6)
    s2['NDWI'] = (s2['B8'] - s2['B11']) / (s2['B8'] + s2['B11'] + 1e-6)
    s2['SAVI'] = 1.5 * (s2['B8'] - s2['B4']) / (s2['B8'] + s2['B4'] + 0.5 + 1e-6)
    s2['EVI'] = 2.5 * (s2['B8'] - s2['B4']) / (s2['B8'] + 6*s2['B4'] - 7.5*s2['B2'] + 1 + 1e-6)
    s2['NDSI'] = (s2['B3'] - s2['B11']) / (s2['B3'] + s2['B11'] + 1e-6)
    s2['B4_B3'] = s2['B4'] / (s2['B3'] + 1e-6)
    s2['B3_B2'] = s2['B3'] / (s2['B2'] + 1e-6)
    s2['B11_B12'] = s2['B11'] / (s2['B12'] + 1e-6)
    s1 = s1.copy()
    s1['VH_VV_ratio'] = s1['VH'] / (s1['VV'] + 1e-6)
    return s1, s2

def aggregate_features_with_indices(s1, s2):
    # Aggregate all band stats and new indices/ratios
    s2_cols = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'NDVI', 'NDWI', 'SAVI', 'EVI', 'NDSI', 'B4_B3', 'B3_B2', 'B11_B12']
    s1_feats = s1[['ID', 'VH', 'VV', 'VH_VV_ratio']].groupby('ID').agg(['mean', 'std', 'min', 'max'])
    s2_feats = s2[['ID'] + s2_cols].groupby('ID').agg(['mean', 'std', 'min', 'max'])
    # Percentiles for key bands/indices
    percentiles = [10, 25, 75, 90]
    for col in ['NDVI', 'B12', 'B11', 'B3']:
        for p in percentiles:
            s2_feats[(col, f'p{p}')] = s2.groupby('ID')[col].quantile(p/100).values
    # NDVI amplitude
    ndvi_amplitude = s2.groupby('ID')['NDVI'].agg(lambda x: x.max() - x.min()).rename('NDVI_amplitude')
    # Timing of max NDVI (day of year)
    s2['date'] = pd.to_datetime(s2['date'])
    ndvi_max_time = s2.loc[s2.groupby('ID')['NDVI'].idxmax()][['ID', 'date']]
    ndvi_max_time['NDVI_max_doy'] = ndvi_max_time['date'].dt.dayofyear
    ndvi_max_time = ndvi_max_time[['ID', 'NDVI_max_doy']]
    # Slope of NDVI over time
    ndvi_slope = s2.groupby('ID').apply(lambda group: linregress(group['date'].map(pd.Timestamp.toordinal), group['NDVI'])[0] if len(group) > 1 else 0).rename('NDVI_slope')
    # Number of green dates (NDVI > 0.3)
    ndvi_green_count = s2.groupby('ID')['NDVI'].apply(lambda x: (x > 0.3).sum()).rename('NDVI_green_count')
    # Merge all features
    s1_feats.columns = ['_'.join(col).strip('_') for col in s1_feats.columns.values]
    s2_feats.columns = ['_'.join(col).strip('_') for col in s2_feats.columns.values]
    feats = s2_feats.merge(s1_feats, on='ID', how='outer')
    feats = feats.merge(ndvi_amplitude, on='ID', how='left')
    feats = feats.merge(ndvi_max_time, on='ID', how='left')
    feats = feats.merge(ndvi_slope, on='ID', how='left')
    feats = feats.merge(ndvi_green_count, on='ID', how='left')
    feats = feats.reset_index(drop=True)

    # Feature selection: keep only top features based on SHAP and domain knowledge
    top_features = [
        'ID',
        'NDVI_max', 'NDVI_slope', 'NDVI_amplitude',
        'NDVI_p90', 'NDVI_p75', 'NDVI_p25', 'NDVI_p10',
        'NDVI_green_count', 'NDVI_mean', 'NDVI_std', 'NDVI_min',
        'NDVI_max_doy',
        'VH_VV_ratio_max', 'VH_VV_ratio_mean', 'VH_VV_ratio_std',
        'VH_mean', 'VH_std', 'VV_mean', 'VV_std',
        'B12_max', 'B12_p90', 'B12_p75', 'B12_p25', 'B12_p10',
        'B11_max', 'B11_p90', 'B11_p75', 'B11_p25', 'B11_p10',
        'B3_max', 'B3_p90', 'B3_p75', 'B3_p25', 'B3_p10',
        'B4_max', 'B4_mean', 'B4_std',
        'EVI_max', 'EVI_mean',
        'NDSI_max', 'NDSI_mean'
    ]
    # Only keep columns that exist in feats
    keep_cols = [col for col in top_features if col in feats.columns]
    feats = feats[keep_cols]
    return feats.reset_index(drop=True)

def preprocess_train(fergana_path, orenburg_path, sentinel1_path, sentinel2_path):
    s1 = pd.read_csv(sentinel1_path)
    s2 = pd.read_csv(sentinel2_path)
    s1, s2 = add_indices_and_ratios(s1, s2)

    # Load both shapefiles and extract labeled geodata
    fergana_gdf = gpd.read_file(fergana_path)
    orenburg_gdf = gpd.read_file(orenburg_path)
    gdf = pd.concat([fergana_gdf, orenburg_gdf])
    gdf = gdf[['Cropland', 'geometry']]
    gdf['lon'] = gdf.geometry.x
    gdf['lat'] = gdf.geometry.y

    # KDTree: Match Sentinel2 ID points to nearest labeled point
    tree = KDTree(gdf[['lat', 'lon']].values)
    s2_points = s2.groupby('ID')[['translated_lat', 'translated_lon']].mean().reset_index()
    dist, idx = tree.query(s2_points[['translated_lat', 'translated_lon']].values, k=1)
    s2_points['label'] = gdf.iloc[idx.flatten()]['Cropland'].values
    s2_labels = s2_points[['ID', 'label']]
    feats = aggregate_features_with_indices(s1, s2)
    train_df = feats.merge(s2_labels, on='ID', how='inner')
    train_df = train_df.dropna()
    return train_df

def preprocess_test(s1, s2, test_path):
    s1, s2 = add_indices_and_ratios(s1, s2)
    test_meta = pd.read_csv(test_path)
    test_ids = test_meta['ID'].unique()
    s1_test = s1[s1['ID'].isin(test_ids)]
    s2_test = s2[s2['ID'].isin(test_ids)]
    feats = aggregate_features_with_indices(s1_test, s2_test)
    return feats

def main():
    data_dir = 'data'
    fergana_path = os.path.join(data_dir, 'Fergana_training_samples.shp')
    orenburg_path = os.path.join(data_dir, 'Orenburg_training_samples.shp')
    sentinel1_path = os.path.join(data_dir, 'Sentinel1.csv')
    sentinel2_path = os.path.join(data_dir, 'Sentinel2.csv')
    test_path = os.path.join(data_dir, 'Test.csv')

    train_df = preprocess_train(fergana_path, orenburg_path, sentinel1_path, sentinel2_path)
    train_df.to_csv(os.path.join(data_dir, 'train_features.csv'), index=False)
    print('✅ Saved train_features.csv')

    # Save test features for pipeline (do NOT drop date before feature engineering)
    s1 = pd.read_csv(sentinel1_path)
    s2 = pd.read_csv(sentinel2_path)
    test_df = preprocess_test(s1, s2, test_path)
    # Now drop date if present
    if 'date' in test_df.columns:
        test_df = test_df.drop(columns=['date'])
    test_df.to_csv(os.path.join(data_dir, 'test_features.csv'), index=False)
    print('✅ Saved test_features.csv')

if __name__ == "__main__":
    main()