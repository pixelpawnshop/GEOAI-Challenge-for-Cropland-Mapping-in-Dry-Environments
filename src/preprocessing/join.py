import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from sklearn.neighbors import KDTree
import os

def aggregate_features(df, id_col='ID'):
    return df.drop(columns=['translated_lat', 'translated_lon'], errors='ignore') \
             .groupby(id_col).agg(['mean', 'std', 'min', 'max']).reset_index()

def preprocess_train(fergana_path, orenburg_path, sentinel1_path, sentinel2_path):
    # Load Sentinel-1 and 2 and drop date if present
    s1 = pd.read_csv(sentinel1_path)
    if 'date' in s1.columns:
        s1 = s1.drop(columns=['date'])
    s2 = pd.read_csv(sentinel2_path)
    if 'date' in s2.columns:
        s2 = s2.drop(columns=['date'])

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

    # Aggregate train features
    s1_feats = aggregate_features(s1[['ID', 'VH', 'VV']])
    s2_feats = aggregate_features(s2[['ID', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']])

    # Flatten multi-index columns
    s1_feats.columns = ['_'.join(col).strip('_') for col in s1_feats.columns.values]
    s2_feats.columns = ['_'.join(col).strip('_') for col in s2_feats.columns.values]

    # Merge training features and labels
    train_df = s2_feats.merge(s1_feats, on='ID', how='outer').merge(s2_labels, on='ID', how='inner')
    train_df = train_df.dropna()
    return train_df

def preprocess_test(s1, s2, test_path):
    test_meta = pd.read_csv(test_path)
    test_ids = test_meta['ID'].unique()
    s1_test = s1[s1['ID'].isin(test_ids)]
    s2_test = s2[s2['ID'].isin(test_ids)]
    s1_test_feats = aggregate_features(s1_test[['ID', 'VH', 'VV']])
    s2_test_feats = aggregate_features(s2_test[['ID', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']])
    s1_test_feats.columns = ['_'.join(col).strip('_') for col in s1_test_feats.columns.values]
    s2_test_feats.columns = ['_'.join(col).strip('_') for col in s2_test_feats.columns.values]
    test_df = s2_test_feats.merge(s1_test_feats, on='ID', how='outer').fillna(0)
    return test_df

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

    # Save test features for pipeline
    s1 = pd.read_csv(sentinel1_path)
    if 'date' in s1.columns:
        s1 = s1.drop(columns=['date'])
    s2 = pd.read_csv(sentinel2_path)
    if 'date' in s2.columns:
        s2 = s2.drop(columns=['date'])
    test_df = preprocess_test(s1, s2, test_path)
    test_df.to_csv(os.path.join(data_dir, 'test_features.csv'), index=False)
    print('✅ Saved test_features.csv')

if __name__ == "__main__":
    main()