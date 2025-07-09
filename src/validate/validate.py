import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import os
import sys

def aggregate_features(df, id_col='ID'):
    return df.drop(columns=['translated_lat', 'translated_lon'], errors='ignore') \
             .groupby(id_col).agg(['mean', 'std', 'min', 'max']).reset_index()

def validate_and_submit(model, test_features_path, sample_submission_path):
    test_df = pd.read_csv(test_features_path)
    X_test = test_df.drop(columns=['ID'])
    test_preds = model.predict(X_test)
    submission = pd.read_csv(sample_submission_path)
    if 'Cropland' in submission.columns:
        submission = submission.drop(columns=['Cropland'])
    submission = submission.merge(test_df[['ID']], on='ID', how='right')
    submission['Cropland'] = test_preds.astype(int)
    submission.to_csv('submission.csv', index=False)
    print('📤 Saved submission.csv')

def main():
    data_dir = 'data'
    test_features_path = os.path.join(data_dir, 'test_features.csv')
    sample_submission_path = os.path.join(data_dir, 'SampleSubmission.csv')
    # Add the parent directory to sys.path to allow import
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modelling')))
    from modelling import train_model
    train_features_path = os.path.join(data_dir, 'train_features.csv')
    model, X_val, y_val = train_model(train_features_path)
    validate_and_submit(model, test_features_path, sample_submission_path)

if __name__ == "__main__":
    main()