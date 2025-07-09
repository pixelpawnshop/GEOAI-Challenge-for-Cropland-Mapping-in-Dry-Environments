import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def train_model(train_features_path):
    train_df = pd.read_csv(train_features_path)
    X = train_df.drop(columns=['ID', 'label'])
    y = train_df['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, tree_method='hist', random_state=42)
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    print(f"✅ Validation Accuracy: {acc:.4f}")
    return model, X_val, y_val

def main():
    data_dir = 'data'
    train_features_path = os.path.join(data_dir, 'train_features.csv')
    model, X_val, y_val = train_model(train_features_path)
    # Optionally save model here

if __name__ == "__main__":
    main()


