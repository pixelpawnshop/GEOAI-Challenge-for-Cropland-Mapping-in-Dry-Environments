import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import numpy as np
import os
import shap
import matplotlib.pyplot as plt

def train_model(train_features_path, n_splits=5, show_shap=True):
    train_df = pd.read_csv(train_features_path)
    X = train_df.drop(columns=['ID', 'label'])
    y = train_df['label']
    # Compute scale_pos_weight for XGBoost
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"Using scale_pos_weight={scale_pos_weight:.2f}")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s, recalls, precisions, aucs = [], [], [], [], []
    models = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, tree_method='hist', random_state=42, scale_pos_weight=scale_pos_weight)
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        val_probs = model.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val, val_preds)
        f1 = f1_score(y_val, val_preds)
        recall = recall_score(y_val, val_preds)
        precision = precision_score(y_val, val_preds)
        try:
            auc = roc_auc_score(y_val, val_probs)
        except ValueError:
            auc = np.nan
        accs.append(acc)
        f1s.append(f1)
        recalls.append(recall)
        precisions.append(precision)
        aucs.append(auc)
        models.append(model)
        print(f"Fold {fold+1} - Acc: {acc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, AUC: {auc:.4f}")
    print(f"\n✅ 5-Fold CV Results:")
    print(f"  Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  F1-score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Recall:   {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"  Precision:{np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"  ROC AUC:  {np.nanmean(aucs):.4f} ± {np.nanstd(aucs):.4f}")
    # Optionally, retrain on all data for final model
    final_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, tree_method='hist', random_state=42, scale_pos_weight=scale_pos_weight)
    final_model.fit(X, y)
    if show_shap:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X)
        print('All features by mean(|SHAP|):')
        shap.summary_plot(shap_values, X, plot_type='bar', show=False, max_display=len(X.columns))
        plt.tight_layout()
        plt.show()
    return final_model, None, None

def main():
    data_dir = 'data'
    train_features_path = os.path.join(data_dir, 'train_features.csv')
    model, _, _ = train_model(train_features_path)
    # Optionally save model here

if __name__ == "__main__":
    main()