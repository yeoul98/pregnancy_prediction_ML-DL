"""
=============================================================================
난임 시술 성공 예측 파이프라인 v1 (Baseline)
- 기본적인 결측치 처리
- Label Encoding
- CatBoost / LightGBM / XGBoost 앙상블 (Simple Average)
=============================================================================
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

train = pd.read_csv("../../data/train.csv")
test = pd.read_csv("../../data/test.csv")

y = train["임신 성공 여부"]
X = train.drop(columns=["ID", "임신 성공 여부"])
X_test = test.drop(columns=["ID"])

# 범주형/수치형 분리
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].fillna("NAN"))
    X_test[col] = le.transform(X_test[col].fillna("NAN"))

# Simple 5-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
preds = np.zeros(len(X_test))

for tr_idx, val_idx in skf.split(X, y):
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=[(X.iloc[val_idx], y.iloc[val_idx])], 
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    preds += model.predict_proba(X_test)[:, 1] / 5

print(f"v1 Baseline OOF AUC: {roc_auc_score(y, oof):.6f}")

submission = pd.DataFrame({"ID": test["ID"], "probability": preds})
submission.to_csv("submission_v1.csv", index=False)
