"""
=============================================================================
난임 시술 성공 예측 파이프라인 v3
- K-Fold Target Encoding (Smoothing, Leakage-Free)
- 다양성 Level-1 모델 (CatBoost×3, LightGBM×3, XGBoost×3 = 9모델)
- RidgeClassifier Stacking Meta-Learner
- v2 대비 t-test 검증
=============================================================================
"""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import xgboost as xgb
from tqdm.auto import tqdm

# =============================================================================
# 0. 설정
# =============================================================================
SEED = 42
N_FOLDS = 5
DATA_DIR = "."

np.random.seed(SEED)

# =============================================================================
# 1. 데이터 로딩
# =============================================================================
print("=" * 70)
print("[1/7] 데이터 로딩")
print("=" * 70)

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")

TARGET = "임신 성공 여부"
ID_COL = "ID"

y = train[TARGET].values
train_ids = train[ID_COL].values
test_ids = test[ID_COL].values

print(f"  Train: {train.shape}, Test: {test.shape}")
print(f"  Target 분포: 0={sum(y==0):,} / 1={sum(y==1):,} ({sum(y==1)/len(y)*100:.1f}%)")

# =============================================================================
# 2. 기본 전처리 (v2와 동일)
# =============================================================================
print("\n" + "=" * 70)
print("[2/7] 데이터 전처리 & 피처 엔지니어링")
print("=" * 70)

drop_cols_basic = [ID_COL]
if TARGET in train.columns:
    drop_cols_basic.append(TARGET)

train_feat = train.drop(columns=drop_cols_basic)
test_feat = test.drop(columns=[ID_COL])

combined = pd.concat([train_feat, test_feat], axis=0, ignore_index=True)
n_train = len(train_feat)
n_test = len(test_feat)

# 2-1. 결측률 80% 이상 드롭
high_missing_cols = [c for c in combined.columns
                     if combined[c].isnull().sum() / len(combined) >= 0.80]
print(f"  결측률 ≥80% 드롭 ({len(high_missing_cols)}개): {high_missing_cols}")
combined.drop(columns=high_missing_cols, inplace=True)

# 2-2. 상수 컬럼 드롭
const_cols = [c for c in combined.columns if combined[c].nunique() <= 1]
print(f"  상수 컬럼 드롭 ({len(const_cols)}개): {const_cols}")
combined.drop(columns=const_cols, inplace=True)

# 2-3. 결측치 통합
unknown_values = ["알 수 없음", "기록되지 않은 시행"]
for col in combined.select_dtypes(include=["object"]).columns:
    combined[col] = combined[col].replace(unknown_values, np.nan)

# 2-4. 결측 이진 피처
missing_feat_cols = []
for col in combined.columns:
    miss_rate = combined[col].isnull().sum() / len(combined)
    if 0.05 <= miss_rate < 0.80:
        feat_name = f"{col}_missing"
        combined[feat_name] = combined[col].isnull().astype(int)
        missing_feat_cols.append(feat_name)
print(f"  결측 이진 피처 생성 ({len(missing_feat_cols)}개)")

# 2-5. 나이 수치화 + 고령 여부
age_map = {
    "만18-34세": 26, "만35-37세": 36, "만38-39세": 38.5,
    "만40-42세": 41, "만43-44세": 43.5, "만45-50세": 47.5,
}
age_col = "시술 당시 나이"
if age_col in combined.columns:
    combined["나이_수치"] = combined[age_col].map(age_map)
    combined["나이_수치"].fillna(combined["나이_수치"].median(), inplace=True)
    combined["고령_40이상"] = (combined["나이_수치"] >= 40).astype(int)
    combined.drop(columns=[age_col], inplace=True)
    print("  나이 수치화 + 고령 여부 피처 생성 완료")

# 2-6. 시술 횟수 비닝
count_cols = [
    "총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
    "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수"
]
count_value_map = {"0회": 0, "1회": 1, "2회": 2, "3회": 3, "4회": 4, "5회": 5, "6회 이상": 6}

def bin_count(val):
    if pd.isna(val): return np.nan
    num = count_value_map.get(val, np.nan)
    if pd.isna(num): return np.nan
    if num == 0: return "bin_0"
    elif num <= 2: return "bin_1_2"
    else: return "bin_3plus"

for col in count_cols:
    if col in combined.columns:
        combined[f"{col}_num"] = combined[col].map(count_value_map)
        combined[f"{col}_bin"] = combined[col].apply(bin_count)
        combined.drop(columns=[col], inplace=True)
print(f"  시술 횟수 비닝 완료 ({len(count_cols)}개 컬럼)")

# 2-7. 특정 시술 유형
specific_type_col = "특정 시술 유형"
if specific_type_col in combined.columns:
    top_types = ["ICSI", "IVF", "Unknown"]
    combined[specific_type_col] = combined[specific_type_col].apply(
        lambda x: x if x in top_types else ("기타" if pd.notna(x) else np.nan)
    )

# 2-8. 수치형 파생 변수
if "총 생성 배아 수" in combined.columns and "혼합된 난자 수" in combined.columns:
    combined["수정률"] = (combined["총 생성 배아 수"] / (combined["혼합된 난자 수"] + 1e-8)).clip(0, 10)
if "이식된 배아 수" in combined.columns and "총 생성 배아 수" in combined.columns:
    combined["이식률"] = (combined["이식된 배아 수"] / (combined["총 생성 배아 수"] + 1e-8)).clip(0, 10)
if "미세주입에서 생성된 배아 수" in combined.columns and "총 생성 배아 수" in combined.columns:
    combined["ICSI배아비율"] = (combined["미세주입에서 생성된 배아 수"] / (combined["총 생성 배아 수"] + 1e-8)).clip(0, 10)
if "저장된 배아 수" in combined.columns and "총 생성 배아 수" in combined.columns:
    combined["저장률"] = (combined["저장된 배아 수"] / (combined["총 생성 배아 수"] + 1e-8)).clip(0, 10)
if "총 생성 배아 수" in combined.columns and "수집된 신선 난자 수" in combined.columns:
    combined["난자대비배아효율"] = (combined["총 생성 배아 수"] / (combined["수집된 신선 난자 수"] + 1e-8)).clip(0, 10)
print("  수치형 파생 변수 생성 완료")

# 2-9. 불임 원인 합산
infertility_cols = [c for c in combined.columns if c.startswith("불임 원인 -")]
if infertility_cols:
    combined["불임원인_총개수"] = combined[infertility_cols].sum(axis=1)
cause_pairs = [("남성 주 불임 원인", "남성 부 불임 원인"),
               ("여성 주 불임 원인", "여성 부 불임 원인"),
               ("부부 주 불임 원인", "부부 부 불임 원인")]
for main_c, sub_c in cause_pairs:
    if main_c in combined.columns and sub_c in combined.columns:
        combined[f"{main_c[:2]}_불임합산"] = combined[main_c] + combined[sub_c]

# 2-10. 기증자 나이 수치화
donor_age_map = {"만20세 이하": 18, "만21-25세": 23, "만26-30세": 28,
                 "만31-35세": 33, "만36-40세": 38, "만41-45세": 43}
for col in ["난자 기증자 나이", "정자 기증자 나이"]:
    if col in combined.columns:
        combined[f"{col}_num"] = combined[col].map(donor_age_map)

# 2-11. 경과일 파생
if "난자 혼합 경과일" in combined.columns and "배아 이식 경과일" in combined.columns:
    combined["혼합_이식_간격"] = combined["배아 이식 경과일"] - combined["난자 혼합 경과일"]

print("  기타 파생 변수 생성 완료")

# =============================================================================
# 3. K-Fold Target Encoding (Leakage-Free)
# =============================================================================
print("\n" + "=" * 70)
print("[3/7] K-Fold Target Encoding (Smoothing, Leakage-Free)")
print("=" * 70)

cat_columns_raw = combined.select_dtypes(include=["object"]).columns.tolist()
num_columns = combined.select_dtypes(include=["number"]).columns.tolist()

print(f"  Target Encoding 대상 범주형 컬럼: {len(cat_columns_raw)}개")

# 범주형 NaN → "MISSING"
for col in cat_columns_raw:
    combined[col] = combined[col].fillna("MISSING").astype(str)

# Train/Test 분리 (Target Encoding 전)
X_train_raw = combined.iloc[:n_train].reset_index(drop=True)
X_test_raw = combined.iloc[n_train:].reset_index(drop=True)

# Smoothing Target Encoding 함수
SMOOTHING = 10  # smoothing factor
GLOBAL_MEAN = y.mean()

def target_encode_column(train_col, test_col, y_col, fold_indices, smoothing=SMOOTHING):
    """
    K-Fold 내부에서만 Target Encoding을 수행하여 Leakage를 방지.
    OOF 방식: 각 fold의 validation에 대해 training 부분에서만 통계 계산.
    Test에 대해서는 전체 train 데이터로 통계 계산.
    """
    train_encoded = np.full(len(train_col), GLOBAL_MEAN)
    
    for tr_idx, val_idx in fold_indices:
        # Training fold에서만 통계 계산
        tr_data = pd.DataFrame({"cat": train_col.iloc[tr_idx], "target": y_col[tr_idx]})
        stats_df = tr_data.groupby("cat")["target"].agg(["mean", "count"])
        
        # Smoothing: (count * mean + smoothing * global_mean) / (count + smoothing)
        smoothed_mean = (stats_df["count"] * stats_df["mean"] + smoothing * GLOBAL_MEAN) / \
                        (stats_df["count"] + smoothing)
        
        # Validation fold 인코딩
        train_encoded[val_idx] = train_col.iloc[val_idx].map(smoothed_mean).fillna(GLOBAL_MEAN).values
    
    # Test 인코딩: 전체 Train 데이터로 통계 계산
    full_data = pd.DataFrame({"cat": train_col, "target": y_col})
    full_stats = full_data.groupby("cat")["target"].agg(["mean", "count"])
    full_smoothed = (full_stats["count"] * full_stats["mean"] + smoothing * GLOBAL_MEAN) / \
                    (full_stats["count"] + smoothing)
    test_encoded = test_col.map(full_smoothed).fillna(GLOBAL_MEAN).values
    
    return train_encoded, test_encoded

# K-Fold indices 미리 생성 (Target Encoding + 모델 학습에 동일 사용)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_indices = list(skf.split(X_train_raw, y))

# Target Encoding 적용
te_train_features = {}
te_test_features = {}

for col in tqdm(cat_columns_raw, desc="Target Encoding"):
    tr_enc, te_enc = target_encode_column(
        X_train_raw[col], X_test_raw[col], y, fold_indices
    )
    te_col_name = f"{col}_te"
    te_train_features[te_col_name] = tr_enc
    te_test_features[te_col_name] = te_enc

te_train_df = pd.DataFrame(te_train_features)
te_test_df = pd.DataFrame(te_test_features)

print(f"  Target Encoding 피처 생성 완료: {te_train_df.shape[1]}개")

# =============================================================================
# 3-1. 최종 피처 세트 구성
# =============================================================================
# 수치형 NaN → -999
for col in num_columns:
    combined[col] = combined[col].fillna(-999)

X_train_num = combined.iloc[:n_train][num_columns].reset_index(drop=True)
X_test_num = combined.iloc[n_train:][num_columns].reset_index(drop=True)

# CatBoost용: 원본 범주형 유지 + 수치형 + TE
X_train_cat = pd.concat([X_train_raw[cat_columns_raw], X_train_num, te_train_df], axis=1)
X_test_cat = pd.concat([X_test_raw[cat_columns_raw], X_test_num, te_test_df], axis=1)
cat_indices_cb = list(range(len(cat_columns_raw)))

# LGB/XGB용: Label Encoding + 수치형 + TE
le_dict = {}
X_train_le = X_train_raw[cat_columns_raw].copy()
X_test_le = X_test_raw[cat_columns_raw].copy()
for col in cat_columns_raw:
    le = LabelEncoder()
    le.fit(pd.concat([X_train_le[col], X_test_le[col]]))
    X_train_le[col] = le.transform(X_train_le[col])
    X_test_le[col] = le.transform(X_test_le[col])
    le_dict[col] = le

X_train_enc = pd.concat([X_train_le, X_train_num, te_train_df], axis=1)
X_test_enc = pd.concat([X_test_le, X_test_num, te_test_df], axis=1)

feature_names = list(X_train_enc.columns)
print(f"\n  최종 피처 수: {len(feature_names)}개 (범주형 {len(cat_columns_raw)} + 수치형 {len(num_columns)} + TE {te_train_df.shape[1]})")

# =============================================================================
# 4. Level-1 모델 정의 (다양성 확보)
# =============================================================================
print("\n" + "=" * 70)
print("[4/7] Level-1 모델 학습 (9개 다양성 모델, Stratified 5-Fold)")
print("=" * 70)

# 모델 설정: (이름, 모델 타입, 하이퍼파라미터, 데이터 타입)
MODEL_CONFIGS = [
    # --- CatBoost 변형 3종 ---
    ("CB_d6_s42", "catboost", dict(
        iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        eval_metric="AUC", random_seed=42, early_stopping_rounds=100,
        verbose=0, task_type="CPU",
    )),
    ("CB_d8_s7", "catboost", dict(
        iterations=2000, learning_rate=0.03, depth=8, l2_leaf_reg=5,
        eval_metric="AUC", random_seed=7, early_stopping_rounds=100,
        verbose=0, task_type="CPU",
    )),
    ("CB_d4_s123", "catboost", dict(
        iterations=3000, learning_rate=0.08, depth=4, l2_leaf_reg=1,
        eval_metric="AUC", random_seed=123, early_stopping_rounds=150,
        verbose=0, task_type="CPU",
    )),
    # --- LightGBM 변형 3종 ---
    ("LGB_nl63_s42", "lgbm", dict(
        n_estimators=2000, learning_rate=0.05, max_depth=-1, num_leaves=63,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, metric="auc",
        random_state=42, verbose=-1, n_jobs=-1,
    )),
    ("LGB_nl127_s7", "lgbm", dict(
        n_estimators=2000, learning_rate=0.03, max_depth=-1, num_leaves=127,
        min_child_samples=50, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=2.0, metric="auc",
        random_state=7, verbose=-1, n_jobs=-1,
    )),
    ("LGB_nl31_s123", "lgbm", dict(
        n_estimators=3000, learning_rate=0.08, max_depth=6, num_leaves=31,
        min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.01, reg_lambda=0.5, metric="auc",
        random_state=123, verbose=-1, n_jobs=-1,
    )),
    # --- XGBoost 변형 3종 ---
    ("XGB_d6_s42", "xgb", dict(
        n_estimators=2000, learning_rate=0.05, max_depth=6,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, eval_metric="auc",
        tree_method="hist", early_stopping_rounds=100,
        random_state=42, verbosity=0, n_jobs=-1,
    )),
    ("XGB_d8_s7", "xgb", dict(
        n_estimators=2000, learning_rate=0.03, max_depth=8,
        min_child_weight=10, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=2.0, eval_metric="auc",
        tree_method="hist", early_stopping_rounds=100,
        random_state=7, verbosity=0, n_jobs=-1,
    )),
    ("XGB_d4_s123", "xgb", dict(
        n_estimators=3000, learning_rate=0.08, max_depth=4,
        min_child_weight=3, subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.01, reg_lambda=0.5, eval_metric="auc",
        tree_method="hist", early_stopping_rounds=150,
        random_state=123, verbosity=0, n_jobs=-1,
    )),
]

N_MODELS = len(MODEL_CONFIGS)
print(f"  Level-1 모델 수: {N_MODELS}개")
for name, mtype, _ in MODEL_CONFIGS:
    print(f"    - {name} ({mtype})")

# =============================================================================
# 5. Level-1 학습 + OOF 메타 피처 생성
# =============================================================================
# OOF / Test 예측 저장
oof_preds = np.zeros((n_train, N_MODELS))  # (n_train, 9)
test_preds = np.zeros((n_test, N_MODELS))  # (n_test, 9)
fold_aucs = {name: [] for name, _, _ in MODEL_CONFIGS}

for model_idx, (model_name, model_type, params) in enumerate(MODEL_CONFIGS):
    print(f"\n{'─'*60}")
    print(f"  모델 [{model_idx+1}/{N_MODELS}]: {model_name}")
    print(f"{'─'*60}")
    
    for fold_idx, (tr_idx, val_idx) in enumerate(tqdm(fold_indices, desc=f"{model_name}", leave=False)):
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        if model_type == "catboost":
            X_tr = X_train_cat.iloc[tr_idx]
            X_val = X_train_cat.iloc[val_idx]
            
            model = CatBoostClassifier(**params)
            train_pool = Pool(X_tr, y_tr, cat_features=cat_indices_cb)
            val_pool = Pool(X_val, y_val, cat_features=cat_indices_cb)
            model.fit(train_pool, eval_set=val_pool)
            
            val_pred = model.predict_proba(X_val)[:, 1]
            test_pred = model.predict_proba(X_test_cat)[:, 1]
            
        elif model_type == "lgbm":
            X_tr = X_train_enc.iloc[tr_idx]
            X_val = X_train_enc.iloc[val_idx]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
            )
            
            val_pred = model.predict_proba(X_val)[:, 1]
            test_pred = model.predict_proba(X_test_enc)[:, 1]
            
        elif model_type == "xgb":
            X_tr = X_train_enc.iloc[tr_idx]
            X_val = X_train_enc.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
            
            val_pred = model.predict_proba(X_val)[:, 1]
            test_pred = model.predict_proba(X_test_enc)[:, 1]
        
        oof_preds[val_idx, model_idx] = val_pred
        test_preds[:, model_idx] += test_pred / N_FOLDS
        
        fold_auc = roc_auc_score(y_val, val_pred)
        fold_aucs[model_name].append(fold_auc)
    
    model_oof_auc = roc_auc_score(y, oof_preds[:, model_idx])
    print(f"  → {model_name} OOF AUC: {model_oof_auc:.6f}  "
          f"(folds: {[f'{a:.4f}' for a in fold_aucs[model_name]]})")

# OOF/Test 예측값 저장
np.save(f"{DATA_DIR}/oof_preds_v3.npy", oof_preds)
np.save(f"{DATA_DIR}/test_preds_v3.npy", test_preds)
np.save(f"{DATA_DIR}/y_train.npy", y)
print("\n  OOF/Test 예측값 .npy 저장 완료")

# =============================================================================
# 6. Level-1 성능 요약 + 단순 앙상블 기준선
# =============================================================================
print("\n" + "=" * 70)
print("[5/7] Level-1 성능 요약")
print("=" * 70)

model_names = [name for name, _, _ in MODEL_CONFIGS]
for i, name in enumerate(model_names):
    auc = roc_auc_score(y, oof_preds[:, i])
    print(f"  {name:20s} OOF AUC: {auc:.6f}")

# 단순 평균 앙상블
simple_avg = oof_preds.mean(axis=1)
simple_avg_auc = roc_auc_score(y, simple_avg)
print(f"\n  균등 평균 앙상블 OOF AUC: {simple_avg_auc:.6f}")

# =============================================================================
# 7. Stacking: RidgeClassifier 메타 러너
# =============================================================================
print("\n" + "=" * 70)
print("[6/7] Stacking: RidgeClassifier Meta-Learner")
print("=" * 70)

# RidgeClassifier는 predict_proba가 없으므로 CalibratedClassifierCV로 감싸서 사용
# 또는 LogisticRegression(Ridge penalty)를 사용
# → LogisticRegression(penalty='l2')가 사실상 Ridge Logistic Regression

# 방법 1: LogisticRegression (L2 = Ridge) - 직접 확률 출력 가능
# 방법 2: RidgeClassifier + CalibratedClassifierCV

# --- 방법 2: 정석 RidgeClassifier + Calibrated ---
print("\n  [Meta-Learner] RidgeClassifier + CalibratedClassifierCV")

oof_meta = np.zeros(n_train)
test_meta_preds = np.zeros(n_test)

meta_fold_aucs = []

for fold_idx, (tr_idx, val_idx) in enumerate(tqdm(fold_indices, desc="Stacking Meta", leave=True)):
    X_meta_tr = oof_preds[tr_idx]   # (len(tr_idx), 9)
    X_meta_val = oof_preds[val_idx]  # (len(val_idx), 9)
    y_meta_tr = y[tr_idx]
    y_meta_val = y[val_idx]
    
    # RidgeClassifier with calibration
    base_ridge = RidgeClassifier(alpha=1.0, class_weight="balanced")
    meta_model = CalibratedClassifierCV(base_ridge, cv=3, method="sigmoid")
    meta_model.fit(X_meta_tr, y_meta_tr)
    
    val_meta_pred = meta_model.predict_proba(X_meta_val)[:, 1]
    test_meta_pred = meta_model.predict_proba(test_preds)[:, 1]
    
    oof_meta[val_idx] = val_meta_pred
    test_meta_preds += test_meta_pred / N_FOLDS
    
    fold_auc = roc_auc_score(y_meta_val, val_meta_pred)
    meta_fold_aucs.append(fold_auc)
    print(f"    Fold {fold_idx+1} Meta AUC: {fold_auc:.6f}")

stacking_oof_auc = roc_auc_score(y, oof_meta)
print(f"\n  Stacking (RidgeClassifier) OOF AUC: {stacking_oof_auc:.6f}")

# --- 최종 RidgeClassifier 메타 모델(전체 train으로 학습)로 가중치 리포트 ---
final_ridge = RidgeClassifier(alpha=1.0, class_weight="balanced")
final_ridge.fit(oof_preds, y)
ridge_coefs = final_ridge.coef_.ravel()
ridge_intercept = final_ridge.intercept_.ravel()[0] if hasattr(final_ridge.intercept_, 'ravel') else final_ridge.intercept_

print(f"\n  ◆ RidgeClassifier 메타 모델 가중치:")
print(f"    {'모델':20s} | 가중치")
print(f"    {'─'*20}─┼─{'─'*12}")
for name, coef in zip(model_names, ridge_coefs):
    print(f"    {name:20s} | {coef:+.6f}")
print(f"    {'Intercept':20s} | {ridge_intercept:+.6f}")

# --- 보충: scipy.optimize 가중 앙상블도 비교 ---
print("\n  [비교] scipy.optimize 가중 앙상블")

def neg_auc_obj(w):
    w_norm = w / w.sum()
    blended = oof_preds @ w_norm
    return -roc_auc_score(y, blended)

best_result = None
best_neg_auc = 0

bounds = [(0.001, 0.999)] * N_MODELS
constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}

for _ in tqdm(range(20), desc="Weight Opt"):
    x0 = np.random.dirichlet(np.ones(N_MODELS))
    result = minimize(neg_auc_obj, x0=x0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-12})
    if best_result is None or result.fun < best_neg_auc:
        best_neg_auc = result.fun
        best_result = result

opt_w = best_result.x / best_result.x.sum()
opt_auc = -best_neg_auc

print(f"  최적 가중 앙상블 OOF AUC: {opt_auc:.6f}")
print(f"  최적 가중치:")
for name, w in zip(model_names, opt_w):
    print(f"    {name:20s}: {w:.4f}")

# =============================================================================
# 7-1. v2 대비 t-test 검증
# =============================================================================
print("\n" + "=" * 70)
print("[6-1/7] v2 대비 성능 비교 (t-test)")
print("=" * 70)

V2_OOF_AUC = 0.740111  # v2 가중 앙상블 OOF AUC

# v2 OOF 예측값 로딩 (존재하는 경우)
v2_oof_path = f"{DATA_DIR}/oof_catboost.npy"
if os.path.exists(v2_oof_path):
    oof_cb_v2 = np.load(f"{DATA_DIR}/oof_catboost.npy")
    oof_lgb_v2 = np.load(f"{DATA_DIR}/oof_lgbm.npy")
    oof_xgb_v2 = np.load(f"{DATA_DIR}/oof_xgb.npy")
    # v2 가중 앙상블 (CB=0.5, LGB=0.25, XGB=0.25)
    oof_v2 = 0.5 * oof_cb_v2 + 0.25 * oof_lgb_v2 + 0.25 * oof_xgb_v2
    
    # Fold별 AUC 비교
    v2_fold_aucs = []
    v3_stack_fold_aucs = []
    v3_weighted_fold_aucs = []
    
    for tr_idx, val_idx in fold_indices:
        v2_fold_aucs.append(roc_auc_score(y[val_idx], oof_v2[val_idx]))
        v3_stack_fold_aucs.append(roc_auc_score(y[val_idx], oof_meta[val_idx]))
        v3_weighted_fold_aucs.append(roc_auc_score(y[val_idx], (oof_preds @ opt_w)[val_idx]))
    
    print(f"\n  v2 가중 앙상블 Fold AUCs: {[f'{a:.4f}' for a in v2_fold_aucs]}")
    print(f"  v3 스태킹       Fold AUCs: {[f'{a:.4f}' for a in v3_stack_fold_aucs]}")
    print(f"  v3 가중 앙상블   Fold AUCs: {[f'{a:.4f}' for a in v3_weighted_fold_aucs]}")
    
    # t-test: v3 Stacking vs v2
    t_stat1, p_val1 = stats.ttest_rel(v3_stack_fold_aucs, v2_fold_aucs)
    sig1 = "유의함 ✓" if p_val1 < 0.01 else "유의하지 않음 ✗"
    print(f"\n  ◆ v3 스태킹 vs v2: t={t_stat1:.4f}, p={p_val1:.6f} → {sig1}")
    
    # t-test: v3 Weighted vs v2
    t_stat2, p_val2 = stats.ttest_rel(v3_weighted_fold_aucs, v2_fold_aucs)
    sig2 = "유의함 ✓" if p_val2 < 0.01 else "유의하지 않음 ✗"
    print(f"  ◆ v3 가중앙상블 vs v2: t={t_stat2:.4f}, p={p_val2:.6f} → {sig2}")
else:
    print("  v2 OOF 예측 파일 없음 — OOF AUC 수치 비교만 수행")
    print(f"  v2 가중 앙상블 OOF AUC: {V2_OOF_AUC:.6f}")

# =============================================================================
# 8. 최종 Submission 생성 (최고 성능 방식 선택)
# =============================================================================
print("\n" + "=" * 70)
print("[7/7] 최종 Submission 생성")
print("=" * 70)

# 성능 비교하여 최고 방식 선택
candidates = {
    "균등 평균": (simple_avg_auc, oof_preds.mean(axis=1), test_preds.mean(axis=1)),
    "가중 앙상블": (opt_auc, oof_preds @ opt_w, test_preds @ opt_w),
    "RidgeClassifier 스태킹": (stacking_oof_auc, oof_meta, test_meta_preds),
}

best_method = max(candidates.keys(), key=lambda k: candidates[k][0])
best_oof_auc, best_oof, best_test = candidates[best_method]

print(f"\n  성능 비교:")
for method, (auc, _, _) in candidates.items():
    marker = " ← 최고" if method == best_method else ""
    print(f"    {method:25s}: OOF AUC = {auc:.6f}{marker}")

# 확률 교정 (Platt Scaling)
print(f"\n  확률 교정 (Platt Scaling) 적용...")
target_rate = y.mean()

calibrator = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
calibrator.fit(best_oof.reshape(-1, 1), y)

oof_calibrated = calibrator.predict_proba(best_oof.reshape(-1, 1))[:, 1]
test_calibrated = calibrator.predict_proba(best_test.reshape(-1, 1))[:, 1]

calib_auc = roc_auc_score(y, oof_calibrated)
print(f"  교정 후 OOF AUC: {calib_auc:.6f}")
print(f"  교정 전 Test 평균: {best_test.mean():.6f}")
print(f"  교정 후 Test 평균: {test_calibrated.mean():.6f} (타겟 비율: {target_rate:.6f})")

# 미세 보정
mean_diff = abs(test_calibrated.mean() - target_rate)
if mean_diff > 0.005:
    scale_factor = target_rate / test_calibrated.mean()
    test_final = np.clip(test_calibrated * scale_factor, 0.0, 1.0)
    print(f"  추가 미세 보정 → Test 평균: {test_final.mean():.6f}")
else:
    test_final = test_calibrated

# submission.csv 생성
submission = pd.DataFrame({"ID": test_ids, "probability": test_final})
submission.to_csv(f"{DATA_DIR}/submission.csv", index=False)

print(f"\n  submission.csv 저장 완료!")
print(f"  행 수: {len(submission):,}")
print(f"  probability: mean={submission['probability'].mean():.6f}, "
      f"std={submission['probability'].std():.6f}")
print(f"  min={submission['probability'].min():.6f}, max={submission['probability'].max():.6f}")

# =============================================================================
# 최종 요약
# =============================================================================
print("\n" + "=" * 70)
print("최종 요약")
print("=" * 70)

print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │                   v3 파이프라인 성능 종합                        │
  ├──────────────────────────┬───────────────────────────────────────┤
  │ v2 가중 앙상블 (기준)    │ {V2_OOF_AUC:.6f}                           │""")

for name, (auc, _, _) in candidates.items():
    diff = auc - V2_OOF_AUC
    sign = "+" if diff > 0 else ""
    print(f"  │ v3 {name:20s} │ {auc:.6f} ({sign}{diff:.6f})               │")

print(f"""  ├──────────────────────────┼───────────────────────────────────────┤
  │ 채택 방식               │ {best_method:37s} │
  │ 교정 후 OOF AUC         │ {calib_auc:.6f}                           │
  └──────────────────────────┴───────────────────────────────────────┘

  ◆ RidgeClassifier 메타 모델 가중치:""")
for name, coef in zip(model_names, ridge_coefs):
    print(f"    {name:20s}: {coef:+.6f}")
print(f"    {'Intercept':20s}: {ridge_intercept:+.6f}")

print(f"""
  ◆ 최적 가중치 (scipy.optimize):""")
for name, w in zip(model_names, opt_w):
    print(f"    {name:20s}: {w:.4f}")

print(f"""
  submission.csv 생성 완료 ({len(submission):,} rows, probability float)
""")
