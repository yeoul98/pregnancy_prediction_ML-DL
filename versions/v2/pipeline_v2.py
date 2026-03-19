"""
=============================================================================
난임 시술 성공 예측 파이프라인
- Adversarial Validation AUC ~0.5 → OOF-AUC ≈ LB 점수
- CatBoost / LightGBM / XGBoost 앙상블
- Stratified 5-Fold CV + t-test 기반 모델 비교
=============================================================================
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
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
print("=" * 60)
print("[1/6] 데이터 로딩")
print("=" * 60)

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
# 2. 전처리 함수
# =============================================================================
print("\n" + "=" * 60)
print("[2/6] 데이터 전처리 & 피처 엔지니어링")
print("=" * 60)

# ID, 타겟 제거
drop_cols_basic = [ID_COL]
if TARGET in train.columns:
    drop_cols_basic.append(TARGET)

train_feat = train.drop(columns=drop_cols_basic)
test_feat = test.drop(columns=[ID_COL])

# Train+Test 합치기
combined = pd.concat([train_feat, test_feat], axis=0, ignore_index=True)
n_train = len(train_feat)
n_test = len(test_feat)

# ---------------------------------------------------------------------------
# 2-1. 결측률 80% 이상 컬럼 드롭
# ---------------------------------------------------------------------------
high_missing_cols = []
for col in combined.columns:
    miss_rate = combined[col].isnull().sum() / len(combined)
    if miss_rate >= 0.80:
        high_missing_cols.append(col)

print(f"  결측률 ≥80% 드롭 ({len(high_missing_cols)}개): {high_missing_cols}")
combined.drop(columns=high_missing_cols, inplace=True)

# ---------------------------------------------------------------------------
# 2-2. 상수 컬럼 드롭 (nunique=1)
# ---------------------------------------------------------------------------
const_cols = [c for c in combined.columns if combined[c].nunique() <= 1]
print(f"  상수 컬럼 드롭 ({len(const_cols)}개): {const_cols}")
combined.drop(columns=const_cols, inplace=True)

# ---------------------------------------------------------------------------
# 2-3. '알 수 없음', '기록되지 않은 시행' → NaN 통합
# ---------------------------------------------------------------------------
unknown_values = ["알 수 없음", "기록되지 않은 시행"]
for col in combined.select_dtypes(include=["object"]).columns:
    combined[col] = combined[col].replace(unknown_values, np.nan)

# ---------------------------------------------------------------------------
# 2-4. 결측 여부 이진 피처 (결측률 5%~80%)
# ---------------------------------------------------------------------------
missing_feat_cols = []
for col in combined.columns:
    miss_rate = combined[col].isnull().sum() / len(combined)
    if 0.05 <= miss_rate < 0.80:
        feat_name = f"{col}_missing"
        combined[feat_name] = combined[col].isnull().astype(int)
        missing_feat_cols.append(feat_name)

print(f"  결측 이진 피처 생성 ({len(missing_feat_cols)}개)")

# ---------------------------------------------------------------------------
# 2-5. 시술 당시 나이 → 중앙값 수치화 + 고령 여부
# ---------------------------------------------------------------------------
age_map = {
    "만18-34세": 26,
    "만35-37세": 36,
    "만38-39세": 38.5,
    "만40-42세": 41,
    "만43-44세": 43.5,
    "만45-50세": 47.5,
}
age_col = "시술 당시 나이"
if age_col in combined.columns:
    combined["나이_수치"] = combined[age_col].map(age_map)
    # NaN인 경우 (알 수 없음 → NaN) → 전체 중위값
    median_age = combined["나이_수치"].median()
    combined["나이_수치"].fillna(median_age, inplace=True)
    combined["고령_40이상"] = (combined["나이_수치"] >= 40).astype(int)
    combined.drop(columns=[age_col], inplace=True)
    print("  나이 수치화 + 고령 여부 피처 생성 완료")

# ---------------------------------------------------------------------------
# 2-6. 시술 횟수 계열 비닝 (0회/1-2회/3회이상)
# ---------------------------------------------------------------------------
count_cols = [
    "총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
    "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수"
]

count_value_map = {
    "0회": 0, "1회": 1, "2회": 2, "3회": 3, "4회": 4, "5회": 5, "6회 이상": 6
}

def bin_count(val):
    """0회/1-2회/3회이상 비닝"""
    if pd.isna(val):
        return np.nan
    num = count_value_map.get(val, np.nan)
    if pd.isna(num):
        return np.nan
    if num == 0:
        return "bin_0"
    elif num <= 2:
        return "bin_1_2"
    else:
        return "bin_3plus"

for col in count_cols:
    if col in combined.columns:
        # 원본 수치화 컬럼도 유지
        combined[f"{col}_num"] = combined[col].map(count_value_map)
        # 비닝 범주
        combined[f"{col}_bin"] = combined[col].apply(bin_count)
        combined.drop(columns=[col], inplace=True)

print(f"  시술 횟수 비닝 완료 ({len(count_cols)}개 컬럼)")

# ---------------------------------------------------------------------------
# 2-7. 특정 시술 유형 상위 범주 통합
# ---------------------------------------------------------------------------
specific_type_col = "특정 시술 유형"
if specific_type_col in combined.columns:
    top_types = ["ICSI", "IVF", "Unknown"]
    combined[specific_type_col] = combined[specific_type_col].apply(
        lambda x: x if x in top_types else ("기타" if pd.notna(x) else np.nan)
    )

# ---------------------------------------------------------------------------
# 2-8. 수치형 파생 변수
# ---------------------------------------------------------------------------
# 수정률: 생성 배아 / 혼합 난자
if "총 생성 배아 수" in combined.columns and "혼합된 난자 수" in combined.columns:
    combined["수정률"] = combined["총 생성 배아 수"] / (combined["혼합된 난자 수"] + 1e-8)
    combined["수정률"] = combined["수정률"].clip(0, 10)  # 이상치 클리핑

# 이식률: 이식 배아 / 생성 배아
if "이식된 배아 수" in combined.columns and "총 생성 배아 수" in combined.columns:
    combined["이식률"] = combined["이식된 배아 수"] / (combined["총 생성 배아 수"] + 1e-8)
    combined["이식률"] = combined["이식률"].clip(0, 10)

# ICSI 비율: ICSI 배아 / 전체 배아
if "미세주입에서 생성된 배아 수" in combined.columns and "총 생성 배아 수" in combined.columns:
    combined["ICSI배아비율"] = combined["미세주입에서 생성된 배아 수"] / (combined["총 생성 배아 수"] + 1e-8)
    combined["ICSI배아비율"] = combined["ICSI배아비율"].clip(0, 10)

# 저장률: 저장 배아 / 생성 배아
if "저장된 배아 수" in combined.columns and "총 생성 배아 수" in combined.columns:
    combined["저장률"] = combined["저장된 배아 수"] / (combined["총 생성 배아 수"] + 1e-8)
    combined["저장률"] = combined["저장률"].clip(0, 10)

# 난자 대비 배아 효율
if "총 생성 배아 수" in combined.columns and "수집된 신선 난자 수" in combined.columns:
    combined["난자대비배아효율"] = combined["총 생성 배아 수"] / (combined["수집된 신선 난자 수"] + 1e-8)
    combined["난자대비배아효율"] = combined["난자대비배아효율"].clip(0, 10)

print("  수치형 파생 변수 생성 완료")

# ---------------------------------------------------------------------------
# 2-9. 불임 원인 합산
# ---------------------------------------------------------------------------
infertility_cols = [c for c in combined.columns if c.startswith("불임 원인 -")]
if infertility_cols:
    combined["불임원인_총개수"] = combined[infertility_cols].sum(axis=1)
    print(f"  불임 원인 합산 피처 생성 (총 {len(infertility_cols)}개 원인)")

# 남성/여성/부부 불임 관련
cause_pairs = [
    ("남성 주 불임 원인", "남성 부 불임 원인"),
    ("여성 주 불임 원인", "여성 부 불임 원인"),
    ("부부 주 불임 원인", "부부 부 불임 원인"),
]
for main_c, sub_c in cause_pairs:
    if main_c in combined.columns and sub_c in combined.columns:
        combined[f"{main_c[:2]}_불임합산"] = combined[main_c] + combined[sub_c]

# ---------------------------------------------------------------------------
# 2-10. 난자 기증자/정자 기증자 나이 수치화
# ---------------------------------------------------------------------------
donor_age_map = {
    "만20세 이하": 18,
    "만21-25세": 23,
    "만26-30세": 28,
    "만31-35세": 33,
    "만36-40세": 38,
    "만41-45세": 43,
}
for col in ["난자 기증자 나이", "정자 기증자 나이"]:
    if col in combined.columns:
        combined[f"{col}_num"] = combined[col].map(donor_age_map)
        # 원본은 범주형으로 유지 (CatBoost용)

# ---------------------------------------------------------------------------
# 2-11. 경과일 파생 변수
# ---------------------------------------------------------------------------
day_cols = ["난자 채취 경과일", "난자 혼합 경과일", "배아 이식 경과일", "배아 해동 경과일"]
available_day_cols = [c for c in day_cols if c in combined.columns]
if len(available_day_cols) >= 2:
    if "난자 혼합 경과일" in combined.columns and "배아 이식 경과일" in combined.columns:
        combined["혼합_이식_간격"] = combined["배아 이식 경과일"] - combined["난자 혼합 경과일"]

print("  기타 파생 변수 생성 완료")

# ---------------------------------------------------------------------------
# 2-12. 범주형/수치형 분리 + 인코딩
# ---------------------------------------------------------------------------
cat_columns = combined.select_dtypes(include=["object"]).columns.tolist()
num_columns = combined.select_dtypes(include=["number"]).columns.tolist()

print(f"\n  범주형 피처: {len(cat_columns)}개")
print(f"  수치형 피처: {len(num_columns)}개")
print(f"  전체 피처: {len(combined.columns)}개")

# 범주형 NaN → "MISSING" 문자열 (CatBoost가 문자열로 처리)
for col in cat_columns:
    combined[col] = combined[col].fillna("MISSING").astype(str)

# 수치형 NaN → -999 (tree 모델에서 분기 가능)
for col in num_columns:
    combined[col] = combined[col].fillna(-999)

# =============================================================================
# 3. Train/Test 분리
# =============================================================================
X_train = combined.iloc[:n_train].reset_index(drop=True)
X_test = combined.iloc[n_train:].reset_index(drop=True)

feature_names = list(X_train.columns)
cat_indices = [i for i, c in enumerate(feature_names) if c in cat_columns]
cat_feature_names = [c for c in feature_names if c in cat_columns]

print(f"\n  최종 X_train: {X_train.shape}, X_test: {X_test.shape}")

# =============================================================================
# LightGBM/XGBoost용 Label Encoding
# =============================================================================
from sklearn.preprocessing import LabelEncoder

le_dict = {}
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

for col in cat_columns:
    le = LabelEncoder()
    le.fit(combined[col])
    X_train_encoded[col] = le.transform(X_train_encoded[col])
    X_test_encoded[col] = le.transform(X_test_encoded[col])
    le_dict[col] = le

# =============================================================================
# 4. 모델 학습 (Stratified 5-Fold)
# =============================================================================
print("\n" + "=" * 60)
print("[3/6] 모델 학습 (Stratified 5-Fold)")
print("=" * 60)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# OOF 예측 저장
oof_catboost = np.zeros(n_train)
oof_lgbm = np.zeros(n_train)
oof_xgb = np.zeros(n_train)

# Test 예측 저장
test_catboost = np.zeros(n_test)
test_lgbm = np.zeros(n_test)
test_xgb = np.zeros(n_test)

# Fold별 AUC
fold_auc_catboost = []
fold_auc_lgbm = []
fold_auc_xgb = []

for fold_idx, (tr_idx, val_idx) in enumerate(tqdm(skf.split(X_train, y), total=N_FOLDS, desc="Folds")):
    print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")

    # CatBoost용 데이터
    X_tr_cat, X_val_cat = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    # LGB/XGB용 데이터
    X_tr_enc, X_val_enc = X_train_encoded.iloc[tr_idx], X_train_encoded.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    # ----- CatBoost -----
    cb_model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        eval_metric="AUC",
        random_seed=SEED,
        early_stopping_rounds=100,
        verbose=100,
        task_type="CPU",
    )

    train_pool = Pool(X_tr_cat, y_tr, cat_features=cat_indices)
    val_pool = Pool(X_val_cat, y_val, cat_features=cat_indices)

    cb_model.fit(train_pool, eval_set=val_pool)

    cb_val_pred = cb_model.predict_proba(X_val_cat)[:, 1]
    cb_test_pred = cb_model.predict_proba(X_test)[:, 1]

    oof_catboost[val_idx] = cb_val_pred
    test_catboost += cb_test_pred / N_FOLDS

    cb_auc = roc_auc_score(y_val, cb_val_pred)
    fold_auc_catboost.append(cb_auc)
    print(f"  CatBoost Fold {fold_idx+1} AUC: {cb_auc:.6f}")

    # ----- LightGBM -----
    lgb_model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        metric="auc",
        random_state=SEED,
        verbose=-1,
        n_jobs=-1,
    )

    lgb_model.fit(
        X_tr_enc, y_tr,
        eval_set=[(X_val_enc, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100),
        ],
        categorical_feature=cat_feature_names,
    )

    lgb_val_pred = lgb_model.predict_proba(X_val_enc)[:, 1]
    lgb_test_pred = lgb_model.predict_proba(X_test_encoded)[:, 1]

    oof_lgbm[val_idx] = lgb_val_pred
    test_lgbm += lgb_test_pred / N_FOLDS

    lgb_auc = roc_auc_score(y_val, lgb_val_pred)
    fold_auc_lgbm.append(lgb_auc)
    print(f"  LightGBM Fold {fold_idx+1} AUC: {lgb_auc:.6f}")

    # ----- XGBoost -----
    xgb_model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="auc",
        tree_method="hist",
        enable_categorical=False,
        early_stopping_rounds=100,
        random_state=SEED,
        verbosity=0,
        n_jobs=-1,
    )

    xgb_model.fit(
        X_tr_enc, y_tr,
        eval_set=[(X_val_enc, y_val)],
        verbose=100,
    )

    xgb_val_pred = xgb_model.predict_proba(X_val_enc)[:, 1]
    xgb_test_pred = xgb_model.predict_proba(X_test_encoded)[:, 1]

    oof_xgb[val_idx] = xgb_val_pred
    test_xgb += xgb_test_pred / N_FOLDS

    xgb_auc = roc_auc_score(y_val, xgb_val_pred)
    fold_auc_xgb.append(xgb_auc)
    print(f"  XGBoost  Fold {fold_idx+1} AUC: {xgb_auc:.6f}")

# =============================================================================
# 4-1. OOF/Test 예측값 저장 (.npy)
# =============================================================================
print("\n  OOF/Test 예측값 .npy 저장 중...")
np.save(f"{DATA_DIR}/oof_catboost.npy", oof_catboost)
np.save(f"{DATA_DIR}/oof_lgbm.npy", oof_lgbm)
np.save(f"{DATA_DIR}/oof_xgb.npy", oof_xgb)
np.save(f"{DATA_DIR}/test_catboost.npy", test_catboost)
np.save(f"{DATA_DIR}/test_lgbm.npy", test_lgbm)
np.save(f"{DATA_DIR}/test_xgb.npy", test_xgb)
np.save(f"{DATA_DIR}/y_train.npy", y)
print("  저장 완료: oof_*.npy, test_*.npy, y_train.npy")

# =============================================================================
# 5. OOF 성능 비교 + t-test
# =============================================================================
print("\n" + "=" * 60)
print("[4/6] OOF 성능 비교 및 t-test")
print("=" * 60)

oof_auc_catboost = roc_auc_score(y, oof_catboost)
oof_auc_lgbm = roc_auc_score(y, oof_lgbm)
oof_auc_xgb = roc_auc_score(y, oof_xgb)

print(f"\n  전체 OOF AUC:")
print(f"    CatBoost : {oof_auc_catboost:.6f}  (folds: {[f'{a:.4f}' for a in fold_auc_catboost]})")
print(f"    LightGBM : {oof_auc_lgbm:.6f}  (folds: {[f'{a:.4f}' for a in fold_auc_lgbm]})")
print(f"    XGBoost  : {oof_auc_xgb:.6f}  (folds: {[f'{a:.4f}' for a in fold_auc_xgb]})")

# t-test (paired)
print(f"\n  ◆ t-test (p-value < 0.01 시 유의한 차이)")

pairs = [
    ("CatBoost vs LightGBM", fold_auc_catboost, fold_auc_lgbm),
    ("CatBoost vs XGBoost", fold_auc_catboost, fold_auc_xgb),
    ("LightGBM vs XGBoost", fold_auc_lgbm, fold_auc_xgb),
]
for name, a, b in pairs:
    t_stat, p_val = stats.ttest_rel(a, b)
    sig = "유의함 ✓" if p_val < 0.01 else "유의하지 않음 ✗"
    print(f"    {name}: t={t_stat:.4f}, p={p_val:.6f} → {sig}")

# =============================================================================
# 6. 앙상블 + 가중치 최적화 (scipy.optimize) + 확률 교정
# =============================================================================
print("\n" + "=" * 60)
print("[5/6] 앙상블 (scipy.optimize 가중치 최적화 + 확률 교정)")
print("=" * 60)

# --- 6-1. 균등 앙상블 기준선 ---
oof_ensemble = (oof_catboost + oof_lgbm + oof_xgb) / 3
oof_auc_ensemble = roc_auc_score(y, oof_ensemble)
print(f"\n  균등 앙상블 OOF AUC: {oof_auc_ensemble:.6f}")

# --- 6-2. scipy.optimize.minimize로 최적 가중치 탐색 ---
print("\n  scipy.optimize.minimize 가중치 최적화 중...")

oof_stack = np.column_stack([oof_catboost, oof_lgbm, oof_xgb])
test_stack = np.column_stack([test_catboost, test_lgbm, test_xgb])

def neg_auc_objective(weights):
    """가중합 OOF 예측의 -AUC를 반환 (최소화 → AUC 최대화)"""
    w = weights / weights.sum()  # 합이 1이 되도록 정규화
    blended = oof_stack @ w
    return -roc_auc_score(y, blended)

# 여러 초기값에서 시작하여 전역 최적에 가까운 해를 탐색
best_result = None
best_neg_auc = 0

initial_guesses = [
    [1/3, 1/3, 1/3],
    [0.5, 0.25, 0.25],
    [0.25, 0.5, 0.25],
    [0.25, 0.25, 0.5],
    [0.6, 0.2, 0.2],
    [0.2, 0.6, 0.2],
    [0.2, 0.2, 0.6],
    [0.4, 0.4, 0.2],
    [0.4, 0.2, 0.4],
    [0.2, 0.4, 0.4],
    [0.7, 0.15, 0.15],
    [0.15, 0.7, 0.15],
    [0.15, 0.15, 0.7],
    [0.5, 0.35, 0.15],
    [0.5, 0.15, 0.35],
    [0.35, 0.5, 0.15],
]

bounds = [(0.01, 0.99)] * 3  # 각 가중치 0.01~0.99
constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}

for x0 in tqdm(initial_guesses, desc="Weight Optimization"):
    result = minimize(
        neg_auc_objective,
        x0=np.array(x0),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if best_result is None or result.fun < best_neg_auc:
        best_neg_auc = result.fun
        best_result = result

optimal_w = best_result.x / best_result.x.sum()
optimal_auc = -best_neg_auc

print(f"\n  최적 가중치 (scipy.optimize):")
print(f"    CatBoost = {optimal_w[0]:.6f}")
print(f"    LightGBM = {optimal_w[1]:.6f}")
print(f"    XGBoost  = {optimal_w[2]:.6f}")
print(f"  최적 가중 앙상블 OOF AUC: {optimal_auc:.6f}")

# --- 6-3. 최적 가중 앙상블 적용 ---
oof_final = oof_stack @ optimal_w
test_final = test_stack @ optimal_w

print(f"\n  교정 전 Test probability 통계:")
print(f"    mean={test_final.mean():.6f}, std={test_final.std():.6f}")
print(f"    min={test_final.min():.6f}, max={test_final.max():.6f}")

# --- 6-4. 확률 교정 (Calibration) ---
# 목표: 앙상블 평균 확률이 학습 타겟 비율(25.8%)과 일치하도록 보정
# Platt Scaling (로지스틱 회귀 기반 교정)
print("\n  확률 교정 (Platt Scaling) 적용 중...")

from sklearn.linear_model import LogisticRegression

target_rate = y.mean()
print(f"  타겟 비율: {target_rate:.6f} ({target_rate*100:.2f}%)")
print(f"  교정 전 OOF 평균: {oof_final.mean():.6f}")
print(f"  교정 전 Test 평균: {test_final.mean():.6f}")

# Platt scaling: OOF 확률을 로짓으로 변환하여 LogisticRegression으로 교정
calibrator = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
calibrator.fit(oof_final.reshape(-1, 1), y)

oof_calibrated = calibrator.predict_proba(oof_final.reshape(-1, 1))[:, 1]
test_calibrated = calibrator.predict_proba(test_final.reshape(-1, 1))[:, 1]

calib_oof_auc = roc_auc_score(y, oof_calibrated)
print(f"\n  교정 후 OOF AUC: {calib_oof_auc:.6f} (교정 전: {optimal_auc:.6f})")
print(f"  교정 후 OOF 평균: {oof_calibrated.mean():.6f}")
print(f"  교정 후 Test 평균: {test_calibrated.mean():.6f}")

# AUC는 순서 불변이므로 동일해야 함. 교정으로 평균 확률이 타겟 비율에 수렴한다.
mean_diff = abs(test_calibrated.mean() - target_rate)
print(f"  Test 평균과 타겟 비율 차이: {mean_diff:.6f}")

# 미세 조정: 교정 후에도 평균이 타겟 비율과 다르면 추가 보정
if mean_diff > 0.005:
    print("  추가 미세 보정 적용 (isotonic adjustment)...")
    # 단순 선형 스케일링: 평균을 타겟 비율에 맞춤
    scale_factor = target_rate / test_calibrated.mean()
    test_calibrated_adj = test_calibrated * scale_factor
    test_calibrated_adj = np.clip(test_calibrated_adj, 0.0, 1.0)
    print(f"  보정 후 Test 평균: {test_calibrated_adj.mean():.6f}")
    test_submission = test_calibrated_adj
else:
    test_submission = test_calibrated

# --- 6-5. submission.csv 생성 ---
submission = pd.DataFrame({
    "ID": test_ids,
    "probability": test_submission
})

submission.to_csv(f"{DATA_DIR}/submission.csv", index=False)

print(f"\n  submission.csv 저장 완료!")
print(f"  행 수: {len(submission):,}")
print(f"  probability 통계: mean={submission['probability'].mean():.6f}, "
      f"std={submission['probability'].std():.6f}")
print(f"  min={submission['probability'].min():.6f}, max={submission['probability'].max():.6f}")

# =============================================================================
# 7. 최종 요약
# =============================================================================
print("\n" + "=" * 60)
print("[6/6] 최종 요약")
print("=" * 60)

print(f"""
  ┌──────────────────────────────────────────────────────────┐
  │               모델별 OOF AUC 비교                       │
  ├────────────────────┬─────────────────────────────────────┤
  │ CatBoost           │ {oof_auc_catboost:.6f}                           │
  │ LightGBM           │ {oof_auc_lgbm:.6f}                           │
  │ XGBoost            │ {oof_auc_xgb:.6f}                           │
  │ 균등 앙상블        │ {oof_auc_ensemble:.6f}                           │
  │ 최적 가중 앙상블   │ {optimal_auc:.6f}                           │
  │ 교정 후 AUC        │ {calib_oof_auc:.6f}                           │
  └────────────────────┴─────────────────────────────────────┘

  최적 가중치: CB={optimal_w[0]:.4f}, LGB={optimal_w[1]:.4f}, XGB={optimal_w[2]:.4f}
  확률 교정: Platt Scaling 적용
  Test 평균 확률: {test_submission.mean():.4f} (타겟 비율: {target_rate:.4f})
  submission.csv 생성 완료 ({len(submission):,} rows, probability float)
""")
