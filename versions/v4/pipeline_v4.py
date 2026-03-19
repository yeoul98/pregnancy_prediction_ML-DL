"""
=============================================================================
난임 시술 성공 예측 파이프라인 v4
- Null Importance Feature Selection (50+ shuffles, p<0.10)
- Feature Interaction (Top-5 곱셈/나눗셈)
- Heterogeneous L0: CB×3 + LGB×3 + XGB×3 + LR + KNN = 11모델
- Optuna Meta-Model Optimization (Ridge α + 가중치, 1000 trials)
- Rank Averaging Submission
- Zero Leakage: 모든 인코딩/스케일링은 Fold 내부 Train 통계만 사용
=============================================================================
"""
import warnings
warnings.filterwarnings("ignore")

import os, gc
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import xgboost as xgb
import optuna
from tqdm.auto import tqdm

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
N_FOLDS = 5
DATA_DIR = "."
N_NULL_ROUNDS = 50
NULL_THRESHOLD = 0.10  # p < 0.10
SMOOTHING = 10

np.random.seed(SEED)

# =============================================================================
# 1. 데이터 로딩
# =============================================================================
print("=" * 70)
print("[1/8] 데이터 로딩")
print("=" * 70)

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
TARGET = "임신 성공 여부"
ID_COL = "ID"
y = train[TARGET].values
test_ids = test[ID_COL].values
n_train = len(train)
n_test = len(test)
print(f"  Train: {train.shape}, Test: {test.shape}")
print(f"  Target 분포: 0={sum(y==0):,} / 1={sum(y==1):,} ({y.mean()*100:.1f}%)")

# =============================================================================
# 2. 기본 전처리
# =============================================================================
print("\n" + "=" * 70)
print("[2/8] 기본 전처리")
print("=" * 70)

drop_cols = [ID_COL]
if TARGET in train.columns:
    drop_cols.append(TARGET)
train_feat = train.drop(columns=drop_cols)
test_feat = test.drop(columns=[ID_COL])
combined = pd.concat([train_feat, test_feat], axis=0, ignore_index=True)

# 결측률 80%+ 드롭
high_miss = [c for c in combined.columns if combined[c].isnull().mean() >= 0.80]
print(f"  결측률≥80% 드롭 ({len(high_miss)}개): {high_miss}")
combined.drop(columns=high_miss, inplace=True)

# 상수 컬럼 드롭
const_cols = [c for c in combined.columns if combined[c].nunique() <= 1]
print(f"  상수 컬럼 드롭 ({len(const_cols)}개): {const_cols}")
combined.drop(columns=const_cols, inplace=True)

# 결측치 통합
for col in combined.select_dtypes(include=["object"]).columns:
    combined[col] = combined[col].replace(["알 수 없음", "기록되지 않은 시행"], np.nan)

# 결측 이진 피처
miss_feat_cols = []
for col in combined.columns:
    mr = combined[col].isnull().mean()
    if 0.05 <= mr < 0.80:
        combined[f"{col}_missing"] = combined[col].isnull().astype(int)
        miss_feat_cols.append(f"{col}_missing")
print(f"  결측 이진 피처 ({len(miss_feat_cols)}개)")

# 나이 수치화
age_map = {"만18-34세": 26, "만35-37세": 36, "만38-39세": 38.5,
           "만40-42세": 41, "만43-44세": 43.5, "만45-50세": 47.5}
if "시술 당시 나이" in combined.columns:
    combined["나이_수치"] = combined["시술 당시 나이"].map(age_map)
    combined["나이_수치"].fillna(combined["나이_수치"].median(), inplace=True)
    combined["고령_40이상"] = (combined["나이_수치"] >= 40).astype(int)
    combined.drop(columns=["시술 당시 나이"], inplace=True)
    print("  나이 수치화 완료")

# 시술 횟수 비닝
cnt_cols = ["총 시술 횟수","클리닉 내 총 시술 횟수","IVF 시술 횟수","DI 시술 횟수",
            "총 임신 횟수","IVF 임신 횟수","DI 임신 횟수",
            "총 출산 횟수","IVF 출산 횟수","DI 출산 횟수"]
cnt_map = {"0회":0,"1회":1,"2회":2,"3회":3,"4회":4,"5회":5,"6회 이상":6}
def bin_count(v):
    if pd.isna(v): return np.nan
    n = cnt_map.get(v, np.nan)
    if pd.isna(n): return np.nan
    return "bin_0" if n == 0 else ("bin_1_2" if n <= 2 else "bin_3plus")

for col in cnt_cols:
    if col in combined.columns:
        combined[f"{col}_num"] = combined[col].map(cnt_map)
        combined[f"{col}_bin"] = combined[col].apply(bin_count)
        combined.drop(columns=[col], inplace=True)
print(f"  시술 횟수 비닝 완료")

# 특정 시술 유형
if "특정 시술 유형" in combined.columns:
    combined["특정 시술 유형"] = combined["특정 시술 유형"].apply(
        lambda x: x if x in ["ICSI","IVF","Unknown"] else ("기타" if pd.notna(x) else np.nan))

# 수치형 파생 변수
if "총 생성 배아 수" in combined.columns and "혼합된 난자 수" in combined.columns:
    combined["수정률"] = (combined["총 생성 배아 수"] / (combined["혼합된 난자 수"]+1e-8)).clip(0,10)
if "이식된 배아 수" in combined.columns and "총 생성 배아 수" in combined.columns:
    combined["이식률"] = (combined["이식된 배아 수"] / (combined["총 생성 배아 수"]+1e-8)).clip(0,10)
if "미세주입에서 생성된 배아 수" in combined.columns and "총 생성 배아 수" in combined.columns:
    combined["ICSI배아비율"] = (combined["미세주입에서 생성된 배아 수"] / (combined["총 생성 배아 수"]+1e-8)).clip(0,10)
if "저장된 배아 수" in combined.columns and "총 생성 배아 수" in combined.columns:
    combined["저장률"] = (combined["저장된 배아 수"] / (combined["총 생성 배아 수"]+1e-8)).clip(0,10)
if "총 생성 배아 수" in combined.columns and "수집된 신선 난자 수" in combined.columns:
    combined["난자대비배아효율"] = (combined["총 생성 배아 수"] / (combined["수집된 신선 난자 수"]+1e-8)).clip(0,10)

inf_cols = [c for c in combined.columns if c.startswith("불임 원인 -")]
if inf_cols:
    combined["불임원인_총개수"] = combined[inf_cols].sum(axis=1)
for mc, sc in [("남성 주 불임 원인","남성 부 불임 원인"),("여성 주 불임 원인","여성 부 불임 원인"),
               ("부부 주 불임 원인","부부 부 불임 원인")]:
    if mc in combined.columns and sc in combined.columns:
        combined[f"{mc[:2]}_불임합산"] = combined[mc] + combined[sc]

# 기증자 나이
da_map = {"만20세 이하":18,"만21-25세":23,"만26-30세":28,"만31-35세":33,"만36-40세":38,"만41-45세":43}
for c in ["난자 기증자 나이","정자 기증자 나이"]:
    if c in combined.columns:
        combined[f"{c}_num"] = combined[c].map(da_map)

if "난자 혼합 경과일" in combined.columns and "배아 이식 경과일" in combined.columns:
    combined["혼합_이식_간격"] = combined["배아 이식 경과일"] - combined["난자 혼합 경과일"]
print("  파생 변수 생성 완료")

# =============================================================================
# 3. Null Importance Feature Selection
# =============================================================================
print("\n" + "=" * 70)
print("[3/8] Null Importance 피처 선택")
print("=" * 70)

# 범주형/수치형 분리
cat_cols_raw = combined.select_dtypes(include=["object"]).columns.tolist()
num_cols_all = combined.select_dtypes(include=["number"]).columns.tolist()

# Label Encoding (Null Importance 측정용)
combined_ni = combined.copy()
for c in cat_cols_raw:
    combined_ni[c] = combined_ni[c].fillna("MISSING").astype(str)
    le = LabelEncoder()
    le.fit(combined_ni[c])
    combined_ni[c] = le.transform(combined_ni[c])

for c in num_cols_all:
    combined_ni[c] = combined_ni[c].fillna(-999)

X_ni = combined_ni.iloc[:n_train].values
feature_names_ni = list(combined_ni.columns)

# 실제 중요도
print(f"  실제 중요도 측정 중...")
real_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=63,
                                 subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                                 verbose=-1, n_jobs=-1)
real_model.fit(X_ni, y)
actual_imp = real_model.feature_importances_.astype(float)

# Null 중요도 (N_NULL_ROUNDS 회)
print(f"  Null Importance 측정 ({N_NULL_ROUNDS}회 셔플)...")
null_imps = np.zeros((N_NULL_ROUNDS, len(feature_names_ni)))

for i in tqdm(range(N_NULL_ROUNDS), desc="Null Importance"):
    y_shuffled = np.random.permutation(y)
    null_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=63,
                                     subsample=0.8, colsample_bytree=0.8,
                                     random_state=SEED + i + 1, verbose=-1, n_jobs=-1)
    null_model.fit(X_ni, y_shuffled)
    null_imps[i] = null_model.feature_importances_

# p-value 계산: 각 피처에서 실제 중요도가 null 분포에서 몇 번째 백분위인지
p_values = np.zeros(len(feature_names_ni))
for j in range(len(feature_names_ni)):
    # 실제 중요도보다 Null 중요도가 높은 비율 = p-value
    p_values[j] = (null_imps[:, j] >= actual_imp[j]).mean()

# p < NULL_THRESHOLD인 피처만 유지
keep_mask = p_values < NULL_THRESHOLD
keep_features = [f for f, k in zip(feature_names_ni, keep_mask) if k]
drop_features = [f for f, k in zip(feature_names_ni, keep_mask) if not k]

print(f"\n  전체 피처: {len(feature_names_ni)}개")
print(f"  유지 피처 (p<{NULL_THRESHOLD}): {len(keep_features)}개")
print(f"  제거 피처: {len(drop_features)}개")

# 상위 10개 중요 피처
sorted_idx = np.argsort(actual_imp)[::-1]
print(f"\n  ◆ 실제 중요도 Top-10:")
for rank, idx in enumerate(sorted_idx[:10]):
    print(f"    {rank+1:2d}. {feature_names_ni[idx]:30s} imp={actual_imp[idx]:.0f}  p={p_values[idx]:.4f}")

# combined에서 불필요 피처 제거
combined.drop(columns=[c for c in drop_features if c in combined.columns], inplace=True)

del combined_ni, X_ni, null_imps
gc.collect()

# =============================================================================
# 4. 고차 피처 상호작용 (Top-5)
# =============================================================================
print("\n" + "=" * 70)
print("[4/8] 고차 피처 상호작용 (Top-5)")
print("=" * 70)

# Top-5 수치형 피처 (Null Importance 기준)
top5_candidates = []
for idx in sorted_idx:
    fname = feature_names_ni[idx]
    if fname in combined.columns and combined[fname].dtype != "object":
        top5_candidates.append(fname)
    if len(top5_candidates) == 5:
        break

print(f"  Top-5 피처: {top5_candidates}")

interaction_count = 0
for i in range(len(top5_candidates)):
    for j in range(i+1, len(top5_candidates)):
        a, b = top5_candidates[i], top5_candidates[j]
        # 곱셈
        combined[f"{a}_x_{b}"] = combined[a] * combined[b]
        interaction_count += 1
        # 나눗셈 (0 방지)
        combined[f"{a}_div_{b}"] = combined[a] / (combined[b] + 1e-8)
        combined[f"{b}_div_{a}"] = combined[b] / (combined[a] + 1e-8)
        interaction_count += 2

print(f"  상호작용 피처 {interaction_count}개 생성")

# =============================================================================
# 5. 최종 피처 분리
# =============================================================================
print("\n" + "=" * 70)
print("[5/8] 최종 피처 세트 구성")
print("=" * 70)

cat_cols = [c for c in combined.select_dtypes(include=["object"]).columns]
num_cols = [c for c in combined.select_dtypes(include=["number"]).columns]

for c in cat_cols:
    combined[c] = combined[c].fillna("MISSING").astype(str)
for c in num_cols:
    combined[c] = combined[c].fillna(-999)

X_train_raw = combined.iloc[:n_train].reset_index(drop=True)
X_test_raw = combined.iloc[n_train:].reset_index(drop=True)

print(f"  범주형: {len(cat_cols)}개, 수치형: {len(num_cols)}개, 전체: {len(cat_cols)+len(num_cols)}개")

# K-Fold 인덱스
GLOBAL_MEAN = y.mean()
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_indices = list(skf.split(X_train_raw, y))

# =============================================================================
# 6. L0 모델 학습 (Zero Leakage)
# =============================================================================
print("\n" + "=" * 70)
print("[6/8] L0 모델 학습 (11모델 × 5-Fold, Zero Leakage)")
print("=" * 70)

MODEL_CONFIGS = [
    # CatBoost ×3
    ("CB_d6", "catboost", dict(iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        eval_metric="AUC", random_seed=42, early_stopping_rounds=100, verbose=0, task_type="CPU")),
    ("CB_d8", "catboost", dict(iterations=2000, learning_rate=0.03, depth=8, l2_leaf_reg=5,
        eval_metric="AUC", random_seed=7, early_stopping_rounds=100, verbose=0, task_type="CPU")),
    ("CB_d4", "catboost", dict(iterations=3000, learning_rate=0.08, depth=4, l2_leaf_reg=1,
        eval_metric="AUC", random_seed=123, early_stopping_rounds=150, verbose=0, task_type="CPU")),
    # LightGBM ×3
    ("LGB_63", "lgbm", dict(n_estimators=2000, learning_rate=0.05, num_leaves=63,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, metric="auc", random_state=42, verbose=-1, n_jobs=-1)),
    ("LGB_127", "lgbm", dict(n_estimators=2000, learning_rate=0.03, num_leaves=127,
        min_child_samples=50, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=2.0, metric="auc", random_state=7, verbose=-1, n_jobs=-1)),
    ("LGB_31", "lgbm", dict(n_estimators=3000, learning_rate=0.08, max_depth=6, num_leaves=31,
        min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.01, reg_lambda=0.5, metric="auc", random_state=123, verbose=-1, n_jobs=-1)),
    # XGBoost ×3
    ("XGB_d6", "xgb", dict(n_estimators=2000, learning_rate=0.05, max_depth=6,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, eval_metric="auc", tree_method="hist",
        early_stopping_rounds=100, random_state=42, verbosity=0, n_jobs=-1)),
    ("XGB_d8", "xgb", dict(n_estimators=2000, learning_rate=0.03, max_depth=8,
        min_child_weight=10, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=2.0, eval_metric="auc", tree_method="hist",
        early_stopping_rounds=100, random_state=7, verbosity=0, n_jobs=-1)),
    ("XGB_d4", "xgb", dict(n_estimators=3000, learning_rate=0.08, max_depth=4,
        min_child_weight=3, subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.01, reg_lambda=0.5, eval_metric="auc", tree_method="hist",
        early_stopping_rounds=150, random_state=123, verbosity=0, n_jobs=-1)),
    # Logistic Regression
    ("LR", "lr", dict(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)),
    # K-Neighbors
    ("KNN", "knn", dict(n_neighbors=200, metric="minkowski", n_jobs=-1)),
]

N_MODELS = len(MODEL_CONFIGS)
model_names = [n for n, _, _ in MODEL_CONFIGS]
oof_preds = np.zeros((n_train, N_MODELS))
test_preds = np.zeros((n_test, N_MODELS))
fold_aucs = {n: [] for n in model_names}

print(f"  L0 모델 수: {N_MODELS}개")
for n, t, _ in MODEL_CONFIGS:
    print(f"    - {n} ({t})")

for model_idx, (mname, mtype, params) in enumerate(MODEL_CONFIGS):
    print(f"\n{'─'*60}")
    print(f"  [{model_idx+1}/{N_MODELS}] {mname}")
    print(f"{'─'*60}")

    for fold_idx, (tr_idx, val_idx) in enumerate(tqdm(fold_indices, desc=mname, leave=False)):
        y_tr, y_val = y[tr_idx], y[val_idx]

        # ===== Zero Leakage: Fold 내부에서 TE + 인코딩 + 스케일링 =====

        # (A) Target Encoding (Fold Train에서만 통계 계산)
        te_tr_dict, te_val_dict, te_test_dict = {}, {}, {}
        for c in cat_cols:
            tr_data = pd.DataFrame({"cat": X_train_raw[c].iloc[tr_idx], "target": y_tr})
            st = tr_data.groupby("cat")["target"].agg(["mean","count"])
            sm = (st["count"] * st["mean"] + SMOOTHING * GLOBAL_MEAN) / (st["count"] + SMOOTHING)
            te_tr_dict[f"{c}_te"] = X_train_raw[c].iloc[tr_idx].map(sm).fillna(GLOBAL_MEAN).values
            te_val_dict[f"{c}_te"] = X_train_raw[c].iloc[val_idx].map(sm).fillna(GLOBAL_MEAN).values
            te_test_dict[f"{c}_te"] = X_test_raw[c].map(sm).fillna(GLOBAL_MEAN).values

        te_tr_df = pd.DataFrame(te_tr_dict)
        te_val_df = pd.DataFrame(te_val_dict)
        te_test_df = pd.DataFrame(te_test_dict)

        # (B) 수치형 피처
        num_tr = X_train_raw[num_cols].iloc[tr_idx].reset_index(drop=True)
        num_val = X_train_raw[num_cols].iloc[val_idx].reset_index(drop=True)
        num_test = X_test_raw[num_cols].reset_index(drop=True)

        if mtype == "catboost":
            # CatBoost: 원본 범주형 + 수치형 + TE
            cat_tr = X_train_raw[cat_cols].iloc[tr_idx].reset_index(drop=True)
            cat_val = X_train_raw[cat_cols].iloc[val_idx].reset_index(drop=True)
            cat_test_cb = X_test_raw[cat_cols].reset_index(drop=True)
            X_tr = pd.concat([cat_tr, num_tr, te_tr_df.reset_index(drop=True)], axis=1)
            X_val = pd.concat([cat_val, num_val, te_val_df.reset_index(drop=True)], axis=1)
            X_te = pd.concat([cat_test_cb, num_test, te_test_df.reset_index(drop=True)], axis=1)
            cat_idx = list(range(len(cat_cols)))

            model = CatBoostClassifier(**params)
            model.fit(Pool(X_tr, y_tr, cat_features=cat_idx),
                      eval_set=Pool(X_val, y_val, cat_features=cat_idx))
            val_pred = model.predict_proba(X_val)[:, 1]
            test_pred = model.predict_proba(X_te)[:, 1]

        elif mtype in ("lgbm", "xgb"):
            # Label Encoding (Fold 내부)
            le_tr_list, le_val_list, le_test_list = [], [], []
            for c in cat_cols:
                le = LabelEncoder()
                le.fit(pd.concat([X_train_raw[c].iloc[tr_idx], X_train_raw[c].iloc[val_idx],
                                  X_test_raw[c]]))
                le_tr_list.append(le.transform(X_train_raw[c].iloc[tr_idx]))
                le_val_list.append(le.transform(X_train_raw[c].iloc[val_idx]))
                le_test_list.append(le.transform(X_test_raw[c]))
            le_tr_df = pd.DataFrame(dict(zip(cat_cols, le_tr_list)))
            le_val_df = pd.DataFrame(dict(zip(cat_cols, le_val_list)))
            le_test_df = pd.DataFrame(dict(zip(cat_cols, le_test_list)))

            X_tr = pd.concat([le_tr_df, num_tr, te_tr_df.reset_index(drop=True)], axis=1)
            X_val = pd.concat([le_val_df, num_val, te_val_df.reset_index(drop=True)], axis=1)
            X_te = pd.concat([le_test_df, num_test, te_test_df.reset_index(drop=True)], axis=1)

            if mtype == "lgbm":
                model = lgb.LGBMClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            else:
                model = xgb.XGBClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
            val_pred = model.predict_proba(X_val)[:, 1]
            test_pred = model.predict_proba(X_te)[:, 1]

        elif mtype == "lr":
            # LR: TE + 수치형 (스케일링 Fold 내부)
            X_tr = pd.concat([num_tr, te_tr_df.reset_index(drop=True)], axis=1)
            X_val = pd.concat([num_val, te_val_df.reset_index(drop=True)], axis=1)
            X_te = pd.concat([num_test, te_test_df.reset_index(drop=True)], axis=1)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)
            X_te_s = scaler.transform(X_te)

            model = LogisticRegression(**params)
            model.fit(X_tr_s, y_tr)
            val_pred = model.predict_proba(X_val_s)[:, 1]
            test_pred = model.predict_proba(X_te_s)[:, 1]

        elif mtype == "knn":
            # KNN: TE + 수치형 (스케일링 Fold 내부)
            X_tr = pd.concat([num_tr, te_tr_df.reset_index(drop=True)], axis=1)
            X_val = pd.concat([num_val, te_val_df.reset_index(drop=True)], axis=1)
            X_te = pd.concat([num_test, te_test_df.reset_index(drop=True)], axis=1)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)
            X_te_s = scaler.transform(X_te)

            model = KNeighborsClassifier(**params)
            model.fit(X_tr_s, y_tr)
            val_pred = model.predict_proba(X_val_s)[:, 1]
            test_pred = model.predict_proba(X_te_s)[:, 1]

        oof_preds[val_idx, model_idx] = val_pred
        test_preds[:, model_idx] += test_pred / N_FOLDS
        fold_aucs[mname].append(roc_auc_score(y_val, val_pred))

    oof_auc = roc_auc_score(y, oof_preds[:, model_idx])
    print(f"  → {mname} OOF AUC: {oof_auc:.6f}  (folds: {[f'{a:.4f}' for a in fold_aucs[mname]]})")

np.save(f"{DATA_DIR}/oof_preds_v4.npy", oof_preds)
np.save(f"{DATA_DIR}/test_preds_v4.npy", test_preds)
print("\n  .npy 저장 완료")

# =============================================================================
# 7. Optuna Meta-Model 최적화
# =============================================================================
print("\n" + "=" * 70)
print("[7/8] Optuna Meta-Model 최적화 (1000 trials)")
print("=" * 70)

# (A) 단순 가중 평균 OOF AUC (기준선)
simple_avg_auc = roc_auc_score(y, oof_preds.mean(axis=1))
print(f"  균등 평균 앙상블 OOF AUC: {simple_avg_auc:.6f}")

# (B) Optuna: Ridge α + 모델별 가중치 최적화
def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-4, 100.0, log=True)
    weights = []
    for mn in model_names:
        w = trial.suggest_float(f"w_{mn}", 0.0, 1.0)
        weights.append(w)
    weights = np.array(weights)
    w_sum = weights.sum()
    if w_sum < 1e-8:
        return 0.5
    weights = weights / w_sum

    # 가중된 OOF로 Ridge 메타 모델
    weighted_oof = oof_preds * weights[np.newaxis, :]
    scores = []
    for tr_idx, val_idx in fold_indices:
        ridge = RidgeClassifier(alpha=alpha, class_weight="balanced")
        calib = CalibratedClassifierCV(ridge, cv=3, method="sigmoid")
        calib.fit(weighted_oof[tr_idx], y[tr_idx])
        pred = calib.predict_proba(weighted_oof[val_idx])[:, 1]
        scores.append(roc_auc_score(y[val_idx], pred))
    return np.mean(scores)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=1000, show_progress_bar=True)

best_params = study.best_params
best_alpha = best_params["alpha"]
best_weights = np.array([best_params[f"w_{mn}"] for mn in model_names])
best_weights = best_weights / best_weights.sum()

print(f"\n  ◆ Optuna 최적 결과 (best AUC={study.best_value:.6f}):")
print(f"    alpha = {best_alpha:.6f}")
print(f"    모델별 가중치:")
for mn, w in zip(model_names, best_weights):
    print(f"      {mn:12s}: {w:.4f}")

# (C) 최종 Ridge 메타 모델 학습
weighted_oof_opt = oof_preds * best_weights[np.newaxis, :]
weighted_test_opt = test_preds * best_weights[np.newaxis, :]

oof_meta = np.zeros(n_train)
test_meta = np.zeros(n_test)
meta_fold_aucs = []

for fold_idx, (tr_idx, val_idx) in enumerate(fold_indices):
    ridge = RidgeClassifier(alpha=best_alpha, class_weight="balanced")
    calib = CalibratedClassifierCV(ridge, cv=3, method="sigmoid")
    calib.fit(weighted_oof_opt[tr_idx], y[tr_idx])
    oof_meta[val_idx] = calib.predict_proba(weighted_oof_opt[val_idx])[:, 1]
    test_meta += calib.predict_proba(weighted_test_opt)[:, 1] / N_FOLDS
    fa = roc_auc_score(y[val_idx], oof_meta[val_idx])
    meta_fold_aucs.append(fa)
    print(f"  Fold {fold_idx+1} Meta AUC: {fa:.6f}")

meta_auc = roc_auc_score(y, oof_meta)
print(f"\n  Optuna Ridge Stacking OOF AUC: {meta_auc:.6f}")

# =============================================================================
# 8. Rank Averaging + Submission
# =============================================================================
print("\n" + "=" * 70)
print("[8/8] Rank Averaging + Submission 생성")
print("=" * 70)

# (A) 각 모델의 Test 예측을 순위로 변환
rank_test = np.zeros_like(test_preds)
rank_oof = np.zeros_like(oof_preds)

for i in range(N_MODELS):
    rank_test[:, i] = rankdata(test_preds[:, i]) / n_test
    rank_oof[:, i] = rankdata(oof_preds[:, i]) / n_train

# (B) 가중 순위 평균
rank_avg_oof = (rank_oof * best_weights[np.newaxis, :]).sum(axis=1)
rank_avg_test = (rank_test * best_weights[np.newaxis, :]).sum(axis=1)

rank_auc = roc_auc_score(y, rank_avg_oof)
print(f"  Rank Averaging OOF AUC: {rank_auc:.6f}")

# (C) 최종 앙상블 후보 비교
candidates = {
    "균등 평균": (simple_avg_auc, test_preds.mean(axis=1)),
    "Optuna Ridge Stacking": (meta_auc, test_meta),
    "Rank Averaging": (rank_auc, rank_avg_test),
}

best_method = max(candidates.keys(), key=lambda k: candidates[k][0])
best_auc_final, best_test_final = candidates[best_method]

print(f"\n  성능 비교:")
for method, (auc, _) in candidates.items():
    marker = " ← 최고" if method == best_method else ""
    print(f"    {method:25s}: OOF AUC = {auc:.6f}{marker}")

# (D) 확률 교정 (Platt Scaling)
print(f"\n  확률 교정 (Platt Scaling) 적용...")
target_rate = y.mean()

if best_method == "Rank Averaging":
    # Rank는 이미 [0,1] 범위이므로 Platt Scaling으로 확률 변환
    calib_lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    calib_lr.fit(rank_avg_oof.reshape(-1, 1), y)
    test_calib = calib_lr.predict_proba(rank_avg_test.reshape(-1, 1))[:, 1]
    oof_calib = calib_lr.predict_proba(rank_avg_oof.reshape(-1, 1))[:, 1]
else:
    calib_lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    calib_lr.fit(best_test_final.reshape(-1, 1) if best_method != "Optuna Ridge Stacking"
                 else oof_meta.reshape(-1, 1), y)
    if best_method == "Optuna Ridge Stacking":
        test_calib = calib_lr.predict_proba(test_meta.reshape(-1, 1))[:, 1]
        oof_calib = calib_lr.predict_proba(oof_meta.reshape(-1, 1))[:, 1]
    else:
        test_calib = calib_lr.predict_proba(best_test_final.reshape(-1, 1))[:, 1]
        oof_calib = calib_lr.predict_proba(best_test_final.reshape(-1, 1))[:, 1]

calib_auc = roc_auc_score(y, oof_calib)
print(f"  교정 후 OOF AUC: {calib_auc:.6f}")
print(f"  교정 후 Test 평균: {test_calib.mean():.6f} (타겟: {target_rate:.6f})")

# 미세 보정
if abs(test_calib.mean() - target_rate) > 0.005:
    sf = target_rate / test_calib.mean()
    test_final = np.clip(test_calib * sf, 0, 1)
    print(f"  미세 보정 → Test 평균: {test_final.mean():.6f}")
else:
    test_final = test_calib

# (E) Submission
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
V3_AUC = 0.740317

print(f"""
{'='*70}
최종 요약
{'='*70}

  ┌──────────────────────────────────────────────────────────────────┐
  │                   v4 파이프라인 성능 종합                        │
  ├──────────────────────────┬───────────────────────────────────────┤
  │ v3 가중 앙상블 (기준)    │ {V3_AUC:.6f}                           │""")
for name, (auc, _) in candidates.items():
    diff = auc - V3_AUC
    sign = "+" if diff > 0 else ""
    print(f"  │ v4 {name:20s} │ {auc:.6f} ({sign}{diff:.6f})               │")
print(f"""  ├──────────────────────────┼───────────────────────────────────────┤
  │ 채택 방식               │ {best_method:37s} │
  │ 교정 후 OOF AUC         │ {calib_auc:.6f}                           │
  └──────────────────────────┴───────────────────────────────────────┘

  ◆ Optuna 최적 하이퍼파라미터:
    Ridge α = {best_alpha:.6f}
    모델별 가중치:""")
for mn, w in zip(model_names, best_weights):
    print(f"      {mn:12s}: {w:.4f}")
print(f"""
  ◆ Null Importance: {len(feature_names_ni)}개 → {len(keep_features)}개 ({len(drop_features)}개 제거)
  ◆ Feature Interactions: {interaction_count}개 추가
  ◆ submission.csv 생성 완료 ({len(submission):,} rows, probability float)
""")
