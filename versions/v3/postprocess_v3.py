"""
v3 Post-processing: .npy 파일 로드 → 스태킹 + 가중 앙상블 + 확률 교정 + submission
모델 재학습 없이 결과물 생성
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
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from tqdm.auto import tqdm

SEED = 42
N_FOLDS = 5
DATA_DIR = "."
np.random.seed(SEED)

# Load data
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
y = train["임신 성공 여부"].values
test_ids = test["ID"].values

# Load v3 predictions
oof_preds = np.load(f"{DATA_DIR}/oof_preds_v3.npy")
test_preds = np.load(f"{DATA_DIR}/test_preds_v3.npy")

MODEL_NAMES = [
    "CB_d6_s42", "CB_d8_s7", "CB_d4_s123",
    "LGB_nl63_s42", "LGB_nl127_s7", "LGB_nl31_s123",
    "XGB_d6_s42", "XGB_d8_s7", "XGB_d4_s123",
]
N_MODELS = len(MODEL_NAMES)
n_train = len(y)
n_test = len(test_ids)

# Fold indices (reproduce exactly)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_indices = list(skf.split(np.zeros(n_train), y))

print("=" * 70)
print("Level-1 성능 요약")
print("=" * 70)
for i, name in enumerate(MODEL_NAMES):
    auc = roc_auc_score(y, oof_preds[:, i])
    print(f"  {name:20s} OOF AUC: {auc:.6f}")

simple_avg = oof_preds.mean(axis=1)
simple_avg_auc = roc_auc_score(y, simple_avg)
print(f"\n  균등 평균 앙상블 OOF AUC: {simple_avg_auc:.6f}")

# ================================================================
# Stacking: RidgeClassifier Meta-Learner
# ================================================================
print("\n" + "=" * 70)
print("Stacking: RidgeClassifier Meta-Learner")
print("=" * 70)

oof_meta = np.zeros(n_train)
test_meta_preds = np.zeros(n_test)
meta_fold_aucs = []

for fold_idx, (tr_idx, val_idx) in enumerate(tqdm(fold_indices, desc="Stacking Meta")):
    X_meta_tr = oof_preds[tr_idx]
    X_meta_val = oof_preds[val_idx]
    y_meta_tr = y[tr_idx]
    y_meta_val = y[val_idx]
    
    base_ridge = RidgeClassifier(alpha=1.0, class_weight="balanced")
    meta_model = CalibratedClassifierCV(base_ridge, cv=3, method="sigmoid")
    meta_model.fit(X_meta_tr, y_meta_tr)
    
    val_meta_pred = meta_model.predict_proba(X_meta_val)[:, 1]
    test_meta_pred = meta_model.predict_proba(test_preds)[:, 1]
    
    oof_meta[val_idx] = val_meta_pred
    test_meta_preds += test_meta_pred / N_FOLDS
    
    fold_auc = roc_auc_score(y_meta_val, val_meta_pred)
    meta_fold_aucs.append(fold_auc)
    print(f"  Fold {fold_idx+1} Meta AUC: {fold_auc:.6f}")

stacking_oof_auc = roc_auc_score(y, oof_meta)
print(f"\n  Stacking (RidgeClassifier) OOF AUC: {stacking_oof_auc:.6f}")

# RidgeClassifier 가중치 리포트
final_ridge = RidgeClassifier(alpha=1.0, class_weight="balanced")
final_ridge.fit(oof_preds, y)
ridge_coefs = final_ridge.coef_.ravel()
ridge_intercept = final_ridge.intercept_.ravel()[0] if hasattr(final_ridge.intercept_, 'ravel') else final_ridge.intercept_

print(f"\n  ◆ RidgeClassifier 메타 모델 가중치:")
print(f"    {'모델':20s} | 가중치")
print(f"    {'─'*20}─┼─{'─'*12}")
for name, coef in zip(MODEL_NAMES, ridge_coefs):
    print(f"    {name:20s} | {coef:+.6f}")
print(f"    {'Intercept':20s} | {ridge_intercept:+.6f}")

# ================================================================
# scipy.optimize 가중 앙상블
# ================================================================
print("\n" + "=" * 70)
print("scipy.optimize 가중 앙상블")
print("=" * 70)

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

print(f"\n  최적 가중 앙상블 OOF AUC: {opt_auc:.6f}")
print(f"  최적 가중치:")
for name, w in zip(MODEL_NAMES, opt_w):
    print(f"    {name:20s}: {w:.4f}")

# ================================================================
# v2 대비 t-test
# ================================================================
print("\n" + "=" * 70)
print("v2 대비 성능 비교 (t-test)")
print("=" * 70)

V2_OOF_AUC = 0.740111

v2_oof_path = f"{DATA_DIR}/oof_catboost.npy"
if os.path.exists(v2_oof_path):
    oof_cb_v2 = np.load(f"{DATA_DIR}/oof_catboost.npy")
    oof_lgb_v2 = np.load(f"{DATA_DIR}/oof_lgbm.npy")
    oof_xgb_v2 = np.load(f"{DATA_DIR}/oof_xgb.npy")
    oof_v2 = 0.5 * oof_cb_v2 + 0.25 * oof_lgb_v2 + 0.25 * oof_xgb_v2
    
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
    
    t_stat1, p_val1 = stats.ttest_rel(v3_stack_fold_aucs, v2_fold_aucs)
    sig1 = "유의함 ✓" if p_val1 < 0.01 else "유의하지 않음 ✗"
    print(f"\n  ◆ v3 스태킹 vs v2: t={t_stat1:.4f}, p={p_val1:.6f} → {sig1}")
    
    t_stat2, p_val2 = stats.ttest_rel(v3_weighted_fold_aucs, v2_fold_aucs)
    sig2 = "유의함 ✓" if p_val2 < 0.01 else "유의하지 않음 ✗"
    print(f"  ◆ v3 가중앙상블 vs v2: t={t_stat2:.4f}, p={p_val2:.6f} → {sig2}")
else:
    print(f"  v2 OOF 예측 파일 없음 — 수치 비교만: v2={V2_OOF_AUC:.6f}")

# ================================================================
# 최종 Submission 생성
# ================================================================
print("\n" + "=" * 70)
print("최종 Submission 생성")
print("=" * 70)

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

# 확률 교정
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

mean_diff = abs(test_calibrated.mean() - target_rate)
if mean_diff > 0.005:
    scale_factor = target_rate / test_calibrated.mean()
    test_final = np.clip(test_calibrated * scale_factor, 0.0, 1.0)
    print(f"  추가 미세 보정 → Test 평균: {test_final.mean():.6f}")
else:
    test_final = test_calibrated

submission = pd.DataFrame({"ID": test_ids, "probability": test_final})
submission.to_csv(f"{DATA_DIR}/submission.csv", index=False)

print(f"\n  submission.csv 저장 완료!")
print(f"  행 수: {len(submission):,}")
print(f"  probability: mean={submission['probability'].mean():.6f}, "
      f"std={submission['probability'].std():.6f}")
print(f"  min={submission['probability'].min():.6f}, max={submission['probability'].max():.6f}")

# ================================================================
# 최종 요약
# ================================================================
print(f"""
{'='*70}
최종 요약
{'='*70}

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
for name, coef in zip(MODEL_NAMES, ridge_coefs):
    print(f"    {name:20s}: {coef:+.6f}")
print(f"    {'Intercept':20s}: {ridge_intercept:+.6f}")

print(f"""
  ◆ 최적 가중치 (scipy.optimize):""")
for name, w in zip(MODEL_NAMES, opt_w):
    print(f"    {name:20s}: {w:.4f}")

print(f"""
  submission.csv 생성 완료 ({len(submission):,} rows, probability float)
""")
