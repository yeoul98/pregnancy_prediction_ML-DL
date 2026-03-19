import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# 데이터 로드
try:
    oof_preds = np.load("oof_preds_v3.npy")
    test_preds = np.load("test_preds_v3.npy")
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    y = train["임신 성공 여부"].values
    test_ids = test["ID"].values
    print(f"v3 데이터 로드 완료: OOF {oof_preds.shape}, Test {test_preds.shape}")
except FileNotFoundError as e:
    print(f"에러: v3 예측값 파일(.npy)을 찾을 수 없습니다. {e}")
    exit()

# Stratified K-Fold (v3와 동일한 SEED=42 가정)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Ridge 스태킹 (Meta-Learner)
# v3에서 검증된 Ridge 스태킹 로직 적용
oof_stacking = np.zeros(len(y))
test_stacking = np.zeros(len(test_ids))

print("Ridge Stacking (Meta-Learner) 학습 중...")
for fold, (tr_idx, val_idx) in enumerate(skf.split(oof_preds, y)):
    ridge = RidgeClassifier(random_state=42)
    # 확률값 출력을 위해 CalibratedClassifierCV 사용
    model = CalibratedClassifierCV(ridge, cv=3, method='sigmoid')
    model.fit(oof_preds[tr_idx], y[tr_idx])
    
    oof_stacking[val_idx] = model.predict_proba(oof_preds[val_idx])[:, 1]
    test_stacking += model.predict_proba(test_preds)[:, 1] / 5

stacking_auc = roc_auc_score(y, oof_stacking)
print(f"v3 Ridge Stacking OOF AUC: {stacking_auc:.6f}")

# 최종 제출 파일 생성
submission = pd.DataFrame({
    "ID": test_ids,
    "probability": test_stacking
})

output_file = "submission_stacking.csv"
submission.to_csv(output_file, index=False)
print(f"제출 파일 저장 완료: {output_file}")
print(f"행 수: {len(submission)}, 확률 평균: {submission['probability'].mean():.4f}")
