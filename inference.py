"""
inference.py
────────────
저장된 best_model.pth를 불러와 test.csv에 대한 확률값을 예측하고
제출 파일(version_1.csv)을 생성합니다.

실행 방법:
    python inference.py
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from preprocessing import run_preprocessing, split_X_y, get_feature_columns
from dataset import IVFDataset
from model import IVFModel


# ────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────

INFER_CONFIG = {
    'model_path':       'best_model.pth',
    'train_path':       './data/train.csv',
    'test_path':        './data/test.csv',
    'submission_path':  './data/sample_submission.csv',
    'output_path':      'version_1.csv',
    'batch_size':       512,
}


# ────────────────────────────────────────────────
# 추론 함수
# ────────────────────────────────────────────────

def load_model(checkpoint: dict, cat_vocab_sizes, num_feature_size, device):
    """체크포인트에서 모델 복원."""
    cfg = checkpoint.get('config', {})
    model = IVFModel(
        cat_vocab_sizes     = cat_vocab_sizes,
        num_feature_size    = num_feature_size,
        hidden_dim          = cfg.get('hidden_dim', 256),
        num_residual_blocks = cfg.get('num_residual', 3),
        dropout             = cfg.get('dropout', 0.3),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict(model, loader, device) -> np.ndarray:
    """DataLoader 전체에 대해 예측 확률 반환."""
    probs = []
    with torch.no_grad():
        for batch in loader:
            cat_batch, num_batch = batch[0].to(device), batch[1].to(device)
            out = model(cat_batch, num_batch).squeeze()
            out = torch.sigmoid(out)
            # 배치 크기 1일 때 dim 보정
            if out.dim() == 0:
                out = out.unsqueeze(0)
            probs.extend(out.cpu().numpy())
    return np.array(probs)


def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  장치: {device}")

    # 1. 데이터 로드
    print("\n📂 데이터 로드 중...")
    train_df = pd.read_csv(INFER_CONFIG['train_path'])
    test_df  = pd.read_csv(INFER_CONFIG['test_path'])
    submission = pd.read_csv(INFER_CONFIG['submission_path'])

    # 2. 전처리 (동일한 파이프라인, 같은 대회 규칙 적용)
    print("\n🔧 전처리 중...")
    train_proc, test_proc, _ = run_preprocessing(train_df, test_df, verbose=False)
    print("   전처리 완료!")

    # 3. 카테고리 vocab 크기 (train 기준 — 학습과 동일하게 맞춤)
    X_cat_train, X_num_train, _, cat_cols, num_cols = split_X_y(train_proc)
    X_cat_test,  X_num_test,  _, _,        _        = split_X_y(test_proc)

    cat_vocab_sizes = [int(train_proc[col].max()) + 1 for col in cat_cols]

    # 4. 체크포인트 로드
    print(f"\n📥 모델 로드: '{INFER_CONFIG['model_path']}'")
    try:
        checkpoint = torch.load(INFER_CONFIG['model_path'], map_location=device)
    except FileNotFoundError:
        print(f"❌ '{INFER_CONFIG['model_path']}' 파일이 없습니다. 먼저 main.py를 실행하세요.")
        return

    model = load_model(checkpoint, cat_vocab_sizes, len(num_cols), device)
    print(f"   Best Val AUC (학습 시): {checkpoint.get('val_auc', 'N/A'):.4f}")

    # 5. DataLoader
    test_dataset = IVFDataset(X_cat_test, X_num_test)
    test_loader  = DataLoader(test_dataset, batch_size=INFER_CONFIG['batch_size'], shuffle=False)

    # 6. 추론
    print("\n🚀 추론 중...")
    probs = predict(model, test_loader, device)
    print(f"   예측 완료! 샘플 수: {len(probs):,}")
    print(f"   확률값 범위: {probs.min():.4f} ~ {probs.max():.4f} (평균: {probs.mean():.4f})")

    # 7. 제출 파일 생성
    target_col = submission.columns[1]
    submission[target_col] = probs
    submission.to_csv(INFER_CONFIG['output_path'], index=False)
    print(f"\n✅ 제출 파일 저장 완료: '{INFER_CONFIG['output_path']}'")
    print(submission.head())


# ────────────────────────────────────────────────

if __name__ == "__main__":
    run_inference()
