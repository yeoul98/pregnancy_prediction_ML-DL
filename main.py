"""
main.py
───────
학습 실행 스크립트.

실행 방법:
    python main.py

주요 기능:
  - 전처리 파이프라인 실행
  - 학습 / 검증 분리 (8:2)
  - Early Stopping (Val ROC-AUC 기준)
  - Best 모델 자동 저장
  - Class Imbalance 대응 (pos_weight)
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score

from preprocessing import run_preprocessing, split_X_y, get_feature_columns
from dataset import IVFDataset
from model import IVFModel


# ────────────────────────────────────────────────
# 0. 설정
# ────────────────────────────────────────────────

CONFIG = {
    'seed':              42,
    'batch_size':        256,
    'epochs':            20,
    'lr':                1e-3,
    'weight_decay':      1e-4,
    'val_ratio':         0.2,
    'hidden_dim':        256,
    'num_residual':      3,
    'dropout':           0.3,
    'early_stop_patience': 7,       # Val AUC 개선 없으면 중단
    'model_save_path':   'best_model.pth',
    'train_path':        './data/train.csv',
    'test_path':         './data/test.csv',
}


# ────────────────────────────────────────────────
# 1. 재현성 설정
# ────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ────────────────────────────────────────────────
# 2. 데이터 로드
# ────────────────────────────────────────────────

def load_data():
    print("📂 데이터 로드 중...")
    try:
        train_df = pd.read_csv(CONFIG['train_path'])
        test_df  = pd.read_csv(CONFIG['test_path'])
        print(f"   Train: {train_df.shape} | Test: {test_df.shape}")
        return train_df, test_df
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        raise


# ────────────────────────────────────────────────
# 3. DataLoader 생성
# ────────────────────────────────────────────────

def build_dataloaders(train_df: pd.DataFrame):
    """학습/검증 DataLoader 생성 (8:2 분리)."""
    X_cat, X_num, y, _, _ = split_X_y(train_df)
    full_dataset = IVFDataset(X_cat, X_num, y)

    n_val   = int(len(full_dataset) * CONFIG['val_ratio'])
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(CONFIG['seed'])
    )

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG['batch_size'], shuffle=False,
                              num_workers=0, pin_memory=True)

    print(f"   Train samples: {n_train:,} | Val samples: {n_val:,}")
    return train_loader, val_loader


# ────────────────────────────────────────────────
# 4. 모델 초기화
# ────────────────────────────────────────────────

def build_model(train_df: pd.DataFrame, device: torch.device) -> IVFModel:
    """카테고리 vocab 크기를 train 기준으로 계산해 모델 생성."""
    X_cat, X_num, _, cat_cols, num_cols = split_X_y(train_df)

    # ✅ vocab_size: train에서 인코딩된 값의 최대값 기준 (안전하게 +1)
    cat_vocab_sizes = [int(train_df[col].max()) + 1 for col in cat_cols]

    model = IVFModel(
        cat_vocab_sizes     = cat_vocab_sizes,
        num_feature_size    = len(num_cols),
        hidden_dim          = CONFIG['hidden_dim'],
        num_residual_blocks = CONFIG['num_residual'],
        dropout             = CONFIG['dropout'],
    ).to(device)

    print(f"   모델 파라미터: {model.count_parameters():,}개")
    print(f"   카테고리 피처: {len(cat_cols)}개 | 수치형 피처: {len(num_cols)}개")
    return model


# ────────────────────────────────────────────────
# 5. 손실 함수 (Class Imbalance 대응)
# ────────────────────────────────────────────────

def build_criterion(train_df: pd.DataFrame) -> nn.BCEWithLogitsLoss:
    """pos_weight로 클래스 불균형 보정."""
    y = train_df['임신 성공 여부']
    neg, pos = (y == 0).sum(), (y == 1).sum()
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32)
    print(f"   클래스 비율 → 음성: {neg:,} | 양성: {pos:,} | pos_weight: {pos_weight.item():.3f}")
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ⚠️ BCEWithLogitsLoss 사용 시 모델 마지막 Sigmoid 제거 필요
    # → model.py의 head 마지막 레이어를 nn.Sigmoid() 대신 아무것도 없게 수정하거나
    #    여기서는 BCELoss + Sigmoid 조합으로 사용 (아래 train_epoch 참고)


def build_bce_loss(train_df: pd.DataFrame) -> nn.BCELoss:
    """기본 BCELoss (모델 출력이 이미 Sigmoid 통과한 경우)."""
    return nn.BCELoss()


# ────────────────────────────────────────────────
# 6. 학습 / 검증 스텝
# ────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for cat_batch, num_batch, y_batch in loader:
        cat_batch = cat_batch.to(device)
        num_batch = num_batch.to(device)
        y_batch   = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(cat_batch, num_batch).squeeze()
        loss = criterion(pred, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device) -> tuple:
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0

    with torch.no_grad():
        for cat_batch, num_batch, y_batch in loader:
            cat_batch = cat_batch.to(device)
            num_batch = num_batch.to(device)
            y_batch   = y_batch.to(device)

            pred = model(cat_batch, num_batch).squeeze()
            loss = criterion(pred, y_batch)
            total_loss += loss.item()

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    auc = roc_auc_score(all_targets, all_preds)
    return total_loss / len(loader), auc


# ────────────────────────────────────────────────
# 7. 메인 학습 루프
# ────────────────────────────────────────────────

def train_and_evaluate():
    # 0. 시드 고정
    set_seed(CONFIG['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  장치: {device}")

    # 1. 데이터 로드
    train_df, test_df = load_data()

    # 2. 전처리
    print("\n🔧 전처리 시작...")
    train_proc, test_proc, artifacts = run_preprocessing(train_df, test_df)

    # 3. DataLoader
    print("\n📦 DataLoader 구성...")
    train_loader, val_loader = build_dataloaders(train_proc)

    # 4. 모델
    print("\n🧠 모델 구성...")
    model = build_model(train_proc, device)

    # 5. 손실함수 & 옵티마이저
    criterion = build_criterion(train_proc)  # pos_weight 포함
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'], eta_min=1e-5
    )

    # 6. 학습 루프 + Early Stopping
    print(f"\n🚀 학습 시작 (최대 {CONFIG['epochs']} epochs, patience={CONFIG['early_stop_patience']})\n")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val AUC':>9} | {'상태':>6}")
    print("─" * 55)

    best_auc     = 0.0
    patience_cnt = 0

    for epoch in range(1, CONFIG['epochs'] + 1):
        train_loss          = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        improved = val_auc > best_auc
        marker   = "✅ best" if improved else ""

        if improved:
            best_auc = val_auc
            patience_cnt = 0
            torch.save({
                'epoch':      epoch,
                'model_state_dict': model.state_dict(),
                'val_auc':    best_auc,
                'config':     CONFIG,
                'artifacts':  {k: v for k, v in artifacts.items() if k in ('num_cols',)},
            }, CONFIG['model_save_path'])
        else:
            patience_cnt += 1

        print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>10.4f} | {val_auc:>9.4f} | {marker}")

        if patience_cnt >= CONFIG['early_stop_patience']:
            print(f"\n⏹️  Early Stopping (patience {CONFIG['early_stop_patience']} 소진)")
            break

    print(f"\n🏆 최고 Val ROC-AUC: {best_auc:.4f}")
    print(f"💾 Best 모델 저장됨: '{CONFIG['model_save_path']}'")


# ────────────────────────────────────────────────

if __name__ == "__main__":
    train_and_evaluate()
