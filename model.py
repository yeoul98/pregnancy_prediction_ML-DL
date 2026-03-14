"""
model.py
────────
IVF 임신 성공 여부 예측 딥러닝 모델.

구조:
  - 카테고리형 변수 → Embedding → BatchNorm
  - 수치형 변수 → Linear projection → BatchNorm
  - 두 표현 합산 → MLP (Residual Block 포함)
  - 이진 분류 출력 (Sigmoid)
"""

import torch
import torch.nn as nn
from typing import List


# ────────────────────────────────────────────────
# 1. 구성 요소
# ────────────────────────────────────────────────

class EmbeddingBlock(nn.Module):
    """카테고리형 피처를 Embedding → BatchNorm1d → Dropout으로 처리."""

    def __init__(self, vocab_sizes: List[int], dropout: float = 0.2):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, min(50, (vocab_size + 1) // 2))
            for vocab_size in vocab_sizes
        ])
        self.embed_dim = sum(min(50, (v + 1) // 2) for v in vocab_sizes)
        self.bn  = nn.BatchNorm1d(self.embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embs = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        out  = torch.cat(embs, dim=1)
        return self.drop(self.bn(out))


class ResidualBlock(nn.Module):
    """잔차 연결(Residual Connection)을 가진 MLP 블록."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


# ────────────────────────────────────────────────
# 2. 메인 모델
# ────────────────────────────────────────────────

class IVFModel(nn.Module):
    """
    IVF 임신 성공 예측 모델.

    Parameters
    ----------
    cat_vocab_sizes : list of int
        각 카테고리 컬럼의 고유값 수 (train 기준)
    num_feature_size : int
        수치형 피처 수
    hidden_dim : int
        MLP 히든 레이어 크기 (기본 256)
    num_residual_blocks : int
        잔차 블록 수 (기본 3)
    dropout : float
        드롭아웃 비율 (기본 0.3)
    """

    def __init__(
        self,
        cat_vocab_sizes: List[int],
        num_feature_size: int,
        hidden_dim: int = 256,
        num_residual_blocks: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        # 카테고리 임베딩 블록
        self.emb_block = EmbeddingBlock(cat_vocab_sizes, dropout=dropout)
        embed_dim = self.emb_block.embed_dim

        # 수치형 입력 projection
        self.num_proj = nn.Sequential(
            nn.Linear(num_feature_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 두 표현 결합 후 hidden_dim으로 맞추는 projection
        combined_dim = embed_dim + hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # Residual MLP
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout=dropout)
              for _ in range(num_residual_blocks)]
        )

        # 출력 헤드
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, cat_x: torch.Tensor, num_x: torch.Tensor) -> torch.Tensor:
        emb_out = self.emb_block(cat_x)
        num_out = self.num_proj(num_x)
        combined = torch.cat([emb_out, num_out], dim=1)
        x = self.input_proj(combined)
        x = self.res_blocks(x)
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
