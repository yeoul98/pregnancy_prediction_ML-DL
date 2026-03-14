"""
dataset.py
──────────
PyTorch Dataset 클래스 정의.
카테고리형(Long 텐서) + 수치형(Float 텐서) + 레이블을 함께 관리.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional
import pandas as pd


class IVFDataset(Dataset):
    """
    IVF(체외수정) 시술 데이터셋.

    Parameters
    ----------
    cat_data : pd.DataFrame
        Label-encoded 카테고리형 피처
    num_data : pd.DataFrame
        스케일링된 수치형 피처
    labels : pd.Series, optional
        타겟 레이블 (추론 시 None)
    """

    def __init__(
        self,
        cat_data: pd.DataFrame,
        num_data: pd.DataFrame,
        labels: Optional[pd.Series] = None
    ):
        self.cat_data = torch.tensor(
            cat_data.values.astype(np.int64), dtype=torch.long
        )
        self.num_data = torch.tensor(
            num_data.values.astype(np.float32), dtype=torch.float32
        )
        self.labels = (
            torch.tensor(labels.values.astype(np.float32), dtype=torch.float32)
            if labels is not None else None
        )

    def __len__(self) -> int:
        return len(self.num_data)

    def __getitem__(self, idx: int):
        cat = self.cat_data[idx]
        num = self.num_data[idx]
        if self.labels is not None:
            return cat, num, self.labels[idx]
        return cat, num

    @property
    def num_cat_features(self) -> int:
        return self.cat_data.shape[1]

    @property
    def num_num_features(self) -> int:
        return self.num_data.shape[1]
