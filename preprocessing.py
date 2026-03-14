"""
preprocessing.py
────────────────
대회 규칙을 준수하는 전처리 파이프라인

⚠️  대회 유의사항 핵심 규칙:
    - Label/One-hot encoding → train + test 합쳐서 fit
    - Data scaling          → test 데이터 통계값으로 독립 처리
    - 결측치 처리            → test는 test 자체 통계값 사용
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict, List


# ────────────────────────────────────────────────
# 1. 상수 정의
# ────────────────────────────────────────────────

# 순서형 인코딩 매핑 (사전 지식 기반 → Leakage 없음)
AGE_MAP = {
    '만18-34세': 1, '만35-37세': 2, '만38-39세': 3,
    '만40-42세': 4, '만43-44세': 5, '만45-50세': 6, '알 수 없음': 0
}

COUNT_MAP = {
    '0회': 0, '1회': 1, '2회': 2, '3회': 3,
    '4회': 4, '5회': 5, '6회 이상': 6
}

DONOR_AGE_MAP = {
    '알 수 없음': 0, '만20세 이하': 1, '만21-25세': 2,
    '만26-30세': 3, '만31-35세': 4, '만36-40세': 5, '만41-45세': 6
}

# 카테고리형 컬럼 (Label Encoding 대상)
CAT_COLS = ['시술 유형', '특정 시술 유형', '난자 출처', '정자 출처',
            '배란 유도 유형', '배아 생성 주요 이유']

# 횟수 컬럼 (순서형 변환 대상)
COUNT_COLS = ['총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수',
              '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수',
              '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수']

# 제외 컬럼
EXCLUDE_COLS = ['ID', '임신 성공 여부', '시술 시기 코드']


# ────────────────────────────────────────────────
# 2. 개별 변환 함수
# ────────────────────────────────────────────────

def encode_age(df: pd.DataFrame) -> pd.DataFrame:
    """나이 컬럼을 순서형 숫자로 변환 (사전 정의 매핑 → Leakage 없음)."""
    df = df.copy()
    df['시술 당시 나이'] = df['시술 당시 나이'].map(AGE_MAP).fillna(0).astype(int)
    return df


def encode_count_columns(df: pd.DataFrame) -> pd.DataFrame:
    """'N회' / '6회 이상' 형태 컬럼을 정수로 변환."""
    df = df.copy()
    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = df[col].map(COUNT_MAP).fillna(0).astype(int)
    return df


def encode_donor_age(df: pd.DataFrame) -> pd.DataFrame:
    """기증자 나이 컬럼을 순서형 숫자로 변환."""
    df = df.copy()
    for col in ['난자 기증자 나이', '정자 기증자 나이']:
        if col in df.columns:
            df[col] = df[col].map(DONOR_AGE_MAP).fillna(0).astype(int)
    return df


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """파생 피처 생성."""
    df = df.copy()

    # 불임 원인 관련 컬럼 합계
    infertility_cols = [c for c in df.columns if '불임 원인' in c]
    if infertility_cols:
        df['infertility_count'] = df[infertility_cols].sum(axis=1)

    # 주요 불임 원인 합계
    main_cause_cols = [c for c in df.columns if '주 불임 원인' in c]
    if main_cause_cols:
        df['main_cause_count'] = df[main_cause_cols].sum(axis=1)

    return df


def label_encode_categorical(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    ✅ 대회 규칙 준수: train + test 합쳐서 fit → 각각 transform.
    알 수 없는 값도 안전하게 처리.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    le_dict = {}

    for col in CAT_COLS:
        if col not in train_df.columns:
            continue

        le = LabelEncoder()
        # train + test 전체 값으로 fit (대회 규칙)
        combined = pd.concat([
            train_df[col].fillna('Unknown').astype(str),
            test_df[col].fillna('Unknown').astype(str)
        ])
        le.fit(combined)

        train_df[col] = le.transform(train_df[col].fillna('Unknown').astype(str))
        test_df[col]  = le.transform(test_df[col].fillna('Unknown').astype(str))
        le_dict[col] = le

    return train_df, test_df, le_dict


def fill_missing_values(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    ✅ 대회 규칙 준수:
        - train → train 중앙값으로 결측치 처리
        - test  → test 자체 중앙값으로 결측치 처리 (독립)
    """
    train_df = train_df.copy()
    test_df  = test_df.copy()

    train_medians = train_df[num_cols].median()
    test_medians  = test_df[num_cols].median()

    train_df[num_cols] = train_df[num_cols].fillna(train_medians)
    test_df[num_cols]  = test_df[num_cols].fillna(test_medians)

    return train_df, test_df


def scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, StandardScaler]:
    """
    ✅ 대회 규칙 준수: train / test 각각 독립적으로 StandardScaler fit_transform.
    """
    train_df = train_df.copy()
    test_df  = test_df.copy()

    train_scaler = StandardScaler()
    test_scaler  = StandardScaler()

    train_df[num_cols] = train_scaler.fit_transform(train_df[num_cols])
    test_df[num_cols]  = test_scaler.fit_transform(test_df[num_cols])

    return train_df, test_df, train_scaler, test_scaler


# ────────────────────────────────────────────────
# 3. 피처 분리 헬퍼
# ────────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """카테고리형 / 수치형 컬럼 목록 반환."""
    cat_features = [c for c in CAT_COLS if c in df.columns]

    # 수치형: 나이(순서형 포함) + 횟수 + 기증자나이 + float 컬럼 + 파생피처
    ordinal_features = ['시술 당시 나이', '난자 기증자 나이', '정자 기증자 나이'] + COUNT_COLS
    ordinal_features = [c for c in ordinal_features if c in df.columns]

    exclude = set(EXCLUDE_COLS + CAT_COLS)
    float_features = [c for c in df.columns
                      if c not in exclude
                      and df[c].dtype in [np.float32, np.float64, np.int32, np.int64]
                      and c != '임신 성공 여부']

    num_features = list(dict.fromkeys(ordinal_features + float_features))  # 순서 유지 + 중복 제거
    return cat_features, num_features


def split_X_y(df: pd.DataFrame):
    """피처와 타겟 분리."""
    cat_cols, num_cols = get_feature_columns(df)
    X_cat = df[cat_cols].astype(int)
    X_num = df[num_cols].astype(np.float32)
    y = df['임신 성공 여부'].astype(np.float32) if '임신 성공 여부' in df.columns else None
    return X_cat, X_num, y, cat_cols, num_cols


# ────────────────────────────────────────────────
# 4. 메인 파이프라인
# ────────────────────────────────────────────────

def run_preprocessing(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    전체 전처리 파이프라인 실행.

    Returns
    -------
    train_proc, test_proc : 전처리 완료된 DataFrame
    artifacts             : 인코더/스케일러 등 부산물 딕셔너리
    """
    if verbose:
        print("🔧 [1/6] 나이 인코딩...")
    train_df = encode_age(train_df)
    test_df  = encode_age(test_df)

    if verbose:
        print("🔧 [2/6] 횟수 컬럼 변환...")
    train_df = encode_count_columns(train_df)
    test_df  = encode_count_columns(test_df)

    if verbose:
        print("🔧 [3/6] 기증자 나이 인코딩...")
    train_df = encode_donor_age(train_df)
    test_df  = encode_donor_age(test_df)

    if verbose:
        print("🔧 [4/6] 파생 피처 생성...")
    train_df = add_feature_engineering(train_df)
    test_df  = add_feature_engineering(test_df)

    if verbose:
        print("🔧 [5/6] 카테고리 Label Encoding (train+test 합산 fit)...")
    train_df, test_df, le_dict = label_encode_categorical(train_df, test_df)

    # 수치형 컬럼 확정
    _, num_cols = get_feature_columns(train_df)

    if verbose:
        print("🔧 [6/6] 결측치 처리 & 스케일링 (각 독립 적용)...")
    train_df, test_df = fill_missing_values(train_df, test_df, num_cols)
    train_df, test_df, train_scaler, test_scaler = scale_features(train_df, test_df, num_cols)

    artifacts = {
        'le_dict':      le_dict,
        'train_scaler': train_scaler,
        'test_scaler':  test_scaler,
        'num_cols':     num_cols,
    }

    if verbose:
        cat_cols, num_cols_ = get_feature_columns(train_df)
        print(f"\n✅ 전처리 완료!")
        print(f"   카테고리 피처: {len(cat_cols)}개 → {cat_cols}")
        print(f"   수치형 피처:   {len(num_cols_)}개")
        print(f"   Train: {train_df.shape} | Test: {test_df.shape}")

    return train_df, test_df, artifacts
