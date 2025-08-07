import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Union

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def stratified_sample(
    df: pd.DataFrame,
    stratify_col: str,
    sample_size: int,
    n_bins: int,
    random_state: Union[int, None] = 42
) -> pd.Index:
    if sample_size > len(df):
        raise ValueError("sample_size가 데이터 개수보다 클 수 없습니다.")
    
    # stratify_col의 값이 0과 1 사이에 있는지 확인
    if not (df[stratify_col].min() >= 0 and df[stratify_col].max() <= 1):
        raise ValueError(f"'{stratify_col}' 컬럼의 값은 0과 1 사이에 있어야 합니다.")

    df_copy = df.copy()
    binned_col_name = f"{stratify_col}_bin"

    bin_edges = np.linspace(0, 1, n_bins + 1)

    # 생성한 구간 배열을 기준으로 binning 수행
    df_copy[binned_col_name] = pd.cut(
        df_copy[stratify_col],
        bins=bin_edges,
        labels=False,
        include_lowest=True # 최솟값(0)을 포함
    )
    
    # train_test_split을 사용하여 계층적 샘플링 수행
    sampled_df, _ = train_test_split(
        df_copy,
        train_size=sample_size,
        stratify=df_copy[binned_col_name],
        random_state=random_state
    )

    return sampled_df


# 예시 사용
if __name__ == "__main__":
    df = pd.read_csv(os.path.join(ROOT_DIR, "data", "train_eda.csv"))

    df_bin = stratified_sample(df, 'coverage_ratio', 500, 4)
    df_bin.to_csv(os.path.join(ROOT_DIR, "data", "train_eda_sampled.csv"), index=False)

