"""
preprocess.py — 数据清洗与编码模块
贷款审批预测项目 | loan-approval-ml

流程：
  1. 加载原始 CSV
  2. 删除无意义列（Loan_ID）
  3. 缺失值填充（分类→众数，数值→中位数）
  4. 标签编码（二值化分类变量）
  5. One-Hot 编码（多类别变量 Property_Area / Dependents）
  6. 标签列转为 0/1 整数
  7. 保存处理后的文件到 data/processed/
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


# ── 路径配置（跨平台兼容） ──────────────────────────────────────────────────
# 以本文件所在目录的父目录作为项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 兼容性：若脚本直接放在项目根目录下执行，parent.parent 可能越界
# 以实际存在的 data/ 目录向上查找确认根目录
_candidate = Path(__file__).resolve().parent.parent
if not (_candidate / "data").exists():
    _candidate = Path(__file__).resolve().parent
PROJECT_ROOT = _candidate
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_RAW  = RAW_DIR / "train.csv"
TEST_RAW   = RAW_DIR / "test.csv"
TRAIN_OUT  = PROCESSED_DIR / "train_clean.csv"
TEST_OUT   = PROCESSED_DIR / "test_clean.csv"

# 二值化映射（仅含两个取值的分类列）
BINARY_COLS = {
    "Gender":        {"Male": 1, "Female": 0},
    "Married":       {"Yes": 1, "No": 0},
    "Education":     {"Graduate": 1, "Not Graduate": 0},
    "Self_Employed": {"Yes": 1, "No": 0},
}

# 标签列
LABEL_COL = "Loan_Status"


def load_raw(path: Path) -> pd.DataFrame:
    """加载原始 CSV 文件，并做基础类型检查。"""
    df = pd.read_csv(path)
    print(f"[load]  {path.name}: {df.shape[0]} 行 × {df.shape[1]} 列")
    return df


def drop_id(df: pd.DataFrame) -> pd.DataFrame:
    """删除 Loan_ID（唯一标识符，对模型无意义）。"""
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])
        print("[drop]  已删除 Loan_ID 列")
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    缺失值填充策略：
      - 分类列 → 众数（mode）
      - 数值列 → 中位数（median）

    各列缺失情况（训练集）：
      Gender(13), Married(3), Dependents(15), Self_Employed(32),
      LoanAmount(22), Loan_Amount_Term(14), Credit_History(50)
    """
    # 分类列（object 类型）用众数填充
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    # 若标签列存在，排除在外（训练集有，测试集无）
    cat_cols = [c for c in cat_cols if c != LABEL_COL]

    for col in cat_cols:
        mode_val = df[col].mode()[0]
        missing_n = df[col].isna().sum()
        if missing_n > 0:
            df[col] = df[col].fillna(mode_val)
            print(f"[fill]  {col:<20} 众数={mode_val!r}  填充 {missing_n} 个缺失值")

    # 数值列用中位数填充
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        missing_n = df[col].isna().sum()
        if missing_n > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[fill]  {col:<20} 中位数={median_val:.2f}  填充 {missing_n} 个缺失值")

    return df


def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    对二值分类列进行标签编码（Label Encoding）。
    例：Male→1, Female→0；Yes→1, No→0
    """
    for col, mapping in BINARY_COLS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            # 若映射后仍存在 NaN（未知取值），用众数补全
            if df[col].isna().any():
                fallback = df[col].mode()[0]
                df[col] = df[col].fillna(fallback)
            print(f"[encode] {col:<20} 二值映射: {mapping}")
    return df


def encode_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """
    对多类别分类列做 One-Hot 编码。
    涉及列：
      - Property_Area（Urban / Rural / Semiurban → 3列）
      - Dependents（0 / 1 / 2 / 3+ → 4列）

    drop_first=False 保留全部哑变量，避免信息丢失（树模型不怕多重共线性）。
    prefix 与列名一致，方便后续特征工程引用。
    """
    ohe_cols = ["Property_Area", "Dependents"]
    ohe_cols_present = [c for c in ohe_cols if c in df.columns]

    if ohe_cols_present:
        df = pd.get_dummies(df, columns=ohe_cols_present, drop_first=False)
        print(f"[encode] One-Hot 编码列: {ohe_cols_present}")
        # 打印新增列名
        new_cols = [c for c in df.columns if any(c.startswith(p) for p in ohe_cols_present)]
        print(f"         新增列: {new_cols}")

    return df


def encode_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    将目标列 Loan_Status 从字符串转为整数：Y→1, N→0。
    仅在训练集（含 Loan_Status 列）时调用。
    """
    if LABEL_COL in df.columns:
        df[LABEL_COL] = df[LABEL_COL].map({"Y": 1, "N": 0})
        print(f"[encode] {LABEL_COL}: Y→1, N→0  |  分布: {df[LABEL_COL].value_counts().to_dict()}")
    return df


def ensure_bool_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """
    pandas get_dummies 有时生成 bool 列，统一转为 int（0/1），
    避免后续 sklearn/XGBoost 处理时的类型警告。
    """
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"[dtype] bool→int: {bool_cols}")
    return df


def validate_no_missing(df: pd.DataFrame, name: str) -> None:
    """断言清洗后无缺失值，否则抛出异常。"""
    remaining = df.isnull().sum()
    remaining = remaining[remaining > 0]
    if not remaining.empty:
        raise ValueError(f"[ERROR] {name} 清洗后仍存在缺失值:\n{remaining}")
    print(f"[validate] {name}: 无缺失值 ✓")


def preprocess(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    完整预处理流水线。

    Args:
        df:       原始 DataFrame
        is_train: True 表示训练集（含标签列），False 表示测试集
    Returns:
        清洗编码后的 DataFrame
    """
    df = df.copy()
    df = drop_id(df)
    df = fill_missing(df)
    df = encode_binary(df)
    df = encode_onehot(df)
    df = ensure_bool_to_int(df)
    if is_train:
        df = encode_label(df)
    return df


def align_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    One-Hot 编码后训练集与测试集的列数可能不同（某类别在测试集中未出现）。
    以训练集特征列为基准，对测试集进行对齐：
      - 缺少的列补 0
      - 多余的列删除
    标签列（Loan_Status）不参与对齐。
    """
    train_feat_cols = [c for c in train_df.columns if c != LABEL_COL]
    test_aligned = test_df.reindex(columns=train_feat_cols, fill_value=0)
    dropped = set(test_df.columns) - set(train_feat_cols)
    added   = set(train_feat_cols) - set(test_df.columns)
    if dropped:
        print(f"[align] 测试集删除多余列: {dropped}")
    if added:
        print(f"[align] 测试集补充缺失列（填0）: {added}")
    return test_aligned


def run():
    """主入口：端到端执行预处理并保存结果。"""
    # 确保输出目录存在
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  STEP 1 / 6  数据加载")
    print("=" * 55)
    train_raw = load_raw(TRAIN_RAW)
    test_raw  = load_raw(TEST_RAW)

    print("\n" + "=" * 55)
    print("  STEP 2 / 6  数据清洗（缺失值填充）")
    print("=" * 55)
    print("\n--- 训练集 ---")
    train_clean = preprocess(train_raw, is_train=True)
    print("\n--- 测试集 ---")
    test_clean  = preprocess(test_raw,  is_train=False)

    print("\n" + "=" * 55)
    print("  STEP 3 / 6  列对齐（训练/测试一致）")
    print("=" * 55)
    test_clean = align_columns(train_clean, test_clean)

    print("\n" + "=" * 55)
    print("  STEP 4 / 6  数据验证")
    print("=" * 55)
    validate_no_missing(train_clean, "训练集")
    validate_no_missing(test_clean,  "测试集")

    print("\n" + "=" * 55)
    print("  STEP 5 / 6  保存结果")
    print("=" * 55)
    train_clean.to_csv(TRAIN_OUT, index=False)
    test_clean.to_csv(TEST_OUT,   index=False)
    print(f"[save]  训练集 → {TRAIN_OUT}")
    print(f"[save]  测试集 → {TEST_OUT}")

    print("\n" + "=" * 55)
    print("  STEP 6 / 6  最终列结构")
    print("=" * 55)
    feat_cols = [c for c in train_clean.columns if c != LABEL_COL]
    print(f"特征数量: {len(feat_cols)}")
    print(f"特征列表: {feat_cols}")
    print(f"标签列:   {LABEL_COL}  (仅训练集)")
    print(f"\n训练集 shape: {train_clean.shape}")
    print(f"测试集 shape: {test_clean.shape}")

    return train_clean, test_clean


if __name__ == "__main__":
    run()