"""
train.py — 模型训练与超参数调优模块
贷款审批预测项目 | loan-approval-ml

流程：
  1. 加载 features.py 输出的特征数据（train_feat.csv）
  2. 分层拆分：80% 训练 / 20% 验证（stratify 保持标签比例）
  3. 三种模型基线评估（5折 CV，默认超参）
  4. GridSearchCV 调优（5折，scoring=roc_auc）
  5. 保存最优模型到 outputs/models/*.pkl
  6. 输出交叉验证对比汇总表

模型：
  - 逻辑回归 (LogisticRegression)   —— 线性基线，Pipeline 含 StandardScaler
  - 随机森林 (RandomForest)          —— 集成树，Bagging
  - XGBoost                          —— 梯度提升树，Boosting

类别不均衡处理（Y:422 / N:192 ≈ 2.2:1）：
  - LR / RF：class_weight='balanced'（sklearn 自动按类别数量加权）
  - XGBoost：scale_pos_weight = neg/pos ≈ 0.45
"""

import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, GridSearchCV,
    StratifiedKFold, cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)


# ── 路径配置 ────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "outputs" / "models"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"

TRAIN_FEAT = PROCESSED_DIR / "train_feat.csv"
LABEL_COL  = "Loan_Status"

# 全项目统一随机种子
SEED = 42
# 交叉验证折数
CV_FOLDS = 5
# 类别权重（neg/pos = 192/422）
SCALE_POS_WEIGHT = round(192 / 422, 4)   # ≈ 0.4549


# ── 数据加载与拆分 ───────────────────────────────────────────────────────────
def load_and_split(test_size: float = 0.2):
    """
    加载特征数据，按 80/20 分层拆分为训练集与验证集。

    stratify=y 保证拆分前后正负样本比例一致，
    避免小样本验证集中标签分布严重偏斜。

    Returns:
        X_train, X_val, y_train, y_val, feat_cols
    """
    df = pd.read_csv(TRAIN_FEAT)
    feat_cols = [c for c in df.columns if c != LABEL_COL]
    X = df[feat_cols]
    y = df[LABEL_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )
    print(f"[split] 全量数据: {df.shape[0]} 行，特征数: {len(feat_cols)}")
    print(f"[split] 训练集: {X_train.shape[0]} 行  |  验证集: {X_val.shape[0]} 行")
    print(f"[split] 训练集标签分布 → Y={y_train.sum()}  N={(y_train==0).sum()}")
    print(f"[split] 验证集标签分布 → Y={y_val.sum()}  N={(y_val==0).sum()}")
    return X_train, X_val, y_train, y_val, feat_cols


# ── 模型与调参网格定义 ───────────────────────────────────────────────────────
def build_model_configs():
    """
    返回三种模型的配置列表，每项为 (名称, 估计器, 参数网格) 三元组。

    调参网格设计原则：
      - 覆盖正则强度（LR:C）、树深度、迭代次数等核心超参
      - 组合数控制在合理范围（LR:4 / RF:72 / XGB:108），
        配合 5折 CV，本地 Mac 约 3~8 分钟可完成
    """
    # ── 逻辑回归 ─────────────────────────────────────────────────────────
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=SEED,
            solver="lbfgs",
        )),
    ])
    lr_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],   # 正则化强度，C 越小正则越强
    }

    # ── 随机森林 ─────────────────────────────────────────────────────────
    rf_model = RandomForestClassifier(
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    rf_grid = {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [4, 6, 8, None],
        "min_samples_split": [2, 5, 10],
        "max_features":      ["sqrt", "log2"],
    }

    # ── XGBoost ───────────────────────────────────────────────────────────
    xgb_model = XGBClassifier(
        scale_pos_weight=SCALE_POS_WEIGHT,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_grid = {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [3, 5, 7],
        "learning_rate":    [0.01, 0.05, 0.1],
        "subsample":        [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    return [
        ("LogisticRegression", lr_pipeline,  lr_grid),
        ("RandomForest",       rf_model,      rf_grid),
        ("XGBoost",            xgb_model,     xgb_grid),
    ]


# ── 基线评估 ─────────────────────────────────────────────────────────────────
def baseline_cv(configs, X_train, y_train, cv) -> pd.DataFrame:
    """
    使用默认超参数进行 5 折交叉验证，记录 ROC-AUC 均值与标准差。
    作为调优前的参考基线，体现超参数调优的实际提升幅度。
    """
    print("\n【基线】默认超参数 5折 ROC-AUC：")
    print("-" * 48)
    rows = []
    for name, estimator, _ in configs:
        scores = cross_val_score(
            estimator, X_train, y_train,
            cv=cv, scoring="roc_auc", n_jobs=-1,
        )
        mean_, std_ = scores.mean(), scores.std()
        rows.append({"模型": name, "基线 CV AUC": round(mean_, 4),
                     "基线 AUC Std": round(std_, 4)})
        print(f"  {name:<22}  AUC = {mean_:.4f} ± {std_:.4f}")
    print("-" * 48)
    return pd.DataFrame(rows)


# ── GridSearchCV 调优 ────────────────────────────────────────────────────────
def tune_model(name: str, estimator, param_grid: dict,
               X_train, y_train, cv) -> GridSearchCV:
    """
    对单个模型执行 GridSearchCV：
      scoring = roc_auc  —— 对不均衡数据比 accuracy 更可靠
      refit   = True     —— 用最优参数在全训练集重新拟合，得到最终模型
      n_jobs  = -1       —— 并行利用全部 CPU 核心

    Returns:
        已拟合的 GridSearchCV 对象（.best_estimator_ 为最终可用模型）
    """
    n_combinations = 1
    for v in param_grid.values():
        n_combinations *= len(v)
    print(f"\n[tune] {name}")
    print(f"       参数组合数: {n_combinations}  ×  {CV_FOLDS}折 = {n_combinations * CV_FOLDS} 次训练")

    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        refit=True,
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )
    gs.fit(X_train, y_train)

    print(f"       最优 CV AUC : {gs.best_score_:.4f}")
    print(f"       最优参数    : {gs.best_params_}")
    return gs


# ── 模型保存 ─────────────────────────────────────────────────────────────────
def save_model(gs: GridSearchCV, name: str) -> Path:
    """将最优模型（best_estimator_）序列化为 .pkl 文件。"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}_best.pkl"
    joblib.dump(gs.best_estimator_, path)
    print(f"       已保存 → {path.relative_to(PROJECT_ROOT)}")
    return path


# ── 结果汇总 ─────────────────────────────────────────────────────────────────
def summarize(baseline_df: pd.DataFrame,
              tuned_results: list) -> pd.DataFrame:
    """
    合并基线与调优结果，生成对比汇总表并保存为 CSV。

    tuned_results: [(name, GridSearchCV), ...]
    """
    tuned_rows = []
    for name, gs in tuned_results:
        tuned_rows.append({
            "模型":        name,
            "调优 CV AUC": round(gs.best_score_, 4),
        })
    tuned_df = pd.DataFrame(tuned_rows)

    summary = baseline_df.merge(tuned_df, on="模型")
    summary["AUC 提升"] = (
        summary["调优 CV AUC"] - summary["基线 CV AUC"]
    ).round(4)

    print("\n" + "=" * 60)
    print("  交叉验证 ROC-AUC 对比汇总")
    print("=" * 60)
    print(summary[["模型", "基线 CV AUC", "调优 CV AUC", "AUC 提升"]].to_string(index=False))
    print("=" * 60)
    return summary


# ── 主流程 ───────────────────────────────────────────────────────────────────
def run():
    """端到端执行模型训练与调优，返回供 evaluate.py 使用的对象。"""

    # ── STEP 1：数据准备 ──────────────────────────────────────────────────
    print("=" * 55)
    print("  STEP 1 / 4  数据加载与拆分（80/20 分层）")
    print("=" * 55)
    X_train, X_val, y_train, y_val, feat_cols = load_and_split()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    # ── STEP 2：基线评估 ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 2 / 4  基线评估（默认超参数）")
    print("=" * 55)
    configs = build_model_configs()
    baseline_df = baseline_cv(configs, X_train, y_train, cv)

    # ── STEP 3：GridSearchCV 调优 ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 3 / 4  GridSearchCV 超参调优")
    print("=" * 55)
    tuned_results = []
    for name, estimator, param_grid in configs:
        gs = tune_model(name, estimator, param_grid, X_train, y_train, cv)
        save_model(gs, name)
        tuned_results.append((name, gs))

    # ── STEP 4：汇总 ─────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 4 / 4  结果汇总")
    print("=" * 55)
    summary = summarize(baseline_df, tuned_results)

    # 保存汇总 CSV
    summary_path = OUTPUTS_DIR / "cv_summary.csv"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"\n[save] 汇总表 → {summary_path.relative_to(PROJECT_ROOT)}")
    print("\n训练完成 ✓  下一步请运行 src/evaluate.py")

    return tuned_results, X_val, y_val, feat_cols


if __name__ == "__main__":
    run()