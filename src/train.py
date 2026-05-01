"""
train.py — 模型训练与超参数调优模块
贷款审批预测项目 | loan-approval-ml

流程：
  1. 加载 features.py 输出的特征数据（train_feat.csv）
  2. 分层拆分：80% 训练 / 20% 验证（stratify 保持标签比例）
  3. 三种模型基线评估（5折 CV，默认超参）
  4. GridSearchCV 调优（5折，scoring=roc_auc）
  5. [新增 P1] SMOTE 过采样对比实验（独立运行，不替换主流程）
  6. 保存最优模型到 outputs/models/*.pkl
  7. 输出交叉验证对比汇总表

模型：
  - 逻辑回归 (LogisticRegression)   —— 线性基线，Pipeline 含 StandardScaler
  - 随机森林 (RandomForest)          —— 集成树，Bagging
  - XGBoost                          —— 梯度提升树，Boosting

类别不均衡处理（Y:422 / N:192 ≈ 2.2:1）：
  主流程：LR / RF：class_weight='balanced'；XGBoost：scale_pos_weight ≈ 0.45
  对比实验：SMOTE 过采样（仅对训练集，验证集保持原始分布）
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
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)


# ── 路径配置 ────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "outputs" / "models"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"

TRAIN_FEAT = PROCESSED_DIR / "train_feat.csv"
LABEL_COL  = "Loan_Status"

SEED             = 42
CV_FOLDS         = 5
SCALE_POS_WEIGHT = round(192 / 422, 4)   # ≈ 0.4549（neg/pos）


# ── 数据加载与拆分 ───────────────────────────────────────────────────────────
def load_and_split(test_size: float = 0.2):
    """
    加载特征数据，按 80/20 分层拆分为训练集与验证集。
    stratify=y 保证拆分前后正负样本比例一致。

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
        "clf__C": [0.01, 0.1, 1.0, 10.0],
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
    """使用默认超参数进行 5 折交叉验证，记录 ROC-AUC 均值与标准差。"""
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
    执行 GridSearchCV，scoring=roc_auc，refit=True。
    Returns: 已拟合的 GridSearchCV 对象。
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
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}_best.pkl"
    joblib.dump(gs.best_estimator_, path)
    print(f"       已保存 → {path.relative_to(PROJECT_ROOT)}")
    return path


# ── P1 新增：SMOTE 对比实验 ──────────────────────────────────────────────────
def run_smote_comparison(X_train, X_val, y_train, y_val, cv) -> pd.DataFrame:
    """
    使用 SMOTE（Synthetic Minority Over-sampling Technique）对少数类（N）进行
    合成过采样，与 class_weight 方案在验证集 ROC-AUC 上进行对比。

    SMOTE 原理：
    - 对每个少数类样本，在其 k 近邻中随机选取一个邻居
    - 在两者之间的连线上随机插值生成新样本
    - 不简单复制，而是合成新的特征向量，增加特征空间多样性

    关键设计决策：
    - 仅对"训练集"过采样，验证集保持原始不均衡分布
    - 在验证集上评估，保证评估环境与真实部署一致（无数据泄露）
    - 三种模型均去除 class_weight / scale_pos_weight（由 SMOTE 替代均衡）

    Returns:
        对比汇总 DataFrame（两种不均衡处理策略的验证集 ROC-AUC 对比）
    """
    try:
        from imblearn.over_sampling import SMOTE
    except Exception as e:
        print(f"[SMOTE] 跳过：{type(e).__name__}: {e}")
        return pd.DataFrame()

    print("\n" + "=" * 55)
    print("  SMOTE 对比实验（P1 加分项）")
    print("=" * 55)

    # ── SMOTE 过采样（仅作用于训练集）──────────────────────────────────
    smote = SMOTE(random_state=SEED, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    print(f"[SMOTE] 过采样前: Y={y_train.sum()} / N={(y_train==0).sum()} "
          f"（比例 {y_train.sum()/(y_train==0).sum():.2f}:1）")
    print(f"[SMOTE] 过采样后: Y={y_train_sm.sum()} / N={(y_train_sm==0).sum()} "
          f"（比例 {y_train_sm.sum()/(y_train_sm==0).sum():.2f}:1）")
    print(f"[SMOTE] 训练集样本数: {len(y_train)} → {len(y_train_sm)}")

    # ── 在 SMOTE 数据上训练简化版三模型（无 class_weight）─────────────
    smote_models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000,
                                       random_state=SEED, solver="lbfgs")),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=200, random_state=SEED, n_jobs=-1,
            eval_metric="logloss", verbosity=0),
    }

    # class_weight 方案基准（主流程中调优后的最优参数，此处用默认简化对比）
    cw_models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000,
                                       class_weight="balanced",
                                       random_state=SEED, solver="lbfgs")),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=200, scale_pos_weight=SCALE_POS_WEIGHT,
            random_state=SEED, n_jobs=-1,
            eval_metric="logloss", verbosity=0),
    }

    rows = []
    print("\n[对比] 验证集 ROC-AUC（相同超参数，仅不均衡处理策略不同）：")
    print(f"{'模型':<22}  {'class_weight':>12}  {'SMOTE':>10}  {'差值':>8}")
    print("-" * 58)

    for name in ["LogisticRegression", "RandomForest", "XGBoost"]:
        # class_weight 方案
        cw_models[name].fit(X_train, y_train)
        cw_auc = roc_auc_score(y_val, cw_models[name].predict_proba(X_val)[:, 1])

        # SMOTE 方案
        smote_models[name].fit(X_train_sm, y_train_sm)
        sm_auc = roc_auc_score(y_val, smote_models[name].predict_proba(X_val)[:, 1])

        diff = sm_auc - cw_auc
        diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
        print(f"  {name:<22}  {cw_auc:.4f}        {sm_auc:.4f}    {diff_str}")

        rows.append({
            "模型":          name,
            "class_weight AUC": round(cw_auc, 4),
            "SMOTE AUC":        round(sm_auc, 4),
            "差值（SMOTE-CW）":  round(diff, 4),
        })

    print("-" * 58)
    smote_df = pd.DataFrame(rows)

    # 保存对比结果
    smote_path = OUTPUTS_DIR / "smote_comparison.csv"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    smote_df.to_csv(smote_path, index=False)
    print(f"\n[save] SMOTE 对比结果 → {smote_path.relative_to(PROJECT_ROOT)}")

    # 结论输出
    winner_col = "SMOTE AUC" if smote_df["SMOTE AUC"].mean() > smote_df["class_weight AUC"].mean() else "class_weight AUC"
    print(f"\n[结论] 平均 AUC 更优策略：{'SMOTE' if winner_col == 'SMOTE AUC' else 'class_weight'}")
    print("       在小样本（~490行）场景下，两种方式差异通常不显著；")
    print("       SMOTE 在特征空间均匀分布时更有优势，class_weight 在样本不足时更稳定。")

    return smote_df


# ── 结果汇总 ─────────────────────────────────────────────────────────────────
def summarize(baseline_df: pd.DataFrame,
              tuned_results: list) -> pd.DataFrame:
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
    """端到端执行模型训练与调优。"""

    # ── STEP 1：数据准备 ──────────────────────────────────────────────────
    print("=" * 55)
    print("  STEP 1 / 5  数据加载与拆分（80/20 分层）")
    print("=" * 55)
    X_train, X_val, y_train, y_val, feat_cols = load_and_split()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    # ── STEP 2：基线评估 ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 2 / 5  基线评估（默认超参数）")
    print("=" * 55)
    configs = build_model_configs()
    baseline_df = baseline_cv(configs, X_train, y_train, cv)

    # ── STEP 3：GridSearchCV 调优 ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 3 / 5  GridSearchCV 超参调优（约5~10分钟）")
    print("=" * 55)
    tuned_results = []
    for name, estimator, param_grid in configs:
        gs = tune_model(name, estimator, param_grid, X_train, y_train, cv)
        save_model(gs, name)
        tuned_results.append((name, gs))

    # ── STEP 4：汇总 ─────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 4 / 5  结果汇总")
    print("=" * 55)
    summary = summarize(baseline_df, tuned_results)

    summary_path = OUTPUTS_DIR / "cv_summary.csv"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"\n[save] 汇总表 → {summary_path.relative_to(PROJECT_ROOT)}")

    # ── STEP 5：SMOTE 对比实验（P1 加分项）──────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 5 / 5  SMOTE 不均衡处理对比实验")
    print("=" * 55)
    run_smote_comparison(X_train, X_val, y_train, y_val, cv)

    print("\n训练完成 ✓  下一步请运行 src/evaluate.py")

    return tuned_results, X_val, y_val, feat_cols


if __name__ == "__main__":
    run()