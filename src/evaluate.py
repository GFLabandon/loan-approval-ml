"""
evaluate.py — 模型评估与可视化模块
贷款审批预测项目 | loan-approval-ml

输出内容：
  ① 验证集完整性能指标表（Accuracy / Precision / Recall / F1 / ROC-AUC）
  ② 三模型 ROC 曲线对比图          → outputs/figures/roc_curves.png
  ③ 三模型混淆矩阵对比图           → outputs/figures/confusion_matrices.png
  ④ 三模型 Precision-Recall 曲线   → outputs/figures/pr_curves.png
  ⑤ 三模型指标柱状图对比           → outputs/figures/metrics_bar.png
  ⑥ 三模型学习曲线图               → outputs/figures/learning_curves.png  [新增 P2]
  ⑦ 完整性能报告 CSV              → outputs/eval_report.csv
  ⑧ 测试集预测结果                → outputs/submission.csv               [新增 P1]

运行前提：
  已运行 preprocess.py → features.py → train.py
  outputs/models/ 中存在三个 *_best.pkl 文件
"""

import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, auc,
    confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")


# ── 路径配置 ────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR   = PROJECT_ROOT / "outputs" / "figures"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"

TRAIN_FEAT = PROCESSED_DIR / "train_feat.csv"
TEST_FEAT  = PROCESSED_DIR / "test_feat.csv"
RAW_TEST   = PROJECT_ROOT / "data" / "raw" / "test.csv"
LABEL_COL  = "Loan_Status"
SEED       = 42

# 与 train.py 保持完全一致的模型顺序与颜色
MODEL_NAMES  = ["LogisticRegression", "RandomForest", "XGBoost"]
MODEL_COLORS = ["#2196F3", "#4CAF50", "#FF5722"]   # 蓝 / 绿 / 橙
MODEL_LABELS = {
    "LogisticRegression": "Logistic Regression",
    "RandomForest":       "Random Forest",
    "XGBoost":            "XGBoost",
}


# ── 字体配置 ────────────────────────────────────────────────────────────────
def _setup_font():
    preferred = ["PingFang SC", "Heiti TC", "SimHei", "WenQuanYi Micro Hei"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in preferred:
        if font in available:
            plt.rcParams["font.family"] = font
            return
    plt.rcParams["font.family"] = "DejaVu Sans"


# ── 数据加载与验证集重建 ─────────────────────────────────────────────────────
def load_val_set():
    """
    用与 train.py 完全相同的随机种子重建验证集，
    保证评估在模型未见过的数据上进行。
    验证集: 123 行（Y=85, N=38）
    """
    df = pd.read_csv(TRAIN_FEAT)
    feat_cols = [c for c in df.columns if c != LABEL_COL]
    X = df[feat_cols]
    y = df[LABEL_COL]

    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"[data]  验证集: {X_val.shape[0]} 行  "
          f"| Y={y_val.sum()}  N={(y_val == 0).sum()}")
    return X_val, y_val, feat_cols


def load_train_set():
    """加载完整训练集（用于学习曲线计算）。"""
    df = pd.read_csv(TRAIN_FEAT)
    feat_cols = [c for c in df.columns if c != LABEL_COL]
    X = df[feat_cols]
    y = df[LABEL_COL]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    return X_train, y_train


# ── 加载已训练模型 ───────────────────────────────────────────────────────────
def load_models() -> dict:
    """加载三个最优模型，返回 {名称: 模型} 字典。"""
    models = {}
    for name in MODEL_NAMES:
        path = MODELS_DIR / f"{name}_best.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"找不到 {path}，请先运行 train.py"
            )
        models[name] = joblib.load(path)
        print(f"[load]  {name:<22} ← {path.name}")
    return models


# ── 指标计算 ─────────────────────────────────────────────────────────────────
def compute_metrics(models: dict, X_val, y_val) -> pd.DataFrame:
    """
    计算每个模型在验证集上的六项指标：
      Accuracy / Precision / Recall / F1 / ROC-AUC / PR-AUC
    """
    rows = []
    for name, model in models.items():
        y_pred  = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        precision_arr, recall_arr, _ = precision_recall_curve(y_val, y_proba)
        pr_auc = auc(recall_arr, precision_arr)

        rows.append({
            "模型":        MODEL_LABELS[name],
            "Accuracy":   round(accuracy_score(y_val, y_pred), 4),
            "Precision":  round(precision_score(y_val, y_pred, zero_division=0), 4),
            "Recall":     round(recall_score(y_val, y_pred, zero_division=0), 4),
            "F1":         round(f1_score(y_val, y_pred, zero_division=0), 4),
            "ROC-AUC":    round(roc_auc_score(y_val, y_proba), 4),
            "PR-AUC":     round(pr_auc, 4),
        })

    df = pd.DataFrame(rows).set_index("模型")
    return df


# ── 图1：ROC 曲线对比 ────────────────────────────────────────────────────────
def plot_roc_curves(models: dict, X_val, y_val, save_path: Path):
    _setup_font()
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, color in zip(MODEL_NAMES, MODEL_COLORS):
        y_proba = models[name].predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        score = roc_auc_score(y_val, y_proba)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{MODEL_LABELS[name]}  (AUC = {score:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Validation Set", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] ROC 曲线图 → {save_path.name}")


# ── 图2：混淆矩阵对比 ────────────────────────────────────────────────────────
def plot_confusion_matrices(models: dict, X_val, y_val, save_path: Path):
    _setup_font()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Confusion Matrices — Validation Set",
                 fontsize=14, fontweight="bold", y=1.01)
    class_labels = ["Rejected (N)", "Approved (Y)"]

    for ax, (name, color) in zip(axes, zip(MODEL_NAMES, MODEL_COLORS)):
        y_pred = models[name].predict(X_val)
        cm     = confusion_matrix(y_val, y_pred)
        cm_pct = cm.astype(float) / cm.sum() * 100

        ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0)
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i,
                        f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                        ha="center", va="center", fontsize=12,
                        color="white" if cm[i, j] > thresh else "black")

        ax.set_xticks([0, 1]); ax.set_xticklabels(class_labels, fontsize=9)
        ax.set_yticks([0, 1]); ax.set_yticklabels(class_labels, fontsize=9, rotation=90, va="center")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_title(MODEL_LABELS[name], fontsize=11, fontweight="bold", color=color)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] 混淆矩阵图 → {save_path.name}")


# ── 图3：Precision-Recall 曲线 ───────────────────────────────────────────────
def plot_pr_curves(models: dict, X_val, y_val, save_path: Path):
    _setup_font()
    baseline_pr = y_val.mean()
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, color in zip(MODEL_NAMES, MODEL_COLORS):
        y_proba = models[name].predict_proba(X_val)[:, 1]
        prec, rec, _ = precision_recall_curve(y_val, y_proba)
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, color=color, lw=2,
                label=f"{MODEL_LABELS[name]}  (PR-AUC = {pr_auc:.4f})")

    ax.axhline(y=baseline_pr, color="k", linestyle="--", lw=1,
               alpha=0.5, label=f"Random Baseline ({baseline_pr:.2f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Validation Set",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] PR 曲线图 → {save_path.name}")


# ── 图4：指标柱状图对比 ──────────────────────────────────────────────────────
def plot_metrics_bar(metrics_df: pd.DataFrame, save_path: Path):
    _setup_font()
    plot_metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    data = metrics_df[plot_metrics]

    n_models  = len(data)
    n_metrics = len(plot_metrics)
    x = np.arange(n_models)
    width = 0.14
    offsets = np.linspace(-(n_metrics - 1) / 2 * width,
                           (n_metrics - 1) / 2 * width, n_metrics)
    metric_colors = ["#1565C0", "#2E7D32", "#C62828", "#6A1B9A", "#E65100"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (metric, mc) in enumerate(zip(plot_metrics, metric_colors)):
        bars = ax.bar(x + offsets[i], data[metric].values, width,
                      label=metric, color=mc, alpha=0.85, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(data.index, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison — Validation Set",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] 指标柱状图 → {save_path.name}")


# ── 图5：学习曲线（P2 新增）────────────────────────────────────────────────
def plot_learning_curves(models: dict, X_train, y_train, save_path: Path):
    """
    绘制三模型的学习曲线（训练集大小 vs. 5折 CV ROC-AUC）。

    分析价值：
    - Train/Val 曲线差距大 → 过拟合（高方差）
    - 两条曲线都低且收敛 → 欠拟合（高偏差）
    - 曲线趋于平坦 → 增加样本量收益递减

    注：XGBoost Pipeline 不支持 learning_curve 的 clone，
    此处直接使用 best_estimator_ 对象（已为 sklearn 兼容估计器）。
    """
    _setup_font()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Learning Curves — Training Size vs. CV ROC-AUC",
                 fontsize=13, fontweight="bold", y=1.02)

    train_sizes_pct = np.linspace(0.1, 1.0, 8)   # 10%~100% 共8个点

    for ax, (name, color) in zip(axes, zip(MODEL_NAMES, MODEL_COLORS)):
        model = models[name]
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            cv=5,
            scoring="roc_auc",
            train_sizes=train_sizes_pct,
            n_jobs=-1,
            shuffle=True,
            random_state=SEED,
        )

        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)

        # 训练曲线（实线）+ 置信带
        ax.plot(train_sizes, train_mean, color=color, lw=2,
                label="Train AUC")
        ax.fill_between(train_sizes,
                        train_mean - train_std,
                        train_mean + train_std,
                        alpha=0.12, color=color)

        # 验证曲线（虚线）+ 置信带
        ax.plot(train_sizes, val_mean, color=color, lw=2,
                linestyle="--", label="CV Val AUC")
        ax.fill_between(train_sizes,
                        val_mean - val_std,
                        val_mean + val_std,
                        alpha=0.18, color=color)

        # 标注最终验证 AUC 值
        final_val = val_mean[-1]
        ax.annotate(f"Val={final_val:.3f}",
                    xy=(train_sizes[-1], final_val),
                    xytext=(-30, 8), textcoords="offset points",
                    fontsize=8.5, color=color,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8))

        ax.set_title(MODEL_LABELS[name], fontsize=11,
                     fontweight="bold", color=color)
        ax.set_xlabel("Training Samples", fontsize=10)
        ax.set_ylabel("ROC-AUC", fontsize=10)
        ax.set_ylim([0.55, 1.02])
        ax.legend(fontsize=9, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] 学习曲线图 → {save_path.name}")


# ── P1 新增：测试集预测 ──────────────────────────────────────────────────────
def predict_test_set(models: dict, feat_cols: list) -> None:
    """
    使用验证集 ROC-AUC 最优的模型（逻辑回归）对原始测试集进行预测，
    生成 Kaggle 格式的 submission.csv。

    注意事项：
    - 特征列顺序必须与训练时一致，由 feat_cols 参数保证
    - 仅对训练集中存在的特征列进行预测（reindex 对齐）
    - 预测标签从 0/1 还原为 Y/N，保持与原始数据格式一致
    """
    if not TEST_FEAT.exists():
        print(f"[skip] {TEST_FEAT} 不存在，跳过测试集预测")
        return
    if not RAW_TEST.exists():
        print(f"[skip] {RAW_TEST} 不存在，跳过（无法获取 Loan_ID）")
        return

    test_df  = pd.read_csv(TEST_FEAT)
    raw_test = pd.read_csv(RAW_TEST)

    # 对齐特征列：测试集可能因 One-Hot 缺少某列，补 0
    X_test = test_df.reindex(columns=feat_cols, fill_value=0)

    # 选择验证集 ROC-AUC 最优模型（逻辑回归）进行最终预测
    best_model = models["LogisticRegression"]
    predictions = best_model.predict(X_test)
    proba_Y     = best_model.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({
        "Loan_ID":     raw_test["Loan_ID"],
        "Loan_Status": ["Y" if p == 1 else "N" for p in predictions],
        "Prob_Y":      proba_Y.round(4),   # 附带概率值，便于说明书分析
    })

    output_path = OUTPUTS_DIR / "submission.csv"
    submission.to_csv(output_path, index=False)

    n_approve = (predictions == 1).sum()
    n_reject  = (predictions == 0).sum()
    print(f"[save]  测试集预测结果 → {output_path.relative_to(PROJECT_ROOT)}")
    print(f"        367 条：Y(批准)={n_approve}  N(拒绝)={n_reject}  "
          f"批准率={n_approve/len(predictions)*100:.1f}%")


# ── 主流程 ───────────────────────────────────────────────────────────────────
def run():
    """端到端执行评估流程，生成所有图表与报告。"""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── STEP 1：数据与模型加载 ─────────────────────────────────────────────
    print("=" * 55)
    print("  STEP 1 / 5  数据与模型加载")
    print("=" * 55)
    X_val, y_val, feat_cols = load_val_set()
    X_train, y_train        = load_train_set()
    models = load_models()

    # ── STEP 2：指标计算 ───────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 2 / 5  验证集指标计算")
    print("=" * 55)
    metrics_df = compute_metrics(models, X_val, y_val)

    print("\n▼ 验证集性能指标汇总（阈值 = 0.5）：")
    print(metrics_df.to_string())

    report_path = OUTPUTS_DIR / "eval_report.csv"
    metrics_df.to_csv(report_path)
    print(f"\n[save]  指标表 → {report_path.relative_to(PROJECT_ROOT)}")

    print("\n▼ 分类详细报告（Classification Report）：")
    for name, model in models.items():
        y_pred = model.predict(X_val)
        print(f"\n── {MODEL_LABELS[name]} ──")
        print(classification_report(
            y_val, y_pred,
            target_names=["N (Rejected)", "Y (Approved)"],
            digits=4,
        ))

    # ── STEP 3：可视化（含学习曲线）────────────────────────────────────────
    print("=" * 55)
    print("  STEP 3 / 5  生成可视化图表")
    print("=" * 55)
    plot_roc_curves(models,         X_val,   y_val,   FIGURES_DIR / "roc_curves.png")
    plot_confusion_matrices(models, X_val,   y_val,   FIGURES_DIR / "confusion_matrices.png")
    plot_pr_curves(models,          X_val,   y_val,   FIGURES_DIR / "pr_curves.png")
    plot_metrics_bar(metrics_df,                      FIGURES_DIR / "metrics_bar.png")
    plot_learning_curves(models,    X_train, y_train, FIGURES_DIR / "learning_curves.png")

    # ── STEP 4：测试集预测（P1）────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 4 / 5  测试集预测（submission.csv）")
    print("=" * 55)
    predict_test_set(models, feat_cols)

    # ── STEP 5：最终结论 ───────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 5 / 5  最终结论")
    print("=" * 55)
    best_model = metrics_df["ROC-AUC"].idxmax()
    best_auc   = metrics_df.loc[best_model, "ROC-AUC"]
    best_f1    = metrics_df.loc[best_model, "F1"]
    print(f"  最优模型（ROC-AUC）: {best_model}")
    print(f"  ROC-AUC = {best_auc:.4f}   F1 = {best_f1:.4f}")
    print(f"\n  所有图表已保存至: outputs/figures/")
    print(f"  性能报告已保存至: outputs/eval_report.csv")
    print(f"  测试集预测已保存: outputs/submission.csv")

    return metrics_df


if __name__ == "__main__":
    run()