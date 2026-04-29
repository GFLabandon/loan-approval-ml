"""
evaluate.py — 模型评估与可视化模块
贷款审批预测项目 | loan-approval-ml

输出内容：
  ① 验证集完整性能指标表（Accuracy / Precision / Recall / F1 / ROC-AUC）
  ② 三模型 ROC 曲线对比图          → outputs/figures/roc_curves.png
  ③ 三模型混淆矩阵对比图           → outputs/figures/confusion_matrices.png
  ④ 三模型 Precision-Recall 曲线   → outputs/figures/pr_curves.png
  ⑤ 三模型指标柱状图对比           → outputs/figures/metrics_bar.png
  ⑥ 完整性能报告 CSV              → outputs/eval_report.csv

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
from sklearn.model_selection import train_test_split
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

    分类阈值统一使用 0.5（默认），与 sklearn predict() 一致。
    所有指标精确到小数点后 4 位。
    """
    rows = []
    for name, model in models.items():
        y_pred  = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        # PR-AUC（Precision-Recall 曲线下面积）
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
    """
    将三个模型的 ROC 曲线绘制在同一坐标系中，
    图例中标注各自的 AUC 值，便于直观比较。
    """
    _setup_font()
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, color in zip(MODEL_NAMES, MODEL_COLORS):
        y_proba = models[name].predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        score = roc_auc_score(y_val, y_proba)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{MODEL_LABELS[name]}  (AUC = {score:.4f})")

    # 随机猜测基线
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
    """
    将三个模型的混淆矩阵并排绘制（1行×3列），
    格子内同时显示计数与百分比，便于报告引用。
    """
    _setup_font()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Confusion Matrices — Validation Set",
                 fontsize=14, fontweight="bold", y=1.01)

    class_labels = ["Rejected (N)", "Approved (Y)"]

    for ax, (name, color) in zip(axes, zip(MODEL_NAMES, MODEL_COLORS)):
        y_pred = models[name].predict(X_val)
        cm     = confusion_matrix(y_val, y_pred)
        cm_pct = cm.astype(float) / cm.sum() * 100   # 百分比

        im = ax.imshow(cm, interpolation="nearest",
                       cmap=plt.cm.Blues, vmin=0)

        # 格子内标注
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
    """
    PR 曲线对不均衡数据集比 ROC 更具辨别力，
    图例中标注各自的 PR-AUC（AP）值。
    """
    _setup_font()
    # 基线：随机猜测的 PR-AUC = 正类比例
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
    """
    将 Accuracy / Precision / Recall / F1 / ROC-AUC 五项指标
    用分组柱状图并排展示，每组对应一个模型。
    """
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
        # 在每根柱顶标注数值
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


# ── 主流程 ───────────────────────────────────────────────────────────────────
def run():
    """端到端执行评估流程，生成所有图表与报告。"""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. 数据与模型加载 ──────────────────────────────────────────────────
    print("=" * 55)
    print("  STEP 1 / 4  数据与模型加载")
    print("=" * 55)
    X_val, y_val, feat_cols = load_val_set()
    models = load_models()

    # ── 2. 指标计算 ────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 2 / 4  验证集指标计算")
    print("=" * 55)
    metrics_df = compute_metrics(models, X_val, y_val)

    print("\n▼ 验证集性能指标汇总（阈值 = 0.5）：")
    print(metrics_df.to_string())

    # 保存指标 CSV
    report_path = OUTPUTS_DIR / "eval_report.csv"
    metrics_df.to_csv(report_path)
    print(f"\n[save]  指标表 → {report_path.relative_to(PROJECT_ROOT)}")

    # 打印各模型详细分类报告
    print("\n▼ 分类详细报告（Classification Report）：")
    for name, model in models.items():
        y_pred = model.predict(X_val)
        print(f"\n── {MODEL_LABELS[name]} ──")
        print(classification_report(
            y_val, y_pred,
            target_names=["N (Rejected)", "Y (Approved)"],
            digits=4,
        ))

    # ── 3. 可视化 ──────────────────────────────────────────────────────────
    print("=" * 55)
    print("  STEP 3 / 4  生成可视化图表")
    print("=" * 55)
    plot_roc_curves(models,         X_val, y_val, FIGURES_DIR / "roc_curves.png")
    plot_confusion_matrices(models, X_val, y_val, FIGURES_DIR / "confusion_matrices.png")
    plot_pr_curves(models,          X_val, y_val, FIGURES_DIR / "pr_curves.png")
    plot_metrics_bar(metrics_df,                  FIGURES_DIR / "metrics_bar.png")

    # ── 4. 最终结论 ────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 4 / 4  最终结论")
    print("=" * 55)
    best_model = metrics_df["ROC-AUC"].idxmax()
    best_auc   = metrics_df.loc[best_model, "ROC-AUC"]
    best_f1    = metrics_df.loc[best_model, "F1"]
    print(f"  最优模型（ROC-AUC）: {best_model}")
    print(f"  ROC-AUC = {best_auc:.4f}   F1 = {best_f1:.4f}")
    print(f"\n  所有图表已保存至: outputs/figures/")
    print(f"  性能报告已保存至: outputs/eval_report.csv")

    return metrics_df


if __name__ == "__main__":
    run()