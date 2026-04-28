"""
features.py — 特征工程模块
贷款审批预测项目 | loan-approval-ml

流程：
  1. 加载 preprocess.py 输出的清洗数据
  2. 构造三个业务驱动的派生特征
       - TotalIncome      : 申请人 + 共同申请人收入之和
       - EMI              : 月均还款额（贷款额 / 还款期数）
       - BalanceIncome    : 还款后剩余收入（偿债能力代理指标）
  3. 对高偏度数值列做 log1p 变换（降低极端值影响）
  4. 保存最终特征表到 data/processed/
  5. 输出随机森林特征重要性图（outputs/figures/feature_importance.png）

注：log1p 平移量基于 train+test 全局最小值，避免测试集产生 NaN，
    同时与标签无关，不构成数据泄露。
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # 无显示器环境下使用非交互后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier


# ── 路径配置 ────────────────────────────────────────────────────────────────
_THIS_FILE    = Path(__file__).resolve()
PROJECT_ROOT  = _THIS_FILE.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR   = PROJECT_ROOT / "outputs" / "figures"

TRAIN_IN  = PROCESSED_DIR / "train_clean.csv"
TEST_IN   = PROCESSED_DIR / "test_clean.csv"
TRAIN_OUT = PROCESSED_DIR / "train_feat.csv"
TEST_OUT  = PROCESSED_DIR / "test_feat.csv"

LABEL_COL = "Loan_Status"

# 需要做 log1p 变换的列（原始列 + 派生列）
LOG_COLS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "TotalIncome",
    "EMI",
    "BalanceIncome",
]


# ── 字体配置（macOS 中文显示）───────────────────────────────────────────────
def _setup_font():
    """
    自动检测可用中文字体（macOS / Linux 均适用）。
    优先使用 PingFang SC，退而使用 Heiti TC，最后回退到 DejaVu Sans。
    """
    preferred = ["PingFang SC", "Heiti TC", "SimHei", "WenQuanYi Micro Hei"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in preferred:
        if font in available:
            plt.rcParams["font.family"] = font
            return font
    plt.rcParams["font.family"] = "DejaVu Sans"
    return "DejaVu Sans"


# ── Step 1：派生特征构造 ─────────────────────────────────────────────────────
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造三个业务驱动的派生特征：

    TotalIncome（总收入）
    ─────────────────────
      = ApplicantIncome + CoapplicantIncome
      贷款审批通常考虑家庭合计收入，而非仅主申请人收入。

    EMI（月均还款额 Equated Monthly Installment）
    ──────────────────────────────────────────────
      = LoanAmount / Loan_Amount_Term
      衡量申请人每月实际还款压力。
      数据集中 LoanAmount 单位为千元，Loan_Amount_Term 单位为月。

    BalanceIncome（还款后剩余收入）
    ────────────────────────────────
      = TotalIncome - (EMI × 1000)
      将 EMI 换算回与收入相同单位后相减，代理申请人偿债后的生活余裕度。
      负值表示收入不足以覆盖月供（高违约风险信号）。
    """
    df = df.copy()

    df["TotalIncome"]   = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["EMI"]           = df["LoanAmount"] / df["Loan_Amount_Term"]
    df["BalanceIncome"] = df["TotalIncome"] - df["EMI"] * 1000

    print("[feature] 派生特征已添加: TotalIncome, EMI, BalanceIncome")
    print(f"          TotalIncome   均值={df['TotalIncome'].mean():.0f}  "
          f"中位数={df['TotalIncome'].median():.0f}")
    print(f"          EMI           均值={df['EMI'].mean():.2f}  "
          f"中位数={df['EMI'].median():.2f}")
    print(f"          BalanceIncome 均值={df['BalanceIncome'].mean():.0f}  "
          f"负值比例={( df['BalanceIncome'] < 0).mean()*100:.1f}%")
    return df


# ── Step 2：log1p 平移量计算 ─────────────────────────────────────────────────
def compute_shifts(train_df: pd.DataFrame,
                   test_df: pd.DataFrame) -> dict:
    """
    预计算 log1p 所需的平移量（shift），基于 train+test 的全局最小值。

    为什么用全局 min 而非仅训练集 min：
      测试集 BalanceIncome 最小值（约 -10776）远小于训练集（约 -1768）。
      若只用训练集 min 计算 shift，测试集中更极端的负值经平移后仍为负，
      导致 log1p 输入 < -1 → 产生 NaN。

    数据泄露分析：
      shift 是与标签（Loan_Status）完全无关的纯数值常数，
      仅用于保证对数变换的定义域合法，不引入标签信息，不构成泄露。
    """
    shifts = {}
    for col in LOG_COLS:
        mins = []
        if col in train_df.columns:
            mins.append(train_df[col].min())
        if col in test_df.columns:
            mins.append(test_df[col].min())
        if not mins:
            continue
        global_min = min(mins)
        # 非负列 shift=0，含负值列 shift = |min| + 1
        shifts[col] = max(0.0, -global_min + 1.0)
    return shifts


# ── Step 3：log1p 变换 ───────────────────────────────────────────────────────
def log_transform(df: pd.DataFrame, shifts: dict,
                  label: str = "") -> pd.DataFrame:
    """
    对高偏度数值列进行 log1p 变换（ln(x+1)）。

    处理原始偏度（训练集）：
      ApplicantIncome   ≈ 6.5  → 变换后接近正态
      CoapplicantIncome ≈ 7.5
      LoanAmount        ≈ 2.7
      TotalIncome       ≈ 5.8（预估）
      EMI               ≈ 2.5（预估）
      BalanceIncome     含负值 → 先用全局 shift 平移再变换

    Args:
        df:     待变换的 DataFrame（训练集或测试集）
        shifts: compute_shifts() 返回的平移量字典
        label:  打印时显示的数据集名称
    """
    df = df.copy()
    tag = f"[{label}] " if label else ""

    for col in LOG_COLS:
        if col not in df.columns:
            continue

        shift = shifts.get(col, 0.0)
        if shift > 0:
            df[col] = np.log1p(df[col] + shift)
            print(f"[log1p] {tag}{col:<20} 含负值，平移 {shift:.2f} 后变换")
        else:
            df[col] = np.log1p(df[col])
            print(f"[log1p] {tag}{col:<20} 直接变换")

    return df


# ── Step 4：特征重要性分析 ───────────────────────────────────────────────────
def plot_feature_importance(X: pd.DataFrame, y: pd.Series,
                             save_path: Path) -> pd.Series:
    """
    使用随机森林（n_estimators=200，固定随机种子）计算特征重要性，
    并生成横向条形图保存至 outputs/figures/。

    Returns:
        importance_series: 按重要性降序排列的 Series（索引为特征名）
    """
    _setup_font()

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)

    importance = pd.Series(rf.feature_importances_, index=X.columns)
    importance_sorted_asc = importance.sort_values(ascending=True)   # barh 从下到上

    # ── 绘图 ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))

    threshold = importance_sorted_asc.quantile(0.75)
    colors = ["#4C72B0" if v >= threshold else "#A8C8E8"
              for v in importance_sorted_asc.values]
    bars = ax.barh(importance_sorted_asc.index, importance_sorted_asc.values,
                   color=colors, edgecolor="white", height=0.65)

    # 在条形末端标注数值
    for bar, val in zip(bars, importance_sorted_asc.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=9)

    ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
    ax.set_title("Random Forest Feature Importance\n(n_estimators=200, seed=42)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, importance_sorted_asc.max() * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[figure] 特征重要性图已保存 → {save_path}")
    print("\n特征重要性排名（Top 10）：")
    top10 = importance.sort_values(ascending=False).head(10)
    for rank, (feat, imp) in enumerate(top10.items(), 1):
        print(f"  {rank:2d}. {feat:<30} {imp:.4f}")

    return importance.sort_values(ascending=False)


# ── Step 5：数据验证 ─────────────────────────────────────────────────────────
def validate(df: pd.DataFrame, name: str) -> None:
    """检查无缺失值、无 inf 值。"""
    missing = df.isnull().sum().sum()
    inf_cnt = np.isinf(df.select_dtypes(include=np.number)).sum().sum()
    if missing > 0 or inf_cnt > 0:
        raise ValueError(
            f"[ERROR] {name}: 缺失值={missing}, inf值={inf_cnt}"
        )
    print(f"[validate] {name}: 无缺失值，无 inf 值 ✓")


# ── 主流程 ───────────────────────────────────────────────────────────────────
def run():
    """端到端执行特征工程并保存结果。"""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── 加载清洗后数据 ─────────────────────────────────────────────────────
    print("=" * 55)
    print("  STEP 1 / 5  加载预处理数据")
    print("=" * 55)
    train = pd.read_csv(TRAIN_IN)
    test  = pd.read_csv(TEST_IN)
    print(f"[load]  训练集: {train.shape}  测试集: {test.shape}")

    # ── 派生特征 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 2 / 5  派生特征构造")
    print("=" * 55)
    print("--- 训练集 ---")
    train = add_derived_features(train)
    print("--- 测试集 ---")
    test  = add_derived_features(test)

    # ── 计算全局平移量 ─────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 3 / 5  log1p 平移量计算（全局最小值）")
    print("=" * 55)
    shifts = compute_shifts(train, test)
    for col, s in shifts.items():
        tag = f"shift={s:.2f}" if s > 0 else "无需平移"
        print(f"  {col:<20} {tag}")

    # ── log1p 变换 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 4 / 5  log1p 变换（消除偏度）")
    print("=" * 55)
    train = log_transform(train, shifts, label="train")
    test  = log_transform(test,  shifts, label="test")

    validate(train, "训练集")
    validate(test,  "测试集")

    # ── 特征重要性 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 5 / 5  特征重要性分析")
    print("=" * 55)
    feat_cols = [c for c in train.columns if c != LABEL_COL]
    X_train   = train[feat_cols]
    y_train   = train[LABEL_COL]

    importance = plot_feature_importance(
        X_train, y_train,
        save_path=FIGURES_DIR / "feature_importance.png"
    )

    # ── 保存结果 ───────────────────────────────────────────────────────────
    train.to_csv(TRAIN_OUT, index=False)
    test.to_csv(TEST_OUT,   index=False)
    print(f"\n[save]  训练集特征 → {TRAIN_OUT}")
    print(f"[save]  测试集特征 → {TEST_OUT}")

    print("\n" + "=" * 55)
    print("  最终特征列表")
    print("=" * 55)
    print(f"特征数量: {len(feat_cols)}")
    print(f"特征列表: {feat_cols}")
    print(f"\n训练集 shape: {train.shape}")
    print(f"测试集 shape: {test.shape}")

    return train, test, importance


if __name__ == "__main__":
    run()