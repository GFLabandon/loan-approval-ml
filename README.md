# 贷款审批预测（Loan Approval Prediction）

> 机器学习实践课程考核大作业 · 题目 1  
> 使用 **3 种机器学习模型**，按标准 ML 流程完成贷款审批二分类预测，并对比各模型优缺点。

---

## 项目概述

本项目基于 [Kaggle Loan Prediction Dataset](https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset)，根据申请人的婚姻状况、教育程度、收入、征信记录等信息，预测贷款是否获批（`Loan_Status: Y/N`）。

完整实现了以下 ML 流程：

```
目标分析 → 数据处理 → 特征工程 → 模型选择 → 模型训练&调优 → 性能度量与模型应用
```

---

## 目录结构

```text
loan-approval-ml/
├── data/
│   ├── raw/                        # 原始数据（train.csv / test.csv）
│   └── processed/                  # 预处理与特征工程后的数据
│       ├── train_clean.csv         # 清洗后训练集（614×17）
│       ├── test_clean.csv          # 清洗后测试集（367×16）
│       ├── train_feat.csv          # 特征工程后训练集（614×20）
│       └── test_feat.csv           # 特征工程后测试集（367×19）
├── notebooks/
│   └── exploration.ipynb           # EDA 探索分析（含可视化）
├── src/
│   ├── preprocess.py               # 数据清洗与编码
│   ├── features.py                 # 特征工程（派生特征 + log变换 + 重要性分析）
│   ├── train.py                    # 三模型训练 + GridSearchCV 调优
│   └── evaluate.py                 # 性能评估 + 可视化图表生成
├── outputs/
│   ├── figures/                    # 所有图表输出
│   │   ├── feature_importance.png  # 随机森林特征重要性
│   │   ├── roc_curves.png          # 三模型 ROC 曲线对比
│   │   ├── confusion_matrices.png  # 三模型混淆矩阵对比
│   │   ├── pr_curves.png           # 三模型 PR 曲线对比
│   │   └── metrics_bar.png         # 五项指标分组柱状图
│   ├── models/                     # 最优模型（.pkl）
│   │   ├── LogisticRegression_best.pkl
│   │   ├── RandomForest_best.pkl
│   │   └── XGBoost_best.pkl
│   ├── cv_summary.csv              # 基线 vs 调优 CV AUC 对比表
│   └── eval_report.csv             # 验证集完整指标报告
├── report/                         # 大作业说明书
├── requirements.txt
└── README.md
```

---

## 数据集说明

| 项目 | 内容 |
|------|------|
| 来源 | Kaggle Loan Prediction Problem Dataset |
| 训练集 | 614 行 × 13 列 |
| 测试集 | 367 行 × 12 列 |
| 标签列 | `Loan_Status`（Y=批准→1，N=拒绝→0） |
| 类别分布 | Y:422 / N:192（约 2.2:1，存在轻度不均衡） |
| 缺失列 | Gender(13)、Married(3)、Dependents(15)、Self_Employed(32)、LoanAmount(22)、Loan_Amount_Term(14)、Credit_History(50) |

---

## 三种模型

| 模型 | 类型 | 不均衡处理 | 说明 |
|------|------|-----------|------|
| **逻辑回归** | 线性 | `class_weight='balanced'` | 线性基线，可解释性强 |
| **随机森林** | 集成树（Bagging） | `class_weight='balanced'` | 低方差，泛化稳健 |
| **XGBoost** | 梯度提升（Boosting） | `scale_pos_weight=0.4549` | 高精度，强特征捕获能力 |

---

## 特征工程

在原始 16 个特征基础上，构造 3 个业务驱动派生特征：

| 特征 | 公式 | 业务意义 |
|------|------|---------|
| `TotalIncome` | ApplicantIncome + CoapplicantIncome | 家庭合计收入 |
| `EMI` | LoanAmount / Loan_Amount_Term | 月均还款压力 |
| `BalanceIncome` | TotalIncome − EMI×1000 | 还款后剩余收入（偿债能力） |

高偏度列（偏度 2.7~7.5）均做 `log1p` 变换，最终特征数：**19 个**。

---

## 实验结果

### 交叉验证（5折 ROC-AUC，训练集 491 行）

| 模型 | 基线 CV AUC | 调优后 CV AUC | 提升 |
|------|------------|--------------|------|
| Logistic Regression | 0.7205 | 0.7291 | +0.0086 |
| Random Forest | 0.7666 | 0.7709 | +0.0043 |
| **XGBoost** | 0.7467 | **0.7734** | +0.0267 |

### 验证集评估（123 行，留出集）

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------|----------|-----------|--------|-----|---------|
| **Logistic Regression** | 0.8374 | 0.8916 | 0.8706 | **0.8810** | **0.8734** |
| Random Forest | 0.8293 | 0.8636 | **0.8941** | 0.8786 | 0.8260 |
| XGBoost | 0.8049 | **0.8861** | 0.8235 | 0.8537 | 0.8251 |

> **最优模型**（验证集 ROC-AUC）：逻辑回归（0.8734）  
> CV 阶段 XGBoost 领先，验证集逻辑回归反超，说明逻辑回归在此小样本任务中泛化更稳定。

---

## 环境配置

```bash
# 创建 conda 环境
conda create -n ml_hw python=3.10 -y
conda activate ml_hw

# 安装依赖
pip install -r requirements.txt
```

`requirements.txt` 内容：

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
matplotlib>=3.7
joblib>=1.3
jupyter
```

---

## 运行方式

在项目根目录按顺序执行：

```base
python src/preprocess.py   # 数据清洗与编码
python src/features.py     # 特征工程（含重要性图）
python src/train.py        # 三模型训练与调优（约 5~10 分钟）
python src/evaluate.py     # 评估 + 生成所有图表
```

或直接在 Jupyter 中查看 EDA 过程：

```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## 各模型优缺点总结

| 模型 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| 逻辑回归 | 可解释性强、训练快、系数直接表示特征影响方向 | 仅捕获线性关系，对特征交互建模能力弱 | 需要模型可解释性的金融风控场景 |
| 随机森林 | 对噪声鲁棒、自带特征重要性、不需标准化 | 训练慢、内存消耗大、模型黑盒 | 特征较多、数据量中等、追求稳定性 |
| XGBoost | 精度高、支持缺失值、内置正则化 | 超参数多、调优耗时、小数据集易过拟合 | 竞赛或追求最高精度的场景 |

---

## 提交说明（作业要求对照）

- [x] 源代码工程包（本仓库）
- [x] 3 种以上机器学习模型（LR / RF / XGBoost）
- [x] 标准 6 步 ML 流程（目标分析→数据处理→特征工程→模型选择→训练调优→性能度量）
- [x] 模型优缺点与局限性分析（见说明书 & 本 README）
- [ ] 运行界面截图（请在本地运行后截图）
- [ ] 5 分钟以内简介视频（MP4/MPEG 1920×1080）
- [ ] 大作业说明书（4000 字以上，查重 20% 以下）
