# 贷款审批预测（loan-approval-ml）

本项目对应《机器学习实践课程》大作业，选题为：
- 题目1：使用不少于 3 种机器学习模型完成完整预测流程
- 项目3：贷款预测（Loan Prediction）

目标是基于申请人信息（婚姻、教育、收入、征信等）预测贷款是否获批（`Loan_Status`）。

## 1. 项目内容

项目按照以下流程实现：
- 目标分析
- 数据处理（缺失值、编码、列对齐）
- 特征工程（业务派生特征 + 偏态变换）
- 模型选择（逻辑回归 / 随机森林 / XGBoost）
- 模型训练与调优（5 折交叉验证 + GridSearchCV）
- 性能度量与结果分析

## 2. 目录结构

```text
loan-approval-ml/
├── data/
│   ├── raw/                  # 原始 train.csv / test.csv
│   └── processed/            # 清洗与特征工程结果
├── notebooks/
│   └── exploration.ipynb     # EDA 探索笔记本
├── src/
│   ├── preprocess.py         # 数据清洗与编码
│   ├── features.py           # 特征工程
│   ├── train.py              # 三模型训练 + 调参
│   └── evaluate.py           # 评估模块（待补充）
├── outputs/
│   ├── figures/              # 图表输出
│   ├── models/               # 已训练模型 .pkl
│   └── cv_summary.csv        # 交叉验证汇总
├── report/                   # 作业说明书
├── requirements.txt
└── README.md
```

## 3. 数据集说明

- 数据源：Kaggle Loan Prediction Problem Dataset  
  [https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset](https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset)
- 训练集：`data/raw/train.csv`
- 测试集：`data/raw/test.csv`
- 标签列：`Loan_Status`（`Y`=1, `N`=0）

## 4. 环境安装

建议 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 用 .venv\\Scripts\\activate
pip install -U pip
pip install pandas numpy scikit-learn matplotlib xgboost joblib jupyter
```

可将上述依赖写入 `requirements.txt` 后执行：

```bash
pip install -r requirements.txt
```

## 5. 运行方式

在项目根目录执行：

```bash
python src/preprocess.py
python src/features.py
python src/train.py
```

执行后可得到：
- 清洗数据：`data/processed/train_clean.csv`、`test_clean.csv`
- 特征数据：`data/processed/train_feat.csv`、`test_feat.csv`
- 特征图：`outputs/figures/feature_importance.png`
- 最优模型：`outputs/models/*.pkl`
- CV 汇总：`outputs/cv_summary.csv`

## 6. 当前训练结果（已生成）

来自 `outputs/cv_summary.csv`：

| 模型 | 基线 CV AUC | 调优 CV AUC | AUC 提升 |
|---|---:|---:|---:|
| LogisticRegression | 0.7205 | 0.7291 | +0.0086 |
| RandomForest | 0.7666 | 0.7709 | +0.0043 |
| XGBoost | 0.7467 | 0.7734 | +0.0267 |

当前最佳模型：`XGBoost`（调优 CV AUC = `0.7734`）。

## 7. 作业提交对照（来自任务文档）

- 截止时间：2026-05-17 24:00（平台提交）
- 需提交内容：
- 源代码工程包
- 运行界面截图
- 5分钟以内成果简介视频（MP4/MPEG 1920x1080）
- 大作业说明书（建议 4000 字以上，不含代码）

建议配套材料：
- `notebooks/exploration.ipynb`：EDA 过程与图表
- `outputs/figures/`：可直接插入说明书
- `outputs/cv_summary.csv`：模型对比结论依据

## 8. 后续建议

- 补全 `src/evaluate.py`：在独立验证集输出 Accuracy / Precision / Recall / F1 / ROC-AUC、混淆矩阵与 ROC 曲线。
- 将依赖固化到 `requirements.txt`，保证可复现。
- 在 `report/` 内按“目标分析→数据处理→特征工程→模型训练调优→结果分析→优缺点与局限”组织说明书。
