# -*- coding: utf-8 -*-
"""
车险报价单成交概率预测 - XGBoost + Optuna
==================================================
功能：
1. 数据加载与清洗
2. 类别特征编码（保留编码映射，方便产线复用）
3. 特征工程
4. Optuna 自动调参
5. XGBoost 二分类训练
6. 模型评估与可视化
7. 模型保存

适用于：
- 数据量：约 6GB CSV
- CPU环境：24核 / 48线程 / 64GB内存
- Sklearn 风格 API

作者建议：
- Python >= 3.10
- xgboost >= 2.0
- optuna >= 3.0

==================================================
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)

from xgboost import XGBClassifier, plot_tree

# =========================================================
# 配置区域
# =========================================================

DATA_PATH = "your_data.csv"

TARGET_COL = "保单件数"

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 可用特征
FEATURE_COLUMNS = [
    '自主定价系数',
    '手续费不含税金额',
    '众享分_商交关联',
    '众享分',
    '二级能源种类名称',
    '新续转标识名称',
    '无赔款优待系数',
    '自主定价系数均值',
    '连续承保年数',
    '纯风险保费',
    '使用年限',
    '尊享分_商交关联',
    '保费不含税',
    '车损险限额',
    '车辆实际价值',
    '被保险人年龄',
    '被保险人性别',
    '本地车型库中的新车购置价',
    '标准保费',
    '二级机构名称',
    '三级机构名称',
    '四级机构名称',
    '品牌',
    '车型名称',
    '出险次数',
    '二手车标志'
]

# 需要编码的字段
CATEGORICAL_COLUMNS = [
    '二级能源种类名称',
    '新续转标识名称',
    '二级机构名称',
    '三级机构名称',
    '四级机构名称',
    '品牌',
    '车型名称',
    '被保险人性别'
]

# 输出目录
OUTPUT_DIR = "model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CPU线程数（充分利用48逻辑线程）
N_JOBS = 48

# =========================================================
# 1. 数据加载与预处理
# =========================================================

def load_and_preprocess_data(file_path):
    """
    数据加载、缺失值处理、编码处理
    """

    print("=" * 60)
    print("开始加载数据...")
    print("=" * 60)

    use_cols = FEATURE_COLUMNS + [TARGET_COL]

    df1 = pd.read_csv('D:\车险智能报价\训练数据\报价单去重晚.csv',usecols=use_cols)
    df2 = pd.read_csv('D:\车险智能报价\训练数据\报价单去重早.csv',usecols=use_cols)

    # 合并
    combined = pd.concat([df1, df2], ignore_index=True)
    # combined = pd.read_csv('D:\车险智能报价\去重数据.csv')
    # 随机打乱
    df = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"原始数据量: {len(df)}")

    # -----------------------------------------------------
    # 标签构建
    # -----------------------------------------------------
    df["label"] = (df[TARGET_COL] > 0).astype(int)

    # -----------------------------------------------------
    # 缺失值处理
    # 删除任意特征为空的样本
    # -----------------------------------------------------
    before_rows = len(df)

    df_clean = df.dropna(subset=FEATURE_COLUMNS)

    after_rows = len(df_clean)

    print(f"删除前数据量: {before_rows}")
    print(f"删除后数据量: {after_rows}")
    print(f"删除缺失值样本数量: {before_rows - after_rows}")

    # -----------------------------------------------------
    # 类别编码
    # 产线部署时需复用编码规则
    # -----------------------------------------------------
    encoding_maps = {}

    for col in CATEGORICAL_COLUMNS:

        # 转字符串防止混合类型
        df_clean[col] = df_clean[col].astype(str)

        unique_values = sorted(df_clean[col].unique())

        mapping = {v: i for i, v in enumerate(unique_values)}

        encoding_maps[col] = mapping

        df_clean[col] = df_clean[col].map(mapping)

        print(f"\n字段编码完成: {col}")
        print(f"类别数量: {len(mapping)}")

    # 保存编码映射
    encoding_path = os.path.join(
        OUTPUT_DIR,
        "encoding_mappings.json"
    )

    with open(encoding_path, "w", encoding="utf-8") as f:
        json.dump(
            encoding_maps,
            f,
            ensure_ascii=False,
            indent=4
        )

    print(f"\n编码映射已保存: {encoding_path}")

    return df_clean, encoding_maps


# =========================================================
# 2. 特征工程
# =========================================================

def feature_engineering(df):
    """
    构造业务组合特征
    """

    print("=" * 60)
    print("开始特征工程...")
    print("=" * 60)

    # -----------------------------------------------------
    # 保费风险比
    # -----------------------------------------------------
    df["保费风险比"] = (
        df["保费不含税"] /
        (df["纯风险保费"] + 1e-6)
    )

    # -----------------------------------------------------
    # 车辆折旧率
    # -----------------------------------------------------
    df["车辆折旧率"] = (
        df["车辆实际价值"] /
        (df["本地车型库中的新车购置价"] + 1e-6)
    )

    # -----------------------------------------------------
    # 出险频率特征
    # -----------------------------------------------------
    df["年均出险次数"] = (
        df["出险次数"] /
        (df["连续承保年数"] + 1)
    )

    # -----------------------------------------------------
    # 风险定价偏离
    # -----------------------------------------------------
    df["自主定价偏离度"] = (
        df["自主定价系数"] -
        df["自主定价系数均值"]
    )

    # -----------------------------------------------------
    # 车龄价值交叉
    # -----------------------------------------------------
    df["车龄价值交叉"] = (
        df["使用年限"] *
        df["车辆实际价值"]
    )

    print("特征工程完成")

    return df


# =========================================================
# 3. 模型训练 + Optuna调参
# =========================================================

def train_model(X_train, y_train, X_valid, y_valid):

    print("=" * 60)
    print("开始 Optuna 超参数优化...")
    print("=" * 60)

    # 类别不平衡处理
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    def objective(trial):

        params = {

            "objective": "binary:logistic",

            "eval_metric": "auc",

            "tree_method": "hist",

            "max_depth": trial.suggest_int("max_depth", 4, 12),

            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.2,
                log=True
            ),

            "n_estimators": trial.suggest_int(
                "n_estimators",
                300,
                1500
            ),

            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                1,
                15
            ),

            "subsample": trial.suggest_float(
                "subsample",
                0.6,
                1.0
            ),

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0
            ),

            "gamma": trial.suggest_float(
                "gamma",
                0,
                10
            ),

            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-3,
                10,
                log=True
            ),

            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1e-3,
                10,
                log=True
            ),

            "scale_pos_weight": pos_weight,

            "random_state": 42,

            "n_jobs": N_JOBS,

            "verbosity": 0
        }

        model = XGBClassifier(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )

        preds = model.predict_proba(X_valid)[:, 1]

        auc = roc_auc_score(y_valid, preds)

        return auc

    # -----------------------------------------------------
    # Optuna Study
    # -----------------------------------------------------

    study = optuna.create_study(
        direction="maximize",
        study_name="xgboost_insurance_auc"
    )

    study.optimize(
        objective,
        n_trials=50,
        n_jobs=8,
        show_progress_bar=True
    )

    print("\nBest AUC:", study.best_value)
    print("\nBest Params:")
    print(study.best_params)

    best_params = study.best_params

    best_params.update({
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss"],
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": N_JOBS,
        "scale_pos_weight": pos_weight
    })

    # -----------------------------------------------------
    # 最终模型
    # -----------------------------------------------------

    model = XGBClassifier(**best_params)

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_train, y_train),
            (X_valid, y_valid)
        ],
        verbose=100
    )

    return model, study


# =========================================================
# 4. 模型评估
# =========================================================

def evaluate_model(model, X_test, y_test):

    print("=" * 60)
    print("模型评估")
    print("=" * 60)

    # -----------------------------------------------------
    # 概率预测
    # -----------------------------------------------------
    y_prob = model.predict_proba(X_test)[:, 1]

    y_pred = (y_prob >= 0.5).astype(int)

    # -----------------------------------------------------
    # AUC
    # -----------------------------------------------------
    auc = roc_auc_score(y_test, y_prob)

    print(f"\nAUC: {auc:.6f}")

    # -----------------------------------------------------
    # Top20%转化率
    # -----------------------------------------------------
    result_df = pd.DataFrame({
        "label": y_test,
        "prob": y_prob
    })

    result_df = result_df.sort_values(
        by="prob",
        ascending=False
    )

    top_20_num = int(len(result_df) * 0.2)

    top_20_df = result_df.iloc[:top_20_num]

    top20_conversion_rate = top_20_df["label"].mean()

    print(f"Top20%转化率: {top20_conversion_rate:.6f}")

    # -----------------------------------------------------
    # 分类报告
    # -----------------------------------------------------
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # -----------------------------------------------------
    # Precision / Recall
    # -----------------------------------------------------
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")

    # -----------------------------------------------------
    # 混淆矩阵
    # -----------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    disp.plot(ax=ax)

    plt.title("Confusion Matrix")

    cm_path = os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(cm_path, dpi=300, bbox_inches="tight")

    plt.close()

    print(f"混淆矩阵已保存: {cm_path}")

    # -----------------------------------------------------
    # 训练过程曲线
    # -----------------------------------------------------
    results = model.evals_result()

    plt.figure(figsize=(10, 6))

    plt.plot(
        results["validation_0"]["auc"],
        label="Train AUC"
    )

    plt.plot(
        results["validation_1"]["auc"],
        label="Valid AUC"
    )

    plt.xlabel("Iteration")
    plt.ylabel("AUC")
    plt.title("XGBoost Training Curve")
    plt.legend()

    train_curve_path = os.path.join(
        OUTPUT_DIR,
        "training_curve.png"
    )

    plt.savefig(
        train_curve_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"训练曲线已保存: {train_curve_path}")

    # -----------------------------------------------------
    # 特征重要性
    # -----------------------------------------------------
    feature_importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance": model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        by="importance",
        ascending=False
    )

    print("\nTop20重要特征:")
    print(feature_importance.head(20))

    # -----------------------------------------------------
    # XGBoost树结构可视化
    # 注意：
    # 树太大会很慢，仅画第一棵树
    # -----------------------------------------------------
    plt.figure(figsize=(30, 15))

    plot_tree(
        model,
        num_trees=0,
        rankdir='LR'
    )

    tree_path = os.path.join(
        OUTPUT_DIR,
        "xgb_model_structure.png"
    )

    plt.savefig(
        tree_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"模型结构图已保存: {tree_path}")

    metrics = {
        "auc": auc,
        "top20_conversion_rate": top20_conversion_rate,
        "precision": precision,
        "recall": recall
    }

    return metrics


# =========================================================
# 5. 保存模型与结果
# =========================================================

def save_results(model, metrics):

    print("=" * 60)
    print("保存模型与结果")
    print("=" * 60)

    # -----------------------------------------------------
    # 保存模型
    # -----------------------------------------------------
    model_path = os.path.join(
        OUTPUT_DIR,
        "xgb_insurance_model.pkl"
    )

    joblib.dump(model, model_path)

    print(f"模型已保存: {model_path}")

    # -----------------------------------------------------
    # 保存指标
    # -----------------------------------------------------
    metrics_path = os.path.join(
        OUTPUT_DIR,
        "metrics.json"
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=4
        )

    print(f"评估指标已保存: {metrics_path}")


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # 1. 数据加载与清洗
    # -----------------------------------------------------
    df, encoding_maps = load_and_preprocess_data(DATA_PATH)

    # -----------------------------------------------------
    # 2. 特征工程
    # -----------------------------------------------------
    df = feature_engineering(df)

    # -----------------------------------------------------
    # 3. 准备训练数据
    # -----------------------------------------------------
    feature_cols = [
        col for col in df.columns
        if col not in [TARGET_COL, "label"]
    ]

    X = df[feature_cols]

    y = df["label"]

    # -----------------------------------------------------
    # 划分数据集
    # -----------------------------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    print(f"\n训练集: {X_train.shape}")
    print(f"验证集: {X_valid.shape}")
    print(f"测试集: {X_test.shape}")

    # -----------------------------------------------------
    # 4. 模型训练
    # -----------------------------------------------------
    model, study = train_model(
        X_train,
        y_train,
        X_valid,
        y_valid
    )

    # -----------------------------------------------------
    # 5. 模型评估
    # -----------------------------------------------------
    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    # -----------------------------------------------------
    # 6. 保存结果
    # -----------------------------------------------------
    save_results(model, metrics)

    print("\n全部流程完成")


# =========================================================
# 程序入口
# =========================================================

if __name__ == "__main__":
    main()