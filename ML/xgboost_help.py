import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import xgboost as xgb
from xgboost import plot_importance, plot_tree
import warnings

warnings.filterwarnings('ignore')
import joblib
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_and_prepare_data(file_path):
    """
    加载数据并准备特征
    """
    print("=" * 50)
    print("步骤1: 加载数据")
    print("=" * 50)

    # 加载数据
    df1 = pd.read_csv('D:\车险智能报价\训练数据\报价单去重晚.csv')
    df2 = pd.read_csv('D:\车险智能报价\训练数据\报价单去重早.csv')

    # 合并
    combined = pd.concat([df1, df2], ignore_index=True)
    # combined = pd.read_csv('D:\车险智能报价\去重数据.csv')
    # 随机打乱
    df = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"原始数据形状: {df.shape}")
    print(f"原始数据列数: {len(df.columns)}")

    # 根据文档2，创建目标变量
    # 保单件数 > 0 表示成交，= 0 表示未成交
    df['是否成交'] = (df['保单件数'] > 0).astype(int)
    print(f"目标变量分布:\n{df['是否成交'].value_counts()}")
    print(f"成交比例: {df['是否成交'].mean():.2%}")

    return df


def select_features(df):
    """
    根据特征重要性排名选择前20个特征
    基于文档2中的特征重要性排名
    """
    print("\n" + "=" * 50)
    print("步骤2: 选择特征")
    print("=" * 50)

    # 根据文档2中的特征重要性排名选择前20个特征
    important_features = [
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

    # 检查哪些特征在数据中实际存在
    available_features = []
    missing_features = []

    for feature in important_features:
        if feature in df.columns:
            available_features.append(feature)
        else:
            missing_features.append(feature)

    print(f"可用的重要特征 ({len(available_features)}个): {available_features}")
    print(f"缺失的重要特征 ({len(missing_features)}个): {missing_features}")

    # 添加目标变量
    available_features.append('是否成交')

    # 选择这些特征
    selected_df = df[available_features].copy()
    print(f"testtest : {available_features[:-1]}")
    return selected_df, available_features[:-1]  # 排除目标变量


def handle_missing_values(df, features):
    """
    处理缺失值：直接删除有缺失的行
    """
    print("\n" + "50")
    print("步骤4: 处理缺失值")
    print("=" * 50)

    original_shape = df.shape
    print(f"删除前数据形状: {original_shape}")

    # 只保留选择的特征
    df_selected = df[features + ['是否成交']].copy()

    # 删除有缺失值的行
    df_clean = df_selected.dropna()

    new_shape = df_clean.shape
    print(f"删除后数据形状: {new_shape}")
    print(f"删除行数: {original_shape[0] - new_shape[0]}")
    # print(f"删除比例: {(original_shape[0] - new_shape[0] / original_shape[0]):.2 %}")

    print(f"删除缺失值后目标变量分布:\n{df_clean['是否成交'].value_counts()}")
    print(f"删除缺失值后成交比例: {df_clean['是否成交'].mean():.2%}")

    return df_clean



# def handle_missing_values(df, features):
#     """处理缺失值 - 使用填充而非删除"""
#     print("处理缺失值（采用填充策略）...")
#     df_clean = df.copy()
#
#     # 分类变量填充
#     categorical_cols = df_clean[features].select_dtypes(include=['object']).columns.tolist()
#     for col in categorical_cols:
#         if col in df_clean.columns:
#             mode_value = df_clean[col].mode()
#             fill_value = mode_value[0] if not mode_value.empty else '未知'
#             df_clean[col] = df_clean[col].fillna(fill_value)
#             print(f"分类变量 {col}: 填充了 {df_clean[col].isnull().sum()} 个缺失值")
#
#     # 数值变量填充
#     numeric_cols = df_clean[features].select_dtypes(include=[np.number]).columns.tolist()
#     for col in numeric_cols:
#         if col in df_clean.columns:
#             median_value = df_clean[col].median()
#             df_clean[col] = df_clean[col].fillna(median_value)
#             print(f"数值变量 {col}: 填充了 {df_clean[col].isnull().sum()} 个缺失值")
#
#     print("缺失值填充完成")
#     print(f"剩余缺失值总数: {df_clean.isnull().sum().sum()}")
#     return df_cl


def encode_categorical_features(df):
    """
    对分类特征进行编码
    """
    print("\n" + "=" * 50)
    print("步骤5: 编码分类特征")
    print("=" * 50)

    df_encoded = df.copy()

    # 识别分类特征（基于文档2中的特征编码部分）
    categorical_features = [
        '二级能源种类名称',
        '新续转标识名称',
        '二级机构名称',
        '三级机构名称',
        '四级机构名称',
        '品牌',
        '车型名称'
    ]

    # 只处理数据中存在的分类特征
    existing_cat_features = [f for f in categorical_features if f in df_encoded.columns]

    label_encoders = {}

    for feature in existing_cat_features:
        if feature in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[feature] = le.fit_transform(df_encoded[feature].astype(str))
            label_encoders[feature] = le
            print(f"已编码特征: {feature}，类别数: {len(le.classes_)}")

    return df_encoded, label_encoders


def prepare_model_data(df):
    """
    准备模型数据
    """
    print("\n" + "=" * 50)
    print("步骤6: 准备模型数据")
    print("=" * 50)

    # 分离特征和目标变量
    X = df.drop('是否成交', axis=1)
    y = df['是否成交']

    print(f"特征数据形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")
    print(f"正样本比例: {y.mean():.2%}")

    # 计算类别权重（用于处理不平衡数据）
    class_counts = y.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1]
    print(f"类别权重 (scale_pos_weight): {scale_pos_weight}")

    return X, y, scale_pos_weight


def train_xgboost_model(X, y, scale_pos_weight):
    """
    训练XGBoost模型
    """
    print("\n" + "=" * 50)
    print("步骤7: 训练XGBoost模型")
    print("=" * 50)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    print(f"训练集正样本比例: {y_train.mean():.2%}")
    print(f"测试集正样本比例: {y_test.mean():.2%}")

    # 创建XGBoost模型
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=4,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        reg_lambda=1.0
    )

    # 训练模型
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )

    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    """
    print("\n" + "=" * 50)
    print("步骤8: 模型评估")
    print("=" * 50)

    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")

    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩阵:")
    print(cm)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def visualize_training_process(model, X_test, y_test, eval_results):
    """
    可视化训练过程和模型结果
    """
    print("\n" + "=" * 50)
    print("步骤9: 可视化")
    print("=" * 50)

    # 创建可视化目录
    os.makedirs('model_visualizations', exist_ok=True)

    # 1. 特征重要性图
    plt.figure(figsize=(12, 8))
    plot_importance(model, max_num_features=20)
    plt.title('XGBoost特征重要性排名 (Top 20)', fontsize=16)
    plt.tight_layout()
    plt.savefig('model_visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # 2. ROC曲线
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, eval_results['y_pred_proba'])
    plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {eval_results["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
    plt.xlabel('假正率 (False Positive Rate)', fontsize=12)
    plt.ylabel('真正率 (True Positive Rate)', fontsize=12)
    plt.title('ROC曲线', fontsize=16)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig('model_visualizations/roc_curve.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # 3. 精确率-召回率曲线
    plt.figure(figsize=(10, 8))
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, eval_results['y_pred_proba'])
    plt.plot(recall_vals, precision_vals)
    plt.xlabel('召回率 (Recall)', fontsize=12)
    plt.ylabel('精确率 (Precision)', fontsize=12)
    plt.title('精确率-召回率曲线', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.savefig('model_visualizations/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # 4. 训练过程图（学习曲线）
    results = model.evals_result()
    plt.figure(figsize=(12, 8))

    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)

    plt.plot(x_axis, results['validation_0']['logloss'], label='训练集损失')
    plt.plot(x_axis, results['validation_1']['logloss'], label='验证集损失')
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('损失值 (Log Loss)', fontsize=12)
    plt.title('XGBoost训练过程学习曲线', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('model_visualizations/learning_curve.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # 5. 预测概率分布图
    plt.figure(figsize=(12, 8))

    # 分离成交和未成交的预测概率
    proba_成交 = eval_results['y_pred_proba'][y_test == 1]
    proba_未成交 = eval_results['y_pred_proba'][y_test == 0]

    plt.hist(proba_未成交, bins=50, alpha=0.5, label='未成交', color='red')
    plt.hist(proba_成交, bins=50, alpha=0.5, label='成交', color='green')
    plt.xlabel('预测成交概率', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.title('预测概率分布', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('model_visualizations/prediction_probability_distribution.png', dpi=300, bbox_inches='tight')
    # plt.show()

    print("可视化图片已保存到 'model_visualizations' 目录")


def save_model_and_results(model, features, eval_results):
    """
    保存模型和结果
    """
    print("\n" + "=" * 50)
    print("步骤10: 保存模型和结果")
    print("=" * 50)

    # 保存模型
    model_filename = 'xgboost_quote_conversion_model.pkl'
    joblib.dump(model, model_filename)
    print(f"模型已保存到: {model_filename}")

    # 保存特征列表
    features_filename = 'model_features.txt'
    with open(features_filename, 'w', encoding='utf-8') as f:
        for feature in features:
            f.write(f"{feature}\n")
    print(f"特征列表已保存到: {features_filename}")

    # 保存评估结果
    results_filename = 'model_evaluation_results.txt'
    with open(results_filename, 'w', encoding='utf-8') as f:
        f.write("模型评估结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"准确率 (Accuracy): {eval_results['accuracy']:.4f}\n")
        f.write(f"精确率 (Precision): {eval_results['precision']:.4f}\n")
        f.write(f"召回率 (Recall): {eval_results['recall']:.4f}\n")
        f.write(f"F1分数: {eval_results['f1']:.4f}\n")
        f.write(f"AUC-ROC: {eval_results['roc_auc']:.4f}\n")
    print(f"评估结果已保存到: {results_filename}")


def predict_probability(model, new_data):
    """
    使用训练好的模型预测报价单成交概率
    """
    # 确保输入数据包含所有必要的特征
    required_features = model.get_booster().feature_names

    if not all(feature in new_data.columns for feature in required_features):
        missing = [f for f in required_features if f not in new_data.columns]
        raise ValueError(f"缺少以下特征: {missing}")

    # 预测概率
    probabilities = model.predict_proba(new_data[required_features])[:, 1]

    return probabilities


def main():
    """
    主函数：执行完整的模型训练流程
    """
    print("车险报价成交概率预测模型")
    print("=" * 50)

    # 步骤1: 加载数据
    # 注意：请将 'your_data.csv' 替换为实际数据文件路径
    data_file = 'your_data.csv'  # 请替换为实际文件路径
    df = load_and_prepare_data(data_file)

    # 步骤2: 创建缺失的特征
    # df = create_missing_features(df)

    # 步骤3: 选择特征
    selected_df, important_features = select_features(df)

    # 步骤4: 处理缺失值
    df_clean = handle_missing_values(df, important_features)

    # 步骤5: 编码分类特征
    df_encoded, label_encoders = encode_categorical_features(df_clean)

    # 步骤6: 准备模型数据
    X, y, scale_pos_weight = prepare_model_data(df_encoded)

    # 步骤7: 训练XGBoost模型
    model, X_train, X_test, y_train, y_test = train_xgboost_model(X, y, scale_pos_weight)

    # # 执行网格搜索
    # best_model, best_params, cv_results = hyperparameter_tuning(
    #     X, y, scale_pos_weight, cv_folds=3, n_jobs=4
    # )
    #
    # # 使用最优参数在整个数据集上训练模型
    # model, X_train, X_test, y_train, y_test = train_optimized_model(
    #     X, y, scale_pos_weight, best_params
    # )

    # 步骤8: 评估模型
    eval_results = evaluate_model(model, X_test, y_test)

    # 步骤9: 可视化
    visualize_training_process(model, X_test, y_test, eval_results)

    # 步骤10: 保存模型和结果
    save_model_and_results(model, important_features, eval_results)

    print("\n" + "=" * 50)
    print("模型训练完成！")
    print("=" * 50)

    # 示例：使用模型进行预测
    print("\n示例预测:")
    sample_data = X_test.iloc[:5].copy()
    probabilities = predict_probability(model, sample_data)

    for i, prob in enumerate(probabilities):
        print(f"样本 {i + 1}: 成交概率 = {prob:.4f} ({'可能成交' if prob > 0.5 else '可能未成交'})")

    return model, important_features, eval_results


if __name__ == "__main__":
    # 执行主函数
    model, features, results = main()
