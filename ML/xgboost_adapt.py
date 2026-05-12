import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve
)

from xgboost import XGBClassifier, plot_importance

warnings.filterwarnings('ignore')

# =========================
# 中文显示
# =========================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 1. 特征配置
# =========================
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

CATEGORICAL_COLUMNS = [
    '二级能源种类名称',
    '新续转标识名称',
    '二级机构名称',
    '三级机构名称',
    '四级机构名称',
    '品牌',
    '车型名称'
]

TARGET_COLUMN = '保单件数'

# =========================
# 2. 数据加载
# =========================
def load_data():

    df1 = pd.read_csv(r'D:\车险智能报价\训练数据\报价单去重晚.csv')
    df2 = pd.read_csv(r'D:\车险智能报价\训练数据\报价单去重早.csv')

    combined = pd.concat([df1, df2], ignore_index=True)

    print("原始总数据量:", combined.shape)

    return combined


# =========================
# 3. 数据预处理
# =========================
def preprocess_data(df):

    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

    # 标签构造
    df['label'] = (df[TARGET_COLUMN] > 0).astype(int)

    print("\n目标变量分布:")
    print(df['label'].value_counts())

    print(f"\n整体成交率: {df['label'].mean():.2%}")

    # 缺失值处理
    df = df.dropna()

    print("\n删除缺失值后数据量:", df.shape)

    print("\n删除缺失值后成交率:")
    print(f"{df['label'].mean():.2%}")

    return df


# =========================
# 4. 特征工程
# =========================
def feature_engineering(df):

    df = df.copy()

    # =========================
    # 类别特征 -> category
    # =========================
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype('category')

    # =========================
    # 数值组合特征
    # =========================

    # 报价竞争力
    df['费率比'] = df['保费不含税'] / (df['标准保费'] + 1e-6)

    # 车辆折旧率
    df['车辆折旧率'] = (
        df['车辆实际价值']
        / (df['本地车型库中的新车购置价'] + 1e-6)
    )

    # 风险保费占比
    df['风险保费比'] = (
        df['纯风险保费']
        / (df['保费不含税'] + 1e-6)
    )

    # 连续承保 + 出险联合
    df['连续承保出险比'] = (
        df['连续承保年数']
        / (df['出险次数'] + 1)
    )

    # 年龄风险
    df['年龄车辆联合风险'] = (
        df['被保险人年龄']
        * df['使用年限']
    )

    return df


# =========================
# 5. 数据切分
# =========================
def split_data(df):

    X = df.drop(columns=[TARGET_COLUMN, 'label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# =========================
# 6. 寻找最佳阈值
# =========================
def find_best_threshold(y_true, y_prob):

    precisions, recalls, thresholds = precision_recall_curve(
        y_true,
        y_prob
    )

    f1_scores = (
        2 * precisions * recalls
        / (precisions + recalls + 1e-6)
    )

    best_idx = np.argmax(f1_scores)

    best_threshold = thresholds[best_idx]

    print("\n" + "=" * 60)
    print("最佳阈值搜索结果")
    print("=" * 60)

    print(f"最佳阈值:      {best_threshold:.4f}")
    print(f"最佳Precision: {precisions[best_idx]:.4f}")
    print(f"最佳Recall:    {recalls[best_idx]:.4f}")
    print(f"最佳F1:        {f1_scores[best_idx]:.4f}")

    return best_threshold


# =========================
# 7. TopK命中率
# =========================
def top_k_lift(y_true, y_prob, k=0.2):

    temp = pd.DataFrame({
        'y': y_true,
        'prob': y_prob
    })

    temp = temp.sort_values(
        by='prob',
        ascending=False
    )

    top_k = temp.head(int(len(temp) * k))

    return top_k['y'].mean()


# =========================
# 8. 模型训练
# =========================
def train_model(X_train, y_train, X_test, y_test):

    # 类别不平衡处理
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    scale_pos_weight = neg / pos

    print("\n" + "=" * 60)
    print("类别不平衡处理")
    print("=" * 60)

    print(f"负样本数: {neg}")
    print(f"正样本数: {pos}")
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(

        # 树数量
        n_estimators=1000,

        # 学习率
        learning_rate=0.03,

        # 树深
        max_depth=6,

        # 随机采样
        subsample=0.8,
        colsample_bytree=0.8,

        # 防过拟合
        min_child_weight=5,
        gamma=1,

        reg_alpha=1,
        reg_lambda=3,

        # 类别不平衡
        scale_pos_weight=scale_pos_weight,

        # categorical支持
        enable_categorical=True,
        tree_method='hist',

        # 评估
        eval_metric='auc',

        # 随机种子
        random_state=42
    )

    eval_set = [
        (X_train, y_train),
        (X_test, y_test)
    ]

    model.fit(
        X_train,
        y_train,

        eval_set=eval_set,

        verbose=50
    )

    return model


# =========================
# 9. 模型评估
# =========================
def evaluate_model(model, X_test, y_test):

    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)

    # 概率预测
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 自动寻找最佳阈值
    best_threshold = find_best_threshold(
        y_test,
        y_pred_proba
    )

    # 使用最佳阈值分类
    y_pred = (
        y_pred_proba > best_threshold
    ).astype(int)

    # 指标
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print("\n" + "=" * 60)
    print("最终评估结果")
    print("=" * 60)

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"AUC-ROC   : {auc:.4f}")

    print("\n" + "=" * 60)
    print("分类报告")
    print("=" * 60)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=['未成交', '成交']
        )
    )

    print("\n" + "=" * 60)
    print("混淆矩阵")
    print("=" * 60)

    print(confusion_matrix(y_test, y_pred))

    # TopK
    print("\n" + "=" * 60)
    print("业务指标")
    print("=" * 60)

    for k in [0.05, 0.1, 0.2, 0.3]:

        lift = top_k_lift(
            y_test,
            y_pred_proba,
            k
        )

        print(f"Top {int(k*100)}% 转化率: {lift:.4f}")

    print(f"\n整体转化率: {y_test.mean():.4f}")

    return y_pred_proba


# =========================
# 10. 训练过程可视化
# =========================
def plot_training_curve(model):

    results = model.evals_result()

    plt.figure(figsize=(10, 6))

    plt.plot(
        results['validation_0']['auc'],
        label='Train AUC'
    )

    plt.plot(
        results['validation_1']['auc'],
        label='Test AUC'
    )

    plt.legend()

    plt.title('Training AUC')

    plt.savefig('training_auc_curve.png')

    plt.close()

    print("\n训练曲线已保存: training_auc_curve.png")


# =========================
# 11. 特征重要性
# =========================
def plot_feature_importance_chart(model):

    plt.figure(figsize=(12, 10))

    plot_importance(
        model,
        max_num_features=30,
        importance_type='gain'
    )

    plt.title('Feature Importance')

    plt.savefig('feature_importance.png')

    plt.close()

    print("特征重要性图已保存: feature_importance.png")


# =========================
# 12. ROC曲线
# =========================
def plot_roc(y_test, y_prob):

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(8, 6))

    plt.plot(fpr, tpr)

    plt.xlabel("FPR")
    plt.ylabel("TPR")

    plt.title("ROC Curve")

    plt.savefig("roc_curve.png")

    plt.close()

    print("ROC曲线已保存: roc_curve.png")


# =========================
# 13. 主函数
# =========================
def main():

    print("=" * 60)
    print("车险报价成交预测模型")
    print("=" * 60)

    # 读取数据
    df = load_data()

    # 数据预处理
    df = preprocess_data(df)

    # 特征工程
    df = feature_engineering(df)

    # 数据切分
    X_train, X_test, y_train, y_test = split_data(df)

    print("\n训练集大小:", X_train.shape)
    print("测试集大小:", X_test.shape)

    # 模型训练
    model = train_model(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # 模型评估
    y_prob = evaluate_model(
        model,
        X_test,
        y_test
    )

    # 可视化
    plot_training_curve(model)

    plot_feature_importance_chart(model)

    plot_roc(y_test, y_prob)

    print("\n预测概率示例:")
    print(y_prob[:20])


# =========================
# 14. 执行
# =========================
if __name__ == "__main__":
    main()