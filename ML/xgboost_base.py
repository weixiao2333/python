import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 1. 特征配置
# =========================
FEATURE_COLUMNS = [
    '自主定价系数', '手续费不含税金额', '众享分_商交关联', '众享分',
    '二级能源种类名称', '新续转标识名称', '无赔款优待系数', '自主定价系数均值',
    '连续承保年数', '纯风险保费', '使用年限', '尊享分_商交关联',
    '保费不含税', '车损险限额', '车辆实际价值', '被保险人年龄',
    '被保险人性别', '本地车型库中的新车购置价', '标准保费',
    '二级机构名称', '三级机构名称', '四级机构名称',
    '品牌', '车型名称', '出险次数', '二手车标志'
]

CATEGORICAL_COLUMNS = [
    '二级能源种类名称', '新续转标识名称',
    '二级机构名称', '三级机构名称', '四级机构名称',
    '品牌', '车型名称'
]

TARGET_COLUMN = '保单件数'


# =========================
# 2. 数据加载
# =========================
def load_data():
    df1 = pd.read_csv('D:\车险智能报价\训练数据\报价单去重晚.csv')
    df2 = pd.read_csv('D:\车险智能报价\训练数据\报价单去重早.csv')

    # 合并
    combined = pd.concat([df1, df2], ignore_index=True)
    # combined = pd.read_csv('D:\车险智能报价\去重数据.csv')
    # 随机打乱
    df = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# =========================
# 3. 数据预处理
# =========================
def preprocess_data(df):
    print("原始数据量:", df.shape)
    # 构造标签（是否成交）

    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]]
    df['label'] = (df[TARGET_COLUMN] > 0).astype(int)
    print(f"目标变量分布:\n{df['label'].value_counts()}")
    print(f"成交比例: {df['label'].mean():.2%}")

    # 删除缺失值
    df = df.dropna()

    print("删除缺失值后数据量:", df.shape)

    print(f"删除缺失值后目标变量分布:\n{df['label'].value_counts()}")
    print(f"删除缺失值后成交比例: {df['label'].mean():.2%}")
    return df


# =========================
# 4. 特征工程
# =========================
def feature_engineering(df):
    df = df.copy()

    # ===== 类别特征编码 =====
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # ===== 示例组合特征 =====
    df['费率比'] = df['保费不含税'] / (df['标准保费'] + 1e-6)
    df['车辆折旧率'] = df['车辆实际价值'] / (df['本地车型库中的新车购置价'] + 1e-6)
    df['风险保费比'] = df['纯风险保费'] / (df['保费不含税'] + 1e-6)

    return df


# =========================
# 5. 划分数据集
# =========================
def split_data(df):
    X = df.drop(columns=[TARGET_COLUMN, 'label'])
    y = df['label']

    return train_test_split(X, y, test_size=0.2, random_state=42)


# =========================
# 6. 模型训练
# =========================
def train_model(X_train, y_train, X_test, y_test):
    model = XGBClassifier(
        n_estimators=489,
        max_depth=9,
        learning_rate=0.09,
        subsample=0.8,
        colsample_bytree=0.67,
        eval_metric='logloss',
        use_label_encoder=False
    )

    eval_set = [(X_train, y_train), (X_test, y_test)]

    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=True
    )

    return model


# =========================
# 7. 模型评估
# =========================
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
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # 打印评估指标
    print(f"\n{'=' * 50}")
    print(f"模型评估指标")
    print(f"{'=' * 50}")
    # print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    # print(f"F1分数 (F1-Score):  {f1:.4f}")
    print(f"AUC-ROC:            {roc_auc:.4f}")

    # 分类报告
    print(f"\n{'=' * 50}")
    print(f"分类报告")
    print(f"{'=' * 50}")
    print(classification_report(y_test, y_pred, target_names=['未成交', '成交']))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"{'=' * 50}")
    print(f"混淆矩阵")
    print(f"{'=' * 50}")
    print(cm)

    print("Top20%转化率:", top_k_lift(y_test, y_pred_proba, 0.2))
    print("整体转化率:", y_test.mean())

    return y_pred_proba

def top_k_lift(y_true, y_prob, k=0.2):
    df = pd.DataFrame({'y': y_true, 'prob': y_prob})
    df = df.sort_values('prob', ascending=False)

    top_k = df.head(int(len(df) * k))

    return top_k['y'].mean()


# =========================
# 8. 可视化
# =========================
def plot_training(model):
    results = model.evals_result()

    plt.figure()
    plt.plot(results['validation_0']['logloss'], label='Train')
    plt.plot(results['validation_1']['logloss'], label='Test')
    plt.legend()
    plt.title('Training LogLoss')
    plt.savefig('training_curve.png')
    plt.close()


def plot_feature_importance(model):
    plt.figure(figsize=(10, 8))
    plot_importance(model, max_num_features=20)
    plt.title('Feature Importance')
    plt.savefig('feature_importance.png')
    plt.close()


# =========================
# 9. 主函数
# =========================
def main():
    df = load_data()

    df = preprocess_data(df)

    df = feature_engineering(df)

    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train, X_test, y_test)

    y_pred_prob = evaluate_model(model, X_test, y_test)

    plot_training(model)
    plot_feature_importance(model)

    print("预测概率示例：")
    print(y_pred_prob[:10])


# =========================
# 10. 执行
# =========================
if __name__ == "__main__":
    main()