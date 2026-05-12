import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import optuna

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

# =====================================================
# 中文显示
# =====================================================
plt.rcParams['font.sans-serif'] = [
    'SimHei',
    'Arial Unicode MS',
    'DejaVu Sans'
]

plt.rcParams['axes.unicode_minus'] = False


# =====================================================
# 特征配置
# =====================================================
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


# =====================================================
# 数据加载
# =====================================================
def load_data():

    df1 = pd.read_csv(
        r'D:\车险智能报价\训练数据\报价单去重晚.csv'
    )

    df2 = pd.read_csv(
        r'D:\车险智能报价\训练数据\报价单去重早.csv'
    )

    df = pd.concat(
        [df1, df2],
        ignore_index=True
    )

    print("=" * 60)
    print("原始数据量")
    print("=" * 60)

    print(df.shape)

    return df


# =====================================================
# 数据预处理
# =====================================================
def preprocess_data(df):

    df = df[
        FEATURE_COLUMNS + [TARGET_COLUMN]
    ].copy()

    # 标签
    df['label'] = (
        df[TARGET_COLUMN] > 0
    ).astype(int)

    print("\n目标变量分布:")
    print(df['label'].value_counts())

    print(
        f"\n整体成交率:"
        f" {df['label'].mean():.2%}"
    )

    # 删除缺失值
    df = df.dropna()

    print("\n删除缺失值后数据量:")
    print(df.shape)

    return df


# =====================================================
# 特征工程
# =====================================================
def feature_engineering(df):

    df = df.copy()

    # =================================================
    # categorical 类型
    # =================================================
    for col in CATEGORICAL_COLUMNS:

        df[col] = df[col].astype('category')

    # =================================================
    # 衍生特征
    # =================================================

    # 费率比
    df['费率比'] = (
        df['保费不含税']
        / (df['标准保费'] + 1e-6)
    )

    # 折旧率
    df['车辆折旧率'] = (
        df['车辆实际价值']
        / (
            df['本地车型库中的新车购置价']
            + 1e-6
        )
    )

    # 风险保费占比
    df['风险保费比'] = (
        df['纯风险保费']
        / (df['保费不含税'] + 1e-6)
    )

    # 连续承保与出险
    df['连续承保出险比'] = (
        df['连续承保年数']
        / (df['出险次数'] + 1)
    )

    # 年龄车辆联合风险
    df['年龄车辆联合风险'] = (
        df['被保险人年龄']
        * df['使用年限']
    )

    return df


# =====================================================
# 数据切分
# =====================================================
def split_data(df):

    X = df.drop(
        columns=[TARGET_COLUMN, 'label']
    )

    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42,

        stratify=y
    )

    return X_train, X_test, y_train, y_test


# =====================================================
# TopK命中率
# =====================================================
def top_k_lift(
        y_true,
        y_prob,
        k=0.2
):

    temp = pd.DataFrame({

        'y': y_true,
        'prob': y_prob

    })

    temp = temp.sort_values(

        by='prob',
        ascending=False

    )

    top_k = temp.head(

        int(len(temp) * k)

    )

    return top_k['y'].mean()


# =====================================================
# 最佳阈值
# =====================================================
def find_best_threshold(
        y_true,
        y_prob
):

    precisions, recalls, thresholds = (
        precision_recall_curve(
            y_true,
            y_prob
        )
    )

    f1_scores = (

        2 * precisions * recalls

        / (precisions + recalls + 1e-6)

    )

    best_idx = np.argmax(f1_scores)

    best_threshold = thresholds[best_idx]

    print("\n最佳阈值:")
    print(best_threshold)

    print("\n最佳F1:")
    print(f1_scores[best_idx])

    return best_threshold


# =====================================================
# Optuna 自动调参
# =====================================================
def auto_tune_xgb(
        X_train,
        y_train,
        X_valid,
        y_valid
):

    print("\n开始Optuna自动调参...")

    # =================================================
    # 类别不平衡
    # =================================================
    neg = (y_train == 0).sum()

    pos = (y_train == 1).sum()

    base_scale_pos_weight = neg / pos

    print(
        f"\n基础scale_pos_weight:"
        f" {base_scale_pos_weight:.2f}"
    )

    # =================================================
    # 目标函数
    # =================================================
    def objective(trial):

        params = {

            # =========================================
            # 固定树数量
            # =========================================
            'n_estimators': 5000,

            # =========================================
            # 树深
            # =========================================
            'max_depth':

                trial.suggest_int(
                    'max_depth',
                    3,
                    8
                ),

            # =========================================
            # 学习率
            # =========================================
            'learning_rate':

                trial.suggest_float(
                    'learning_rate',
                    0.01,
                    0.08,
                    log=True
                ),

            # =========================================
            # 行采样
            # =========================================
            'subsample':

                trial.suggest_float(
                    'subsample',
                    0.6,
                    1.0
                ),

            # =========================================
            # 列采样
            # =========================================
            'colsample_bytree':

                trial.suggest_float(
                    'colsample_bytree',
                    0.6,
                    1.0
                ),

            # =========================================
            # 最小叶子样本
            # =========================================
            'min_child_weight':

                trial.suggest_int(
                    'min_child_weight',
                    1,
                    12
                ),

            # =========================================
            # gamma
            # =========================================
            'gamma':

                trial.suggest_float(
                    'gamma',
                    0,
                    5
                ),

            # =========================================
            # L1正则
            # =========================================
            'reg_alpha':

                trial.suggest_float(
                    'reg_alpha',
                    0,
                    10
                ),

            # =========================================
            # L2正则
            # =========================================
            'reg_lambda':

                trial.suggest_float(
                    'reg_lambda',
                    1,
                    10
                ),

            # =========================================
            # 类别不平衡
            # =========================================
            'scale_pos_weight':

                trial.suggest_float(
                    'scale_pos_weight',
                    base_scale_pos_weight * 0.7,
                    base_scale_pos_weight * 1.3
                ),

            # =========================================
            # 固定参数
            # =========================================
            'objective': 'binary:logistic',

            'eval_metric': 'auc',

            'enable_categorical': True,

            'tree_method': 'hist',

            'random_state': 42,

            # =========================================
            # 提前停止
            # =========================================
            'early_stopping_rounds': 50
        }

        # =============================================
        # 建模
        # =============================================
        model = XGBClassifier(**params)

        model.fit(

            X_train,
            y_train,

            eval_set=[

                (X_valid, y_valid)

            ],

            verbose=False
        )

        # =============================================
        # 概率预测
        # =============================================
        y_prob = model.predict_proba(
            X_valid
        )[:, 1]

        # =============================================
        # AUC
        # =============================================
        auc = roc_auc_score(
            y_valid,
            y_prob
        )

        return auc

    # =================================================
    # 创建study
    # =================================================
    study = optuna.create_study(

        direction='maximize'

    )

    # =================================================
    # 开始调参
    # =================================================
    study.optimize(

        objective,

        n_trials=100

    )

    # =================================================
    # 输出最佳参数
    # =================================================
    print("\n" + "=" * 60)
    print("最佳参数")
    print("=" * 60)

    for k, v in study.best_params.items():

        print(f"{k}: {v}")

    print("\n最佳AUC:")
    print(study.best_value)

    return study.best_params


# =====================================================
# 最终模型训练
# =====================================================
def train_final_model(

        best_params,

        X_train,
        y_train,

        X_test,
        y_test
):

    print("\n开始训练最终模型...")

    best_params.update({

        'n_estimators': 5000,

        'objective': 'binary:logistic',

        'eval_metric': 'auc',

        'enable_categorical': True,

        'tree_method': 'hist',

        'random_state': 42,

        'early_stopping_rounds': 50
    })

    model = XGBClassifier(**best_params)

    model.fit(

        X_train,
        y_train,

        eval_set=[

            (X_train, y_train),
            (X_test, y_test)

        ],

        verbose=50
    )

    return model


# =====================================================
# 模型评估
# =====================================================
def evaluate_model(

        model,

        X_test,
        y_test
):

    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)

    # =================================================
    # 概率预测
    # =================================================
    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    # =================================================
    # 最佳阈值
    # =================================================
    best_threshold = find_best_threshold(

        y_test,
        y_prob
    )

    # =================================================
    # 分类
    # =================================================
    y_pred = (

        y_prob > best_threshold

    ).astype(int)

    # =================================================
    # 指标
    # =================================================
    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    # =================================================
    # 输出
    # =================================================
    print("\n最终评估结果")

    print(
        f"\nPrecision:"
        f" {precision:.4f}"
    )

    print(
        f"Recall:"
        f" {recall:.4f}"
    )

    print(
        f"F1-Score:"
        f" {f1:.4f}"
    )

    print(
        f"AUC-ROC:"
        f" {auc:.4f}"
    )

    # =================================================
    # 分类报告
    # =================================================
    print("\n分类报告")

    print(

        classification_report(

            y_test,
            y_pred,

            target_names=[
                '未成交',
                '成交'
            ]
        )
    )

    # =================================================
    # 混淆矩阵
    # =================================================
    print("\n混淆矩阵")

    print(

        confusion_matrix(
            y_test,
            y_pred
        )
    )

    # =================================================
    # 业务指标
    # =================================================
    print("\n业务指标")

    for k in [

        0.05,
        0.1,
        0.2,
        0.3

    ]:

        lift = top_k_lift(

            y_test,
            y_prob,
            k
        )

        print(

            f"Top {int(k*100)}% 转化率:"
            f" {lift:.4f}"
        )

    print(
        f"\n整体转化率:"
        f" {y_test.mean():.4f}"
    )

    return y_prob


# =====================================================
# 训练曲线
# =====================================================
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

    plt.title('Training AUC Curve')

    plt.savefig(
        'training_auc_curve.png'
    )

    plt.close()

    print("\n训练曲线已保存")


# =====================================================
# 特征重要性
# =====================================================
def plot_feature_importance_chart(model):

    plt.figure(figsize=(12, 10))

    plot_importance(

        model,

        max_num_features=30,

        importance_type='gain'
    )

    plt.title('Feature Importance')

    plt.savefig(
        'feature_importance.png'
    )

    plt.close()

    print("特征重要性图已保存")


# =====================================================
# ROC曲线
# =====================================================
def plot_roc(
        y_test,
        y_prob
):

    fpr, tpr, _ = roc_curve(

        y_test,
        y_prob
    )

    plt.figure(figsize=(8, 6))

    plt.plot(fpr, tpr)

    plt.xlabel('FPR')

    plt.ylabel('TPR')

    plt.title('ROC Curve')

    plt.savefig(
        'roc_curve.png'
    )

    plt.close()

    print("ROC曲线已保存")


# =====================================================
# 主函数
# =====================================================
def main():

    print("=" * 60)
    print("车险成交预测模型")
    print("=" * 60)

    # =================================================
    # 读取数据
    # =================================================
    df = load_data()

    # =================================================
    # 数据预处理
    # =================================================
    df = preprocess_data(df)

    # =================================================
    # 特征工程
    # =================================================
    df = feature_engineering(df)

    # =================================================
    # 数据切分
    # =================================================
    X_train, X_test, y_train, y_test = split_data(df)

    print("\n训练集大小:")
    print(X_train.shape)

    print("\n测试集大小:")
    print(X_test.shape)

    # =================================================
    # 自动调参
    # =================================================
    best_params = auto_tune_xgb(

        X_train,
        y_train,

        X_test,
        y_test
    )

    # =================================================
    # 最终模型
    # =================================================
    model = train_final_model(

        best_params,

        X_train,
        y_train,

        X_test,
        y_test
    )

    # =================================================
    # 模型评估
    # =================================================
    y_prob = evaluate_model(

        model,

        X_test,
        y_test
    )

    # =================================================
    # 可视化
    # =================================================
    plot_training_curve(model)

    plot_feature_importance_chart(model)

    plot_roc(
        y_test,
        y_prob
    )

    print("\n预测概率示例:")

    print(y_prob[:20])


# =====================================================
# 执行
# =====================================================
if __name__ == "__main__":

    main()