import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier, plot_importance
import warnings

warnings.filterwarnings('ignore')

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
    '品牌', '车型名称','被保险人年龄'
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
# 3. 预处理
# =========================
def preprocess_data(df):
    print("原始数据量:", df.shape)

    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]]
    df = df.dropna()

    print("删除缺失后数据量:", df.shape)

    df['label'] = (df[TARGET_COLUMN] > 0).astype(int)
    return df


# =========================
# 4. 特征工程
# =========================
def feature_engineering(df):
    df = df.copy()

    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # 组合特征
    df['费率比'] = df['保费不含税'] / (df['标准保费'] + 1e-6)
    df['折旧率'] = df['车辆实际价值'] / (df['本地车型库中的新车购置价'] + 1e-6)
    df['风险比'] = df['纯风险保费'] / (df['保费不含税'] + 1e-6)

    return df


# =========================
# 5. 数据划分
# =========================
def split_data(df):
    X = df.drop(columns=[TARGET_COLUMN, 'label'])
    y = df['label']
    return train_test_split(X, y, test_size=0.2, random_state=42)


# =========================
# 6. Optuna目标函数
# =========================
def objective(trial, X_train, X_test, y_train, y_test):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "eval_metric": "logloss",
        "use_label_encoder": False
    }

    model = XGBClassifier(**params)

    model.fit(X_train, y_train)

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)

    return auc


# =========================
# 7. 自动调参
# =========================
def tune_hyperparameters(X_train, X_test, y_train, y_test):

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, X_test, y_train, y_test),
        n_trials=30
    )

    print("最佳参数:", study.best_params)
    print("最佳AUC:", study.best_value)

    return study.best_params


# =========================
# 8. 用最佳参数训练最终模型
# =========================
def train_best_model(best_params, X_train, y_train):

    best_params["eval_metric"] = "logloss"
    best_params["use_label_encoder"] = False

    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    return model


# =========================
# 9. 评估
# =========================
def evaluate_model(model, X_test, y_test):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("AUC:", roc_auc_score(y_test, y_pred_prob))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    return y_pred_prob


# =========================
# 10. 可视化
# =========================
def plot_feature_importance(model):
    plt.figure(figsize=(10, 8))
    plot_importance(model, max_num_features=20)
    plt.savefig("feature_importance.png")
    plt.close()


# =========================
# 11. 主函数
# =========================
def main():

    df = load_data()
    df = preprocess_data(df)
    df = feature_engineering(df)

    X_train, X_test, y_train, y_test = split_data(df)

    best_params = tune_hyperparameters(X_train, X_test, y_train, y_test)

    model = train_best_model(best_params, X_train, y_train)

    evaluate_model(model, X_test, y_test)

    plot_feature_importance(model)


if __name__ == "__main__":
    main()