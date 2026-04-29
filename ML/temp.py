from sklearn.model_selection import GridSearchCV, StratifiedKFold
import time
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split


def hyperparameter_tuning(X, y, scale_pos_weight, cv_folds=5, n_jobs=-1):
    """
    使用网格搜索和交叉验证寻找最优的XGBoost参数

    参数:
    X: 特征数据
    y: 目标变量
    scale_pos_weight: 类别权重
    cv_folds: 交叉验证折数
    n_jobs: 并行作业数

    返回:
    best_model: 最优模型
    best_params: 最优参数
    cv_results: 交叉验证结果
    """
    print("\n" + "=" * 50)
    print("步骤: 网格搜索和交叉验证调参")
    print("=" * 50)

    # 创建基础模型
    base_model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=n_jobs
    )

    # 定义参数网格
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [7, 9],
        'learning_rate': [0.01, 0.05,0.1],
    }

    print(f"参数网格大小: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate'])} 种组合")
    print(f"交叉验证折数: {cv_folds}")
    print(f"总训练次数: {cv_folds * len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate']) }")

    # 创建交叉验证策略（分层K折，保持类别比例）
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # 创建网格搜索对象
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='roc_auc',  # 使用AUC作为评估指标
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True
    )

    print("\n开始网格搜索...")
    start_time = time.time()

    # 执行网格搜索
    grid_search.fit(X, y)

    end_time = time.time()
    print(f"网格搜索完成，耗时: {end_time - start_time:.2f} 秒")

    # 获取最优结果
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"\n最优参数: {best_params}")
    print(f"最优交叉验证AUC分数: {best_score:.4f}")

    # 打印所有参数组合的结果
    print("\n所有参数组合的交叉验证结果（按AUC排序前10）:")
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('mean_test_score', ascending=False)

    # 显示前10个最佳参数组合
    top_results = results_df.head(10)[[
        'mean_test_score', 'std_test_score',
        'param_n_estimators', 'param_max_depth', 'param_learning_rate',
    ]]

    print(top_results.to_string(index=False))

    # 可视化调参结果
    # visualize_grid_search_results(grid_search)

    return best_model, best_params, grid_search.cv_results_

def train_optimized_model(X, y, scale_pos_weight, best_params):
    """
    使用最优参数训练模型
    """
    print("\n" + "=" * 50)
    print("步骤: 使用最优参数训练模型")
    print("=" * 50)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 使用最优参数创建模型
    optimized_model = xgb.XGBClassifier(
        n_estimators=best_params.get('n_estimators', 100),
        max_depth=best_params.get('max_depth', 6),
        learning_rate=best_params.get('learning_rate', 0.1),
        subsample=best_params.get('subsample', 0.8),
        colsample_bytree=best_params.get('colsample_bytree', 0.8),
        gamma=best_params.get('gamma', 0),
        reg_alpha=best_params.get('reg_alpha', 0),
        reg_lambda=best_params.get('reg_lambda', 1),
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )

    print("最优参数模型配置:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # 训练模型
    optimized_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )

    return optimized_model, X_train, X_test, y_train, y_test
