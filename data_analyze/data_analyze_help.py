import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class InsuranceFeatureAnalysis:
    def __init__(self):
        self.df = None
        self.feature_fields = [
            '自主定价系数', '手续费不含税金额', '众享分_商交关联', '众享分',
            '二级能源种类名称', '新续转标识名称', '无赔款优待系数', '自主定价系数均值',
            '连续承保年数', '纯风险保费', '使用年限', '尊享分_商交关联',
            '保费不含税', '车损险限额', '车辆实际价值', '被保险人年龄',
            '被保险人性别', '本地车型库中的新车购置价', '标准保费', '二级机构名称',
            '三级机构名称', '四级机构名称', '品牌', '车型名称', '出险次数', '二手车标志'
        ]
        self.categorical_features = [
            '二级能源种类名称', '新续转标识名称', '二级机构名称',
            '三级机构名称', '四级机构名称', '品牌', '车型名称'
        ]
        self.label_encoders = {}

    def load_and_clean_data(self):
        """加载数据并处理缺失值"""
        print("=== 步骤1: 加载数据 ===")
        # 加载数据
        # df1 = pd.read_csv('D:\车险智能报价\车险报价2025成交.csv')
        # df2 = pd.read_csv('D:\车险智能报价\车险报价20251001至今未成交.csv')
        # 合并
        # combined = pd.concat([df1, df2], ignore_index=True)
        combined = pd.read_csv('D:\车险智能报价\去重数据.csv')
        # 随机打乱
        self.df = combined.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"原始数据量: {len(self.df)}")

        # 检查目标变量是否存在
        if 'target' not in self.df.columns:
            raise ValueError("数据中缺少目标变量'target'")

        # 创建目标变量
        self.df['是否成交'] = (self.df['target'] > 0).astype(int)

        # 只保留需要的特征字段和目标变量
        available_features = [col for col in self.feature_fields if col in self.df.columns]
        required_columns = available_features + ['target', '是否成交']
        self.df = self.df[required_columns]

        print(f"可用特征字段数量: {len(available_features)}")
        print(f"缺失的特征字段: {set(self.feature_fields) - set(available_features)}")

        # 处理缺失值
        print("\n=== 步骤2: 处理缺失值 ===")
        before_drop = len(self.df)
        self.df = self.df.dropna()
        after_drop = len(self.df)

        print(f"删除前数据量: {before_drop}")
        print(f"删除后数据量: {after_drop}")
        print(f"删除缺失值数量: {before_drop - after_drop}")
        print(f"删除比例: {(before_drop - after_drop) / before_drop:.2%}")

        return self.df

    def exploratory_analysis(self):
        """探索性数据分析"""
        print("\n=== 步骤3: 探索性数据分析 ===")

        # 基本统计信息
        print("数据基本信息:")
        print(f"数据形状: {self.df.shape}")
        print(f"成交样本数: {self.df['是否成交'].sum()}")
        print(f"未成交样本数: {(self.df['是否成交'] == 0).sum()}")
        print(f"成交率: {self.df['是否成交'].mean():.2%}")

        # 数值型特征分布分析
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [col for col in numeric_features if col not in ['target', '是否成交']]

        self._plot_numeric_distributions(numeric_features)
        self._plot_target_distribution()
        self._plot_correlation_matrix(numeric_features)

    def _plot_numeric_distributions(self, numeric_features):
        """绘制数值型特征分布图"""
        n_features = len(numeric_features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        for i, feature in enumerate(numeric_features):
            if i < len(axes):
                # 按是否成交分组绘制分布
                sns.histplot(data=self.df, x=feature, hue='是否成交',
                             bins=30, alpha=0.7, ax=axes[i])
                axes[i].set_title(f'{feature}分布')
                axes[i].legend(['未成交', '成交'])

        # 隐藏多余的子图
        for i in range(len(numeric_features), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig('numeric_features_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_target_distribution(self):
        """绘制目标变量分布"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 饼图
        target_counts = self.df['是否成交'].value_counts()
        labels = ['未成交', '成交']
        colors = ['lightcoral', 'lightblue']
        ax1.pie(target_counts.values, labels=labels, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax1.set_title('报价单成交情况分布')

        # 柱状图
        sns.countplot(data=self.df, x='是否成交', palette=colors, ax=ax2)
        ax2.set_title('报价单成交情况计数')
        ax2.set_xlabel('是否成交 (0=未成交, 1=成交)')
        ax2.set_ylabel('数量')

        # 在柱状图上添加数值标签
        for p in ax2.patches:
            height = p.get_height()
            ax2.text(p.get_x() + p.get_width() / 2., height + height * 0.01,
                     f'{int(height)}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_correlation_matrix(self, numeric_features):
        """绘制相关性矩阵热力图"""
        plt.figure(figsize=(12, 10))

        # 计算相关性矩阵
        corr_data = self.df[numeric_features + ['是否成交']].corr()

        # 绘制热力图
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        sns.heatmap(corr_data, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('特征与目标变量相关性热力图')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 打印与成交率相关性最高的特征
        target_corr = corr_data['是否成交'].abs().sort_values(ascending=False)
        print("与成交率相关性最高的特征:")
        print(target_corr.head(10))

    def categorical_feature_analysis(self):
        """分类特征分析"""
        print("\n=== 步骤4: 分类特征分析 ===")

        # 分析每个分类特征与成交率的关系
        for feature in self.categorical_features:
            if feature in self.df.columns:
                self._analyze_categorical_feature(feature)

    def _analyze_categorical_feature(self, feature):
        """分析单个分类特征"""
        # 计算各分类的成交率
        feature_stats = self.df.groupby(feature)['是否成交'].agg(['count', 'sum', 'mean']).reset_index()
        feature_stats.columns = [feature, '总数', '成交数', '成交率']
        feature_stats = feature_stats.sort_values('成交率', ascending=False)

        print(f"\n{feature} 分析结果:")
        print(feature_stats.head())

        # 绘制条形图
        plt.figure(figsize=(12, 6))

        # 如果类别太多，只显示前15个
        if len(feature_stats) > 15:
            feature_stats = feature_stats.head(15)
            title_suffix = " (Top 15)"
        else:
            title_suffix = ""

        sns.barplot(data=feature_stats, x=feature, y='成交率', palette='viridis')
        plt.title(f'{feature}各类别成交率{title_suffix}')
        plt.xlabel(feature)
        plt.ylabel('成交率')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'categorical_analysis_{feature}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def feature_engineering(self):
        """特征工程处理"""
        print("\n=== 步骤5: 特征工程 ===")

        # 复制数据用于特征工程
        df_engineered = self.df.copy()

        # 1. 编码分类特征
        print("编码分类特征...")
        for feature in self.categorical_features:
            if feature in df_engineered.columns:
                le = LabelEncoder()
                df_engineered[f'{feature}_encoded'] = le.fit_transform(df_engineered[feature].astype(str))
                self.label_encoders[feature] = le
                print(f"已编码: {feature} -> {feature}_encoded")

        # 2. 创建新特征
        print("创建新特征...")

        # 保费相关比率特征
        if '保费不含税' in df_engineered.columns and '纯风险保费' in df_engineered.columns:
            df_engineered['保费风险比'] = df_engineered['保费不含税'] / (df_engineered['纯风险保费'] + 1e-8)

        if '车损险限额' in df_engineered.columns and '车辆实际价值' in df_engineered.columns:
            df_engineered['保障充足度'] = df_engineered['车损险限额'] / (df_engineered['车辆实际价值'] + 1e-8)

        # 客户价值特征
        if '众享分' in df_engineered.columns and '尊享分_商交关联' in df_engineered.columns:
            df_engineered['综合评分'] = (df_engineered['众享分'] + df_engineered['尊享分_商交关联']) / 2

        # 车辆使用特征
        if '使用年限' in df_engineered.columns and '连续承保年数' in df_engineered.columns:
            df_engineered['车龄承保比'] = df_engineered['使用年限'] / (df_engineered['连续承保年数'] + 1e-8)

        # 出险风险特征
        if '出险次数' in df_engineered.columns and '连续承保年数' in df_engineered.columns:
            df_engineered['年均出险次数'] = df_engineered['出险次数'] / (df_engineered['连续承保年数'] + 1e-8)

        # 3. 标准化数值特征
        numeric_features_for_scaling = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features_for_scaling = [col for col in numeric_features_for_scaling
                                        if col not in ['target', '是否成交']]

        scaler = StandardScaler()
        df_engineered[numeric_features_for_scaling] = scaler.fit_transform(
            df_engineered[numeric_features_for_scaling])

        print(f"特征工程完成，最终特征数量: {len(df_engineered.columns)}")

        # 保存特征工程后的数据统计
        self._plot_feature_importance_analysis(df_engineered)

        return df_engineered

    def _plot_feature_importance_analysis(self, df_engineered):
        """绘制特征重要性分析"""
        # 选择数值特征进行重要性分析（基于与目标变量的相关性）
        numeric_features = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [col for col in numeric_features
                            if col not in ['target', '是否成交'] and not col.endswith('_encoded')]

        correlations = df_engineered[numeric_features + ['是否成交']].corr()['是否成交'].abs().sort_values(
            ascending=False)

        plt.figure(figsize=(12, 8))
        correlations.head(15).plot(kind='barh')
        plt.title('特征与目标变量相关性排名 (Top 15)')
        plt.xlabel('绝对相关系数')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("特征重要性排名 (与目标变量的相关性):")
        print(correlations.head(10))

    def prepare_model_data(self, df_engineered):
        """准备模型训练数据"""
        print("\n=== 步骤6: 准备模型数据 ===")

        # 选择特征列（排除原始分类特征和ID类特征）
        exclude_features = ['target', '是否成交'] + self.categorical_features
        feature_columns = [col for col in df_engineered.columns if col not in exclude_features]

        X = df_engineered[feature_columns]
        y = df_engineered['是否成交']

        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        print(f"训练集大小: {X_train.shape}")
        print(f"测试集大小: {X_test.shape}")
        print(f"特征数量: {X_train.shape[1]}")

        return X_train, X_test, y_train, y_test, feature_columns

    def run_complete_analysis(self):
        """运行完整分析流程"""
        try:
            # 执行所有步骤
            self.load_and_clean_data()
            self.exploratory_analysis()
            self.categorical_feature_analysis()
            df_engineered = self.feature_engineering()
            X_train, X_test, y_train, y_test, feature_columns = self.prepare_model_data(df_engineered)

            print("\n=== 分析完成 ===")
            print(f"最终可用于建模的特征数量: {len(feature_columns)}")
            print("所有图表已保存到当前目录")

            return {
                'original_df': self.df,
                'engineered_df': df_engineered,
                'train_data': (X_train, X_test, y_train, y_test),
                'feature_columns': feature_columns,
                'label_encoders': self.label_encoders
            }

        except Exception as e:
            print(f"分析过程中出现错误: {str(e)}")
            return None


def main():
    """主函数"""
    print("开始车险报价成交预测的特征工程和数据分析...")

    # 创建分析对象
    analyzer = InsuranceFeatureAnalysis()

    # 运行完整分析
    results = analyzer.run_complete_analysis()

    if results is not None:
        print("\n=== 结果摘要 ===")
        original_df = results['original_df']
        engineered_df = results['engineered_df']

        print(f"原始数据样本数: {len(original_df)}")
        print(f"特征工程后样本数: {len(engineered_df)}")
        print(f"最终特征数量: {len(results['feature_columns'])}")
        print(f"训练集成交率: {results['train_data'][2].mean():.2%}")
        print(f"测试集成交率: {results['train_data'][3].mean():.2%}")

        # 保存处理后的数据
        engineered_df.to_csv('engineered_insurance_data.csv', index=False)
        print("特征工程后的数据已保存为 'engineered_insurance_data.csv'")
    else:
        print("分析失败，请检查数据和代码")


if __name__ == "__main__":
    main()
