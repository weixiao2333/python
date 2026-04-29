import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from torchviz import make_dot
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)


class EnhancedInsurancePredictionModel:
    def __init__(self, premium_weight=3.0):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.premium_weight = premium_weight  # "保费不含税"特征权重
        self.feature_names = None
        print(f"使用设备: {self.device}")
        print(f"'保费不含税'特征增强权重: {premium_weight}")

    def load_and_preprocess_data(self, file_path):
        """加载和预处理数据"""
        print("=== 步骤1: 加载数据 ===")
        # 加载数据
        df1 = pd.read_csv('D:\车险智能报价\车险报价2025成交.csv')
        df2 = pd.read_csv('D:\车险智能报价\车险报价20251001至今未成交.csv')

        # 合并
        combined = pd.concat([df1, df2], ignore_index=True)

        # 随机打乱
        df = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"原始数据量: {len(df)}")

        # 处理缺省值 - 直接删除
        df_clean = df.dropna()
        print(f"删除缺省值后数据量: {len(df_clean)}")
        print(f"删除数据量: {len(df) - len(df_clean)}")

        return df_clean

    def prepare_features(self, df):
        """准备特征数据 - 增强'保费不含税'特征重要性"""
        print("\n=== 步骤2: 准备特征数据（增强'保费不含税'特征） ===")

        # 所有特征字段
        all_features = [
            '自主定价系数', '手续费不含税金额', '众享分_商交关联', '众享分',
            '二级能源种类名称', '新续转标识名称', '无赔款优待系数', '自主定价系数均值',
            '连续承保年数', '纯风险保费', '使用年限', '尊享分_商交关联',
            '保费不含税', '车损险限额', '车辆实际价值', '被保险人年龄',
            '被保险人性别', '本地车型库中的新车购置价', '标准保费',
            '二级机构名称', '三级机构名称', '四级机构名称', '品牌',
            '车型名称', '出险次数', '二手车标志'
        ]

        # 需要编码的分类特征
        categorical_features = [
            '二级能源种类名称', '新续转标识名称', '二级机构名称',
            '三级机构名称', '四级机构名称', '品牌', '车型名称'
        ]

        # 检查数据中是否存在这些特征
        available_features = [f for f in all_features if f in df.columns]
        available_categorical = [f for f in categorical_features if f in df.columns]

        print(f"可用特征数量: {len(available_features)}")
        print(f"需要编码的分类特征数量: {len(available_categorical)}")

        # 提取特征和目标变量
        X = df[available_features].copy()
        y = (df['保单件数'] > 0).astype(int)  # 转换为二分类标签

        # 标记"保费不含税"特征的位置
        self.premium_feature_idx = None
        if '保费不含税' in X.columns:
            self.premium_feature_idx = list(X.columns).index('保费不含税')
            print(f"'保费不含税'特征位置: {self.premium_feature_idx}")

        # 编码分类特征
        print("编码分类特征...")
        for feature in available_categorical:
            le = LabelEncoder()
            X[feature] = le.fit_transform(X[feature].astype(str))
            self.label_encoders[feature] = le
            print(f"  {feature}: {len(le.classes_)} 个类别")

        # 数值特征标准化
        numerical_features = [f for f in available_features if f not in available_categorical]
        if numerical_features:
            X[numerical_features] = self.scaler.fit_transform(X[numerical_features])
            print(f"标准化了 {len(numerical_features)} 个数值特征")

            # 特别增强"保费不含税"特征
            if '保费不含税' in X.columns:
                print(f"增强'保费不含税'特征重要性 (权重: {self.premium_weight})")
                # 方法1: 特征缩放增强
                X['保费不含税'] = X['保费不含税'] * self.premium_weight

                # 方法2: 特征复制（添加多个副本）
                X['保费不含税_copy1'] = X['保费不含税'].copy()
                X['保费不含税_copy2'] = X['保费不含税'].copy()

                # 更新特征列表
                original_features = list(X.columns[:-2])  # 排除刚添加的副本
                enhanced_features = original_features + ['保费不含税_copy1', '保费不含税_copy2']
                X_enhanced = X[enhanced_features]

                print(f"特征增强完成: 原始特征 {len(original_features)}, 增强后特征 {len(enhanced_features)}")

                self.feature_names = enhanced_features
                return X_enhanced, y, enhanced_features

        self.feature_names = list(X.columns)
        return X, y, available_features

    def create_enhanced_model(self, input_size):
        """创建增强的神经网络模型 - 针对'保费不含税'特征优化"""
        print("\n=== 步骤3: 创建增强神经网络模型 ===")

        class EnhancedInsuranceNet(nn.Module):
            def __init__(self, input_size):
                super(EnhancedInsuranceNet, self).__init__()
                # 增加网络容量以更好地处理重要特征
                self.layer1 = nn.Linear(input_size, 256)  # 增加神经元数量
                self.bn1 = nn.BatchNorm1d(256)  # 批归一化
                self.layer2 = nn.Linear(256, 128)
                self.bn2 = nn.BatchNorm1d(128)
                self.layer3 = nn.Linear(128, 64)  # 增加一个隐藏层
                self.layer4 = nn.Linear(64, 32)
                self.output_layer = nn.Linear(32, 1)

                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.4)  # 调整dropout
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.relu(self.bn1(self.layer1(x)))
                x = self.dropout(x)
                x = self.relu(self.bn2(self.layer2(x)))
                x = self.dropout(x)
                x = self.relu(self.layer3(x))
                x = self.dropout(x)
                x = self.relu(self.layer4(x))
                x = self.sigmoid(self.output_layer(x))
                return x

        self.model = EnhancedInsuranceNet(input_size)
        self.model.to(self.device)
        print(f"增强模型已创建，输入维度: {input_size}")
        print(f"增强模型结构:\n{self.model}")

        return self.model

    def train_model_with_premium_focus(self, X, y):
        """训练模型 - 重点关注'保费不含税'特征的影响"""
        print("\n=== 步骤4: 训练模型（重点关注保费特征） ===")

        # 转换数据为tensor
        X_tensor = torch.FloatTensor(X.values).to(self.device)
        y_tensor = torch.FloatTensor(y.values).unsqueeze(1).to(self.device)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y
        )

        print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

        # 定义损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)  # 使用AdamW
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)  # 余弦退火学习率

        # 训练参数
        epochs = 200
        train_losses = []
        val_accuracies = []
        premium_importance_scores = []  # 跟踪保费特征重要性

        print("开始训练...")
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())

            # 验证阶段
            if epoch % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_test)
                    val_pred = (val_outputs > 0.5).float()
                    val_accuracy = (val_pred == y_test).float().mean().item()
                    val_accuracies.append(val_accuracy)

                    # 分析保费特征重要性
                    if epoch % 50 == 0 and hasattr(self, 'premium_feature_idx'):
                        importance = self.calculate_feature_importance(X_train, y_train, epoch)
                        premium_importance_scores.append(importance)

            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}')

        return train_losses, val_accuracies, X_test, y_test, premium_importance_scores

    def calculate_feature_importance(self, X, y, epoch):
        """计算特征重要性（简化版）"""
        try:
            # 使用梯度信息来估计特征重要性
            self.model.eval()
            X.requires_grad_(True)

            outputs = self.model(X[:100])  # 使用小批量计算
            loss = nn.BCELoss()(outputs, y[:100])
            loss.backward()

            # 获取梯度范数作为重要性指标
            gradients = X.grad.abs().mean(dim=0)
            max_grad_idx = gradients.argmax().item()

            X.requires_grad_(False)

            # 检查保费特征是否在最重要的特征中
            if hasattr(self, 'premium_feature_idx'):
                premium_positions = [self.premium_feature_idx,
                                     len(self.feature_names) - 2,  # copy1
                                     len(self.feature_names) - 1]  # copy2

                is_premium_important = max_grad_idx in premium_positions
                if is_premium_important:
                    print(f"  ✓ Epoch {epoch}: '保费不含税'特征被识别为重要特征 (位置: {max_grad_idx})")
                else:
                    print(
                        f"  ✗ Epoch {epoch}: 最重要特征是位置 {max_grad_idx}, '保费不含税'特征位置: {premium_positions}")

            return gradients.mean().item()

        except Exception as e:
            print(f"特征重要性计算失败: {e}")
            return 0.0

    def evaluate_model(self, X_test, y_test):
        """评估模型"""
        print("\n=== 步骤5: 评估模型 ===")

        self.model.eval()
        with torch.no_grad():
            X_test_tensor = X_test.to(self.device)
            y_test_tensor = y_test.to(self.device)

            # 预测概率
            y_pred_proba = self.model(X_test_tensor)
            y_pred = (y_pred_proba > 0.5).float()

            # 转换为numpy数组用于计算指标
            y_true = y_test_tensor.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()
            y_pred_proba_np = y_pred_proba.cpu().numpy()

            # 计算各项指标
            accuracy = accuracy_score(y_true, y_pred_np)
            precision = precision_score(y_true, y_pred_np, zero_division=0)
            recall = recall_score(y_true, y_pred_np, zero_division=0)
            f1 = f1_score(y_true, y_pred_np, zero_division=0)
            auc = roc_auc_score(y_true, y_pred_proba_np)

            print(f"准确率 (Accuracy): {accuracy:.4f}")
            print(f"精确率 (Precision): {precision:.4f}")
            print(f"召回率 (Recall): {recall:.4f}")
            print(f"F1分数 (F1-Score): {f1:.4f}")
            print(f"AUC分数: {auc:.4f}")

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }

    def visualize_training_with_premium_analysis(self, train_losses, val_accuracies, premium_importance_scores):
        """可视化训练过程和保费特征重要性分析"""
        print("\n=== 步骤6: 可视化训练过程和特征重要性 ===")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 绘制损失曲线
        ax1.plot(train_losses, linewidth=2)
        ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)

        # 绘制准确率曲线
        epochs_val = range(0, len(train_losses), 10)
        ax2.plot(epochs_val, val_accuracies, 'o-', color='green', linewidth=2)
        ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, alpha=0.3)

        # 绘制保费特征重要性（如果有）
        if premium_importance_scores:
            epochs_importance = range(0, len(train_losses), 50)
            ax3.plot(epochs_importance, premium_importance_scores, 'ro-', linewidth=2, markersize=6)
            ax3.set_title("Premium Feature Importance Score", fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Importance Score')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Feature Importance\nNot Available',
                     ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title("Premium Feature Importance", fontsize=14, fontweight='bold')

        # 绘制特征权重分布（模拟）
        if hasattr(self, 'feature_names') and len(self.feature_names) > 0:
            # 模拟特征重要性（实际应用中可以通过模型权重分析得到）
            np.random.seed(42)
            feature_importance = np.random.exponential(1, len(self.feature_names))
            feature_importance = feature_importance / feature_importance.sum() * 100

            # 突出显示保费相关特征
            colors = []
            for name in self.feature_names:
                if '保费不含税' in name:
                    colors.append('red')
                else:
                    colors.append('skyblue')

            bars = ax4.bar(range(len(feature_importance)), feature_importance, color=colors)
            ax4.set_title('Simulated Feature Importance Distribution', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Feature Index')
            ax4.set_ylabel('Importance (%)')
            ax4.set_xticks(range(0, len(feature_importance), max(1, len(feature_importance) // 10)))

            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', label="'保费不含税' features"),
                Patch(facecolor='skyblue', label='Other features')
            ]
            ax4.legend(handles=legend_elements)

        plt.tight_layout()
        plt.savefig('enhanced_training_analysis.png', dpi=300, bbox_inches='tight')
        print("增强训练分析图已保存为 'enhanced_training_analysis.png'")
        plt.show()

    def visualize_model(self, X):
        """可视化模型结构"""
        print("\n=== 步骤7: 可视化模型结构 ===")

        try:
            # 创建一个示例输入
            sample_input = torch.randn(1, X.shape[1]).to(self.device)

            # 生成模型图
            dot = make_dot(self.model(sample_input), params=dict(self.model.named_parameters()))
            dot.render('enhanced_model_architecture', format='png', cleanup=True)
            print("增强模型结构图已保存为 'enhanced_model_architecture.png'")

            # 显示模型信息
            print("\n增强模型参数统计:")
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"总参数数量: {total_params:,}")
            print(f"可训练参数数量: {trainable_params:,}")

        except Exception as e:
            print(f"模型可视化失败: {e}")
            print("请确保已安装graphviz: pip install graphviz")

    def save_model(self, filepath='enhanced_insurance_prediction_model.pth'):
        """保存增强模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'device': self.device,
            'premium_weight': self.premium_weight,
            'feature_names': self.feature_names
        }, filepath)
        print(f"\n增强模型已保存为 '{filepath}'")


def main():
    """主函数"""
    print("开始构建增强版车险报价成交概率预测模型...")
    print("重点提升'保费不含税'特征的重要性")

    # 初始化增强模型，设置保费特征权重
    predictor = EnhancedInsurancePredictionModel(premium_weight=3.0)  # 可以调整权重

    try:
        # 步骤1: 加载和预处理数据
        df = predictor.load_and_preprocess_data('your_data.csv')

        # 步骤2: 准备特征数据（增强保费特征）
        X, y, feature_names = predictor.prepare_features(df)
        print(f"\n最终特征数量: {len(feature_names)}")
        print(f"特征列表: {feature_names}")

        # 步骤3: 创建增强模型
        model = predictor.create_enhanced_model(len(feature_names))

        # 步骤4: 训练模型（重点关注保费特征）
        train_losses, val_accuracies, X_test, y_test, premium_importance_scores = predictor.train_model_with_premium_focus(
            X, y)

        # 步骤5: 评估模型
        metrics = predictor.evaluate_model(X_test, y_test)

        # 步骤6: 可视化训练过程和特征重要性
        predictor.visualize_training_with_premium_analysis(train_losses, val_accuracies, premium_importance_scores)

        # 步骤7: 可视化模型结构
        # predictor.visualize_model(X)

        # 保存增强模型
        predictor.save_model()

        print("\n=== 增强模型构建完成 ===")
        print("生成的文件:")
        print("- enhanced_training_analysis.png: 增强训练分析图")
        print("- enhanced_model_architecture.png: 增强模型结构图")
        print("- enhanced_insurance_prediction_model.pth: 增强模型文件")
        print("\n'保费不含税'特征增强策略:")
        print("1. ✓ 特征缩放 (权重倍乘)")
        print("2. ✓ 特征复制 (增加副本)")
        print("3. ✓ 增强网络容量")
        print("4. ✓ 特征重要性监控")

    except FileNotFoundError:
        print("错误: 找不到数据文件 'your_data.csv'，请确保文件路径正确")
    except Exception as e:
        print(f"增强模型构建过程中出现错误: {e}")


if __name__ == "__main__":
    main()
