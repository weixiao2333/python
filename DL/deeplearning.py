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
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)


class InsurancePredictionModel:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

    def load_and_preprocess_data(self, file_path):
        """加载和预处理数据"""
        print("=== 步骤1: 加载数据 ===")
        # 加载数据
        # df1 = pd.read_csv('D:\车险智能报价\车险报价2025成交.csv')
        # df2 = pd.read_csv('D:\车险智能报价\车险报价20251001至今未成交.csv')
        #
        # # 合并
        # combined = pd.concat([df1, df2], ignore_index=True)
        combined = pd.read_csv('D:\车险智能报价\去重数据.csv')

        # 随机打乱
        df = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"原始数据量: {len(df)}")

        # 定义需要编码的特征字段
        categorical_features = [
            '二级能源种类名称', '新续转标识名称', '二级机构名称',
            '三级机构名称', '四级机构名称', '品牌', '车型名称'
        ]

        # 定义所有特征字段
        all_features = [
            '自主定价系数', '手续费不含税金额', '众享分_商交关联', '众享分',
            '二级能源种类名称', '新续转标识名称', '无赔款优待系数', '自主定价系数均值',
            '连续承保年数', '纯风险保费', '使用年限', '尊享分_商交关联',
            '保费不含税', '车损险限额', '车辆实际价值', '被保险人年龄',
            '被保险人性别', '本地车型库中的新车购置价', '标准保费', '二级机构名称',
            '三级机构名称', '四级机构名称', '品牌', '车型名称', '出险次数', '二手车标志'
        ]

        # 检查数据中存在的特征字段
        available_features = [col for col in all_features if col in df.columns]
        print(f"数据中可用的特征字段数量: {len(available_features)}")

        # 选择特征和目标变量
        X = df[available_features].copy()
        y = (df['target'] > 0).astype(int)  # 转换为二分类标签

        # 删除缺失值
        print("=== 步骤2: 处理缺失值 ===")
        original_size = len(X)
        X = X.dropna()
        y = y[X.index]  # 保持y与X的索引一致
        cleaned_size = len(X)

        print(f"删除缺失值前数据量: {original_size}")
        print(f"删除缺失值后数据量: {cleaned_size}")
        print(f"删除的行数: {original_size - cleaned_size}")
        print(f"删除比例: {(original_size - cleaned_size) / original_size * 100:.2f}%")

        return X, y, categorical_features, available_features

    def encode_features(self, X, categorical_features):
        """编码分类特征"""
        print("=== 步骤3: 编码分类特征 ===")
        X_encoded = X.copy()

        for feature in categorical_features:
            if feature in X_encoded.columns:
                le = LabelEncoder()
                # 处理未知类别
                X_encoded[feature] = X_encoded[feature].fillna('Unknown')
                X_encoded[feature] = le.fit_transform(X_encoded[feature])
                self.label_encoders[feature] = le
                print(f"已编码特征: {feature}, 类别数: {len(le.classes_)}")

        return X_encoded

    def prepare_features(self, X_encoded):
        """准备模型输入特征"""
        print("=== 步骤4: 特征标准化 ===")
        # 标准化数值特征
        X_scaled = self.scaler.fit_transform(X_encoded)

        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_scaled)

        print(f"特征维度: {X_tensor.shape}")
        return X_tensor

    def create_model(self, input_dim):
        """创建神经网络模型"""
        print("=== 步骤5: 创建神经网络模型 ===")

        class InsuranceNet(nn.Module):
            def __init__(self, input_dim):
                super(InsuranceNet, self).__init__()
                self.layer1 = nn.Linear(input_dim, 128)
                self.layer2 = nn.Linear(128, 64)
                self.output = nn.Linear(64, 1)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.3)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.relu(self.layer1(x))
                x = self.dropout(x)
                x = self.relu(self.layer2(x))
                x = self.dropout(x)
                x = self.sigmoid(self.output(x))
                return x

        self.model = InsuranceNet(input_dim).to(self.device)
        print(f"模型结构:")
        print(f"- 输入层: {input_dim}")
        print(f"- 隐藏层1: 128 (ReLU + Dropout)")
        print(f"- 隐藏层2: 64 (ReLU + Dropout)")
        print(f"- 输出层: 1 (Sigmoid)")

        return self.model

    def train_model(self, X_tensor, y, test_size=0.2):
        """训练模型"""
        print("=== 步骤6: 训练模型 ===")

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_tensor, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        print(f"正样本比例 - 训练集: {y_train.mean():.3f}, 测试集: {y_test.mean():.3f}")

        # 转换为张量
        y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1).to(self.device)

        # 定义损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        # 训练参数
        epochs = 200
        batch_size = 32
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        # 训练循环
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            epoch_train_loss = 0
            train_predictions = []
            train_targets = []

            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size].to(self.device)
                batch_y = y_train_tensor[i:i + batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                train_predictions.extend((outputs > 0.5).int().cpu().numpy())
                train_targets.extend(batch_y.cpu().numpy())

            # 验证阶段
            self.model.eval()
            epoch_val_loss = 0
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for i in range(0, len(X_test), batch_size):
                    batch_X = X_test[i:i + batch_size].to(self.device)
                    batch_y = y_test_tensor[i:i + batch_size]

                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)

                    epoch_val_loss += loss.item()
                    val_predictions.extend((outputs > 0.5).int().cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())

            # 计算指标
            train_accuracy = accuracy_score(train_targets, train_predictions)
            val_accuracy = accuracy_score(val_targets, val_predictions)

            avg_train_loss = epoch_train_loss / (len(X_train) // batch_size + 1)
            avg_val_loss = epoch_val_loss / (len(X_test) // batch_size + 1)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            scheduler.step()

            # if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                f'Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')

        # 绘制训练过程
        self.plot_training_process(train_losses, val_losses, train_accuracies, val_accuracies)

        return X_test, y_test_tensor, y_test

    def plot_training_process(self, train_losses, val_losses, train_accuracies, val_accuracies):
        """绘制训练过程"""
        print("=== 步骤7: 绘制训练过程 ===")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 损失曲线
        ax1.plot(train_losses, label='Training Loss', color='blue')
        ax1.plot(val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # 准确率曲线
        ax2.plot(train_accuracies, label='Training Accuracy', color='blue')
        ax2.plot(val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        # 损失对比图
        ax3.plot(train_losses, train_accuracies, 'o-', color='green', alpha=0.7, label='Train')
        ax3.plot(val_losses, val_accuracies, 'o-', color='orange', alpha=0.7, label='Val')
        ax3.set_title('Loss vs Accuracy')
        ax3.set_xlabel('Loss')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True)

        # 最终性能
        final_train_acc = train_accuracies[-1]
        final_val_acc = val_accuracies[-1]
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]

        ax4.bar(['Train Acc', 'Val Acc'], [final_train_acc, final_val_acc], color=['blue', 'red'], alpha=0.7)
        ax4.set_title('Final Performance Comparison')
        ax4.set_ylabel('Accuracy')
        for i, v in enumerate([final_train_acc, final_val_acc]):
            ax4.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('training_process.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("训练过程图表已保存为 'training_process.png'")

    def visualize_model(self):
        """可视化模型结构"""
        print("=== 步骤8: 可视化模型结构 ===")

        try:
            from torchviz import make_dot
            # 创建一个示例输入
            example_input = torch.randn(1, len(self.scaler.mean_)).to(self.device)
            output = self.model(example_input)

            # 生成模型图
            dot = make_dot(output, params=dict(self.model.named_parameters()))
            dot.render("insurance_model", format="png", cleanup=True)
            print("模型结构图已保存为 'insurance_model.png'")

            # 显示模型信息
            print("\n模型参数统计:")
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"总参数数量: {total_params:,}")
            print(f"可训练参数数量: {trainable_params:,}")

        except ImportError:
            print("torchviz未安装，跳过模型可视化。可以通过 'pip install torchviz' 安装")
            print("模型架构信息:")
            print(self.model)

    def evaluate_model(self, X_test, y_test_tensor, y_test):
        """评估模型"""
        print("=== 步骤9: 模型评估 ===")

        self.model.eval()
        with torch.no_grad():
            # 预测概率
            y_pred_proba = self.model(X_test)
            y_pred = (y_pred_proba > 0.5).int()

            # 转换回CPU和numpy
            y_pred_proba_cpu = y_pred_proba.cpu().numpy()
            y_pred_cpu = y_pred.cpu().numpy()
            y_test_cpu = y_test_tensor.cpu().numpy()

        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred_cpu)
        precision = precision_score(y_test, y_pred_cpu, zero_division=0)
        recall = recall_score(y_test, y_pred_cpu, zero_division=0)
        f1 = f1_score(y_test, y_pred_cpu, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba_cpu)

        print(f"\n模型评估结果:")
        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数 (F1-Score): {f1:.4f}")
        print(f"AUC值: {auc:.4f}")

        # 绘制混淆矩阵
        self.plot_confusion_matrix(y_test, y_pred_cpu)

        # 绘制ROC曲线
        self.plot_roc_curve(y_test, y_pred_proba_cpu)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }

    def plot_confusion_matrix(self, y_true, y_pred):
        """绘制混淆矩阵"""
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['未成交', '成交'],
                    yticklabels=['未成交', '成交'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("混淆矩阵已保存为 'confusion_matrix.png'")

    def plot_roc_curve(self, y_true, y_pred_proba):
        """绘制ROC曲线"""
        from sklearn.metrics import roc_curve

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc_score(y_true, y_pred_proba):.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("ROC曲线已保存为 'roc_curve.png'")

    def predict_probability(self, X_new):
        """预测新数据的成交概率"""
        self.model.eval()
        with torch.no_grad():
            # 编码新数据
            X_encoded = X_new.copy()
            categorical_features = [
                '二级能源种类名称', '新续转标识名称', '二级机构名称',
                '三级机构名称', '四级机构名称', '品牌', '车型名称'
            ]

            for feature in categorical_features:
                if feature in X_encoded.columns and feature in self.label_encoders:
                    le = self.label_encoders[feature]
                    X_encoded[feature] = X_encoded[feature].fillna('Unknown')
                    # 处理未见过的类别
                    X_encoded[feature] = X_encoded[feature].apply(
                        lambda x: x if x in le.classes_ else 'Unknown'
                    )
                    X_encoded[feature] = le.transform(X_encoded[feature])

            # 标准化
            X_scaled = self.scaler.transform(X_encoded)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            # 预测
            probabilities = self.model(X_tensor).cpu().numpy()

        return probabilities.flatten()


def main():
    """主函数"""
    print("开始车险报价成交概率预测模型构建...")

    # 初始化模型
    model_handler = InsurancePredictionModel()

    try:
        # 步骤1: 加载和预处理数据
        X, y, categorical_features, available_features = model_handler.load_and_preprocess_data('your_data.csv')

        # 步骤2: 编码分类特征
        X_encoded = model_handler.encode_features(X, categorical_features)

        # 步骤3: 准备模型输入特征
        X_tensor = model_handler.prepare_features(X_encoded)

        # 步骤4: 创建模型
        model_handler.create_model(X_tensor.shape[1])

        # 步骤5: 训练模型
        X_test, y_test_tensor, y_test = model_handler.train_model(X_tensor, y)

        # 步骤6: 可视化模型
        # model_handler.visualize_model()

        # 步骤7: 评估模型
        evaluation_results = model_handler.evaluate_model(X_test, y_test_tensor, y_test)

        print("\n=== 模型构建完成 ===")
        print("生成的文件:")
        print("- training_process.png: 训练过程图表")
        print("- insurance_model.png: 模型结构图 (如果安装了torchviz)")
        print("- confusion_matrix.png: 混淆矩阵")
        print("- roc_curve.png: ROC曲线")

        # 示例预测
        print("\n=== 示例预测 ===")
        sample_data = X.iloc[:5].copy()  # 使用前5个样本作为示例
        probabilities = model_handler.predict_probability(sample_data)
        print("前5个样本的成交概率:")
        for i, prob in enumerate(probabilities):
            print(f"样本 {i + 1}: {prob:.4f} ({'成交' if prob > 0.5 else '未成交'})")

    except FileNotFoundError:
        print("错误: 找不到数据文件 'your_data.csv'，请确保文件路径正确")
    except Exception as e:
        print(f"模型构建过程中出现错误: {str(e)}")


if __name__ == "__main__":
    main()
