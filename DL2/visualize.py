import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from torchviz import make_dot
import matplotlib.pyplot as plt


class EnhancedInsuranceNet(nn.Module):
    def __init__(self):
        super(EnhancedInsuranceNet, self).__init__()
        self.layer1 = nn.Linear(28, 256, bias=True)
        self.bn1 = nn.BatchNorm1d(256)
        self.layer2 = nn.Linear(256, 128, bias=True)
        self.bn2 = nn.BatchNorm1d(128)
        self.layer3 = nn.Linear(128, 64, bias=True)
        self.layer4 = nn.Linear(64, 32, bias=True)
        self.output_layer = nn.Linear(32, 1, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer4(x)
        x = self.relu(x)

        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x


# 安装所需包：
# pip install torchviz graphviz

if __name__ == "__main__":
    # 创建模型和示例输入
    model = EnhancedInsuranceNet()
    x = torch.randn(1, 28)
    y = model(x)

    # 生成计算图
    dot = make_dot(y, params=dict(model.named_parameters()),
                   show_attrs=True, show_saved=True)

    # 保存为图片
    dot.render('enhanced_insurance_net', format='png', cleanup=True)
    print("图片已保存为 'enhanced_insurance_net.png'")
