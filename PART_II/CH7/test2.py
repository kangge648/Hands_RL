import torch
import torch.nn as nn
import torch.optim as optim

# 假设Q_net是一个简单的神经网络模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)  # 第一个全连接层，输入特征数为10，输出特征数为5
        self.fc2 = nn.Linear(5, 2)   # 第二个全连接层，输入特征数为5，输出特征数为2

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型
Q_net = SimpleNet()

# 打印模型的所有参数
for param in Q_net.parameters():
    print(param)

# 使用优化器
optimizer = optim.Adam(Q_net.parameters(), lr=0.001)