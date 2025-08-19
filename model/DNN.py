import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Cube import TARGET_STATE_ONE_HOT

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual
        out = F.relu(out)
        return out

class DNN(nn.Module):
    def __init__(self, input_dim, num_residual_blocks=4):
        super(DNN, self).__init__()
        
        # 前两个隐藏层
        self.fc1 = nn.Linear(input_dim, 5000)
        self.bn1 = nn.BatchNorm1d(5000)
        self.fc2 = nn.Linear(5000, 1000)
        self.bn2 = nn.BatchNorm1d(1000)
        
        # 残差块
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_residual_blocks):
            self.residual_blocks.append(ResidualBlock(1000, 1000))
        
        # 输出层
        self.output_layer = nn.Linear(1000, 1)
        
    def forward(self, x):
        # 前两个隐藏层
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        
        # 残差块
        for block in self.residual_blocks:
            x = block(x)
        
        # 输出层
        x = self.output_layer(x)
        
        return x # * self.K

# 示例用法
if __name__ == '__main__':
    # 假设输入维度为54*6=324（根据Readme中提到的魔方状态表示）
    input_dim = 324
    model = DNN(input_dim, num_residual_blocks=4)
    print(model)
    
    # 测试前向传播
    test_input = torch.randn(10, input_dim)
    output = model(test_input)
    print(f'Input shape: {test_input.shape}')
    print(f'Output shape: {output.shape}')