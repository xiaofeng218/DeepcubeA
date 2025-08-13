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

# 判断A中是否有和b相同的张量
def row_allclose_mask(A, b, rtol=1e-4, atol=1e-6):
    # 计算逐元素误差
    diff = torch.abs(A - b)  # (B, D)
    tol = atol + rtol * torch.abs(b)  # (D,), 广播自动扩展到 (B, D)
    
    # 满足误差条件的元素掩码
    mask_elements = diff <= tol  # (B, D), bool
    
    # 判断每行是否所有元素都满足条件
    mask_rows = mask_elements.all(dim=1)  # (B,)
    
    return mask_rows

class DNN(nn.Module):
    def __init__(self, input_dim, num_residual_blocks=4, zero_output=True, K=1):
        super(DNN, self).__init__()

        self.zero_output = zero_output

        # # K 表示模型可能遇到的最大值，为了保证模型的稳定性控制（不输出太大的值），设置模型输出结果乘以K
        # self.register_buffer('K', torch.tensor(K, dtype=torch.float32))

        self.target_state = torch.tensor(TARGET_STATE_ONE_HOT, dtype=torch.float32).reshape(1, -1)
        
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
        # 如果x直接是目标状态，输出0
        if self.zero_output and self.eval():
            return torch.zeros((x.shape[0], 1), device=x.device)
        with torch.no_grad():
            target_mask = row_allclose_mask(x, self.target_state.to(x.device))

        # 前两个隐藏层
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        
        # 残差块
        for block in self.residual_blocks:
            x = block(x)
        
        # 输出层
        x = self.output_layer(x)

        x[target_mask] = 0.0
        
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