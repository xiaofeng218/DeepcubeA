import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from model.DNN import DNN


class RelativeMSELoss(nn.Module):
    def forward(self, pred, target):
        return torch.mean(((pred - target) / (target + 1e-8)) ** 2)

class DeepcubeA(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.convergence_threshold = config.convergence_threshold
        self.chunk_size = config.chunk_size
        self.converged_checkpoint_dir = config.converged_checkpoint_dir
        self.compile = config.compile
                
        # 输入维度（54个贴纸，每个有6种可能的颜色，使用one-hot编码）
        self.input_dim = 54 * 6
        
        self.model_theta = DNN(self.input_dim, num_residual_blocks=4, zero_output=False)  # 训练模型
        self.model_theta_e = DNN(self.input_dim, num_residual_blocks=4, zero_output=True).eval()  # 监督模型

        if self.compile:
            self.model_theta = torch.compile(self.model_theta)
            self.model_theta_e = torch.compile(self.model_theta_e)

        self.updata_K(1)
        
        # 损失函数，评价这种输出的准确性感觉最好还是相对值
        self.criterion = nn.MSELoss()
        
        # 保存超参数
        self.save_hyperparameters(config)
    
    def updata_K(self, K):
        self.K = K
        # self.model_theta.K.data.fill_(torch.tensor(K, dtype=torch.float32, device=self.device))
        if K != 1:
            self.model_theta_e.zero_output = False

    def transfer_batch_to_tensor(self, batch):
        """
        批量将batch中的数据转移到tensor并移动到正确的设备上
        参数:
            batch: 输入的batch数据
        返回:
            处理后的batch字典，包含tensor格式的数据
        """
        batch_dict = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_dict[key] = value.to(self.device)
            else:
                batch_dict[key] = torch.tensor(value, device=self.device)
        return batch_dict
        
    def forward(self, x):
        return self.model_theta(x)
        
    def model_step(self, batch):
        # 从batch中获取状态和邻居
        batch_dict = self.transfer_batch_to_tensor(batch)
        states = batch_dict['state']
        neighbor_states = batch_dict['neighbors']

        B, N, D = neighbor_states.shape

        states = F.one_hot(states.long(), num_classes=6).float().view(B, -1)
        neighbor_states = F.one_hot(neighbor_states.long(), num_classes=6).float().view(B*N, -1)
        
        # 分块预测以避免显存不足
        num_chunks = (B * N + self.chunk_size - 1) // self.chunk_size
        chunked_neighbors = torch.chunk(neighbor_states, num_chunks, dim=0)
        
        with torch.no_grad():
            neighbor_costs = []
            for chunk in chunked_neighbors:
                cost = self.model_theta_e(chunk)
                neighbor_costs.append(cost)
            
            # 聚合结果
            neighbor_costs = torch.cat(neighbor_costs, dim=0)
            neighbor_costs = neighbor_costs.view(B, N)
        
        # 计算min[J_theta_e(A(x_i, a)) + 1]
        min_neighbor_cost = neighbor_costs.min(dim=1)[0] + 1
        
        # 使用model_theta预测当前状态的cost
        current_cost = self.model_theta(states)
        
        # 总是计算损失
        loss = self.criterion(current_cost.squeeze(), min_neighbor_cost)
        return loss, current_cost
        
    def training_step(self, batch, batch_idx):
        # 调用model_step获取损失
        loss, _ = self.model_step(batch)
        
        # 记录指标
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
        
    def on_validation_epoch_end(self):
        # 获取验证损失
        val_loss = self.trainer.callback_metrics.get('val_loss')
        
        if val_loss is not None and val_loss < self.convergence_threshold:
            self.log('converged', True)
            
            # 保存模型参数到专门的收敛模型目录
            import os
            os.makedirs(self.converged_checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(self.converged_checkpoint_dir, f"converged_model_K_{self.K}.pth")
            torch.save(self.model_theta.state_dict(), checkpoint_path)
            print(f'模型已保存到 {checkpoint_path}')

            # 如果收敛，更新model_theta_e
            self.model_theta_e.load_state_dict(self.model_theta.state_dict())
            self.model_theta_e.zero_output = False
            
            # 原文中没有找到上一轮训练的模型下一轮是否要继承参数，这里选择每轮训练重新初始化model_theta，防止模型过拟合到较小的打乱情况
            self.model_theta = DNN(self.input_dim, num_residual_blocks=4, zero_output=False)
            if self.compile:
                self.model_theta = torch.compile(self.model_theta)
            
            # 停止训练
            self.trainer.should_stop = True
        
    def validation_step(self, batch, batch_idx):
        # 计算验证损失
        loss, current_cost = self.model_step(batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_cost', current_cost.mean(), on_epoch=True, logger=True)
        return loss
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model_theta.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        return {'optimizer': optimizer}

    def load_state_dict_theta_e(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.model_theta_e.load_state_dict(state_dict)
        self.model_theta_e.zero_output = False