import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning import LightningDataModule
from model.Cube import Cube, TARGET_STATE

class RubikDataset(Dataset):
    def __init__(self, config, num_samples, is_train=True):
        super().__init__()
        self.config = config
        self.num_samples = num_samples
        self.is_train = is_train
        self.cube = Cube()
        self.K = config.K  # 最大打乱次数
        self.all_actions = list(self.cube.moves.keys())
        
    def __len__(self):
        return self.num_samples
        
    def get_neighbors(self, state):
        """
        获取给定状态的所有邻居状态
        参数:
            state: 当前魔方状态，np.array
        返回:
            所有邻居状态的列表
        """
        return self.cube.get_neibor_state(state)
        
    def __getitem__(self, idx):
        # 随机选择打乱次数 i ∈ [1, K]
        i = np.random.randint(1, self.K + 1)
        
        # 从初始状态开始，随机应用 i 次动作
        state = TARGET_STATE.copy()
        # 采样i个随机动作：
        actions = np.random.choice(self.all_actions, size=i, replace=True)

        for action in actions:
            state = self.cube.apply_action(state, action)
        
        # 获取所有邻居状态
        neighbor_states = self.get_neighbors(state.copy())
        
        # 返回包装成dict的数据
        return {
            'state': state, # 54
            'steps': i, 
            'neighbors': neighbor_states # 12, 54
        }

class RubikDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.num_train_samples = config.num_train_samples
        self.num_val_samples = config.num_val_samples
        
    def prepare_data(self):
        # 不需要下载数据，数据集是自动生成的
        pass
        
    def setup(self, stage=None):
        # 创建训练、验证数据集
        self.train_dataset = RubikDataset(
            self.config, self.num_train_samples, is_train=True
        )
        self.val_dataset = RubikDataset(
            self.config, self.num_val_samples, is_train=False
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            worker_init_fn=self._worker_init_fn
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            worker_init_fn=self._worker_init_fn
        )
        
    def _worker_init_fn(self, worker_id):
        # 获取 worker 的初始种子（会随 epoch 变化）
        worker_seed = (self.config.seed + worker_id + torch.initial_seed()) % 2**32

        # 设置 numpy、torch、python random 的种子
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
