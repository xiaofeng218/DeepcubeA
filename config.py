import argparse

from numpy import False_

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='PyTorch Lightning Training Config')
        
        # 数据配置
        self.parser.add_argument('--data_dir', type=str, default='./data', help='数据存储目录')
        self.parser.add_argument('--batch_size', type=int, default=10000, help='批次大小 (根据Readme设置为10000)')
        self.parser.add_argument('--num_workers', type=int, default=16, help='数据加载线程数')
        self.parser.add_argument('--K', type=int, default=30, help='最大打乱次数 (对于魔方设置为30)')
        self.parser.add_argument('--num_val_samples', type=int, default=10000 * 100, help='每个epoch样本数')
        self.parser.add_argument('--num_train_samples', type=int, default=10000 * 1000, help='每个epoch样本数')
    
        # 训练配置
        self.parser.add_argument('--max_epochs', type=int, default=20, help='最大训练轮数')
        self.parser.add_argument('--learning_rate', type=float, default=2e-4, help='学习率')
        self.parser.add_argument('--weight_decay', type=float, default=0, help='权重衰减 (根据Readme不使用正则化)')
        self.parser.add_argument('--devices', type=str, default="2", help="Devices to use: 'cpu', 'auto', '0', '1', '0,1', etc.")
        self.parser.add_argument('--convergence_threshold', type=float, default=0.05, help='收敛阈值 (根据Readme设置为0.05)')
        self.parser.add_argument('--chunk_size', type=int, default=10000 * 12, help='分块大小 (用于模型预测时的分块处理)')
        self.parser.add_argument('--compile', type=bool, default=True, help='是否编译模型')
        
        # 其他配置
        self.parser.add_argument('--log_dir', type=str, default='./logs', help='日志存储目录')
        self.parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='模型 checkpoint 存储目录')
        self.parser.add_argument('--converged_checkpoint_dir', type=str, default='converged_checkpoints', help='收敛模型 checkpoint 存储目录')
        self.parser.add_argument('--seed', type=int, default=42, help='随机种子')

        # inference
        self.parser.add_argument('--model_path', type=str, default='checkpoint/final_model_K_30.pth', help='模型路径')
        self.parser.add_argument('--actions', type=str, default=None, help='指定的魔方动作序列，用空格分隔，如 "U R F D L B"')
        
    def parse_args(self):
        return self.parser.parse_args()

if __name__ == '__main__':
    config = Config()
    args = config.parse_args()
    print(args)