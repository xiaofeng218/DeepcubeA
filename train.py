import os
import torch
import random
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, model_checkpoint
from pytorch_lightning.loggers import TensorBoardLogger
from config import Config
from dataset.dataloader import RubikDataModule
from model.DeepcubeA_module import DeepcubeA
import datetime

torch.set_float32_matmul_precision('medium')

def main():
    # 解析配置
    config = Config()
    args = config.parse_args()
    
    # 设置随机种子
    seed_everything(args.seed, workers=True)

    args.log_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M"))
    args.checkpoint_dir = os.path.join(args.log_dir, args.checkpoint_dir)
    args.converged_checkpoint_dir = os.path.join(args.log_dir, args.converged_checkpoint_dir)

    # 设置 accelerator & devices
    if args.devices.lower() == "cpu":
        accelerator = "cpu"
        devices = 1   # CPU 默认只用一个进程
    elif args.devices.lower() == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        devices = "auto"
    else:
        # 用户指定了 GPU id(s)
        accelerator = "gpu"
        if "," in args.devices:
            devices = [int(x) for x in args.devices.split(",")]
        else:
            devices = [int(args.devices)]
    
    # 创建必要的目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.converged_checkpoint_dir, exist_ok=True)
    
    # 初始化模型（只初始化一次，后续复用）
    model = DeepcubeA(args)
    
    # 设置初始K值和最大K值
    initial_K = 16
    max_K = args.K  # 可以根据需要调整

    model_e_checkpoint = "logs/20250818_1819/converged_checkpoints/final_model_K_14.pth"
    model.model_theta_e.load_state_dict(torch.load(model_e_checkpoint))
    model_checkpoint = "logs/20250818_1819/converged_checkpoints/final_model_K_15.pth"
    model.model_theta.load_state_dict(torch.load(model_checkpoint))

    for K in range(initial_K, max_K + 1):
        print(f'\n--- 开始训练 K={K} ---')
        
        # 更新模型的K值
        model.K = K
        
        # 创建新的数据集配置
        args.K = K  # 设置当前K值
        
        # 初始化新的数据模块
        data_module = RubikDataModule(args)
        
        # # 设置回调函数，暂时不添加这个，因为好像没什么用
        # checkpoint_callback = ModelCheckpoint(
        #     dirpath=args.checkpoint_dir,
        #     filename=f'K_{K}_'+'{epoch}-{val_loss:.2f}',
        #     save_top_k=3,
        #     monitor='val_loss',
        #     mode='min'
        # )
        
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min',
        )
        
        # lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        # # 设置日志记录器（每个K值使用不同的日志目录）
        # logger = TensorBoardLogger(
        #     save_dir=args.log_dir,
        #     name=f'train_logs_K_{K}'
        # )
        
        # 初始化新的训练器，默认每个epoch验证一次，即5000步
        trainer = Trainer(
            max_epochs=args.max_epochs,
            accelerator=accelerator,
            precision="16-mixed",  # 启用混合精度
            devices=devices,
            logger=False,
            callbacks=[early_stopping_callback],
            deterministic=True,
            enable_progress_bar=True,
            enable_checkpointing=True
        )

        print(trainer.log_every_n_steps)
        
        # 训练模型
        trainer.fit(model, datamodule=data_module)
        
        print(f'--- 完成训练 K={K} ---\n')
    
if __name__ == '__main__':
    main()