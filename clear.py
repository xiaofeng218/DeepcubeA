import gc
import torch

# 删除所有变量引用
gc.collect()

# 清空 PyTorch GPU 缓存
torch.cuda.empty_cache()
