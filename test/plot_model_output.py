import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# Add the project root directory to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.DNN import DNN
from model.Cube import Cube, TARGET_STATE, TARGET_STATE_ONE_HOT
import torch  # Add missing import
from config import Config

plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

def generate_shuffled_state(cube, num_shuffles):
    """
    生成指定打乱次数的魔方状态
    参数:
        cube: Cube实例
        num_shuffles: 打乱次数
    返回:
        打乱后的魔方状态
    """
    state = TARGET_STATE.copy()
    all_actions = list(cube.moves.keys())
    actions = np.random.choice(all_actions, size=num_shuffles, replace=True)
    for action in actions:
        state = cube.apply_action(state, action)
    return state


def state_to_one_hot(state):
    """
    将魔方状态转换为one-hot编码
    参数:
        state: 魔方状态，形状为(54,)
    返回:
        one-hot编码后的状态，形状为(54*6,)
    """
    one_hot = np.eye(6)[state]
    return one_hot.flatten()


def main():
    # 模型路径
    config = Config()
    args = config.parse_args()
    checkpoint_path = args.model_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    input_dim = 54 * 6  # 魔方54个贴纸，每个6种颜色，使用one-hot编码
    model = DNN(input_dim, num_residual_blocks=4)

    # 加载模型权重
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # 检查是否是PyTorch Lightning保存的checkpoint
        # 提取模型权重
        model_weights = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items() if k.startswith('_orig_mod.')}
        model.load_state_dict(model_weights)
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {str(e)}")

    model = model.to(device)
    model.eval()  # 设置为评估模式
    print("模型加载成功")

    # 创建魔方实例
    cube = Cube()

    # 准备数据
    num_shuffles_range = range(1, 30)  # 1到30次打乱
    num_samples = 1000  # 每个打乱次数的样本数
    avg_outputs = []
    max_outputs = []  # 存储每个打乱次数的最大值

    print("开始计算模型输出...")
    for num_shuffles in num_shuffles_range:
        print(f"处理打乱次数: {num_shuffles}")

        # 批量生成样本
        states = []
        for _ in range(num_samples):
            # 生成打乱的状态
            state = generate_shuffled_state(cube, num_shuffles)
            states.append(state)

        # 批量转换为one-hot编码
        batch_one_hot = np.array([state_to_one_hot(state) for state in states])

        # 转换为tensor
        input_tensor = torch.tensor(batch_one_hot, dtype=torch.float32).to(device)

        # 模型预测（批量）
        with torch.no_grad():
            outputs = model(input_tensor)
            # 将输出转换为numpy数组
            outputs_np = outputs.cpu().numpy().flatten()

        # 计算平均值和最大值
        avg_output = np.mean(outputs_np)
        max_output = np.max(outputs_np)  # 计算最大值
        avg_outputs.append(avg_output)
        max_outputs.append(max_output)  # 存储最大值
        print(f"打乱次数 {num_shuffles} 的平均输出: {avg_output:.4f}, 最大输出: {max_output:.4f}")

    # 绘制统计图
    plt.figure(figsize=(12, 6))
    # 绘制平均值曲线
    plt.plot(num_shuffles_range, avg_outputs, marker='o', linestyle='-', color='b', label='Average Value')
    # 绘制最大值曲线
    plt.plot(num_shuffles_range, max_outputs, marker='s', linestyle='--', color='r', label='Maximum Value')
    plt.title("Relationship between Cube Shuffles and DNN Output")
    plt.xlabel("Number of Shuffles")
    plt.ylabel("Model Output")
    plt.grid(True)
    plt.xticks(num_shuffles_range)
    plt.legend()  # 添加图例
    plt.tight_layout()

    # 保存图像
    save_path = "model_output_vs_shuffles.png"
    plt.savefig(save_path, dpi=300)
    print(f"图表已保存到 {save_path}")


if __name__ == '__main__':
    main()