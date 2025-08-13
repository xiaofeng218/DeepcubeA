import torch
import numpy as np
import heapq
import os
from config import Config
from model.DNN import DNN
from model.Cube import Cube, TARGET_STATE
from vis import generate_rubiks_html
import time

def load_model(checkpoint_path, device='cpu'):
    """
    加载指定路径的DNN模型作为启发函数h(x)
    参数:
        checkpoint_path: 模型checkpoint文件路径(.pth文件)
        device: 模型加载的设备，默认为'cpu'
    返回:
        加载好的DNN模型
    """
    # 检查模型文件是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

    print(f"加载模型: {checkpoint_path}")

    # 创建模型并加载权重
    input_dim = 54 * 6  # 魔方54个贴纸，每个6种颜色，使用one-hot编码
    model = DNN(input_dim, num_residual_blocks=4, zero_output=False)

    # 加载PyTorch模型
    try:
        # 尝试直接加载pth文件，添加weights_only=True参数
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # 处理模型权重，移除_orig_mod.前缀（与plot_model_output.py保持一致）
        model_weights = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items() if k.startswith('_orig_mod.')}
        
        # 如果没有_orig_mod.前缀，尝试直接加载
        if not model_weights:
            model_weights = checkpoint

        model.load_state_dict(model_weights)
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {str(e)}")

    # 将模型移动到指定设备
    model = model.to(device)
    model.eval()  # 设置为评估模式

    return model


def state_to_one_hot(state):
    """
    将魔方状态转换为one-hot编码
    参数:
        state: 魔方状态，形状为(54,)
    返回:
        one-hot编码后的状态，形状为(54*6,)
    """
    one_hot = np.zeros(54 * 6)
    for i, color in enumerate(state):
        one_hot[i * 6 + color] = 1
    return one_hot


def h(state, model):
    """
    启发函数h(x)，使用DNN模型估计状态到目标状态的距离
    参数:
        state: 魔方状态 (单个状态或批量状态)
        model: 加载好的DNN模型
    返回:
        估计距离 (单个值或批量值)
    """
    # 模型预测
    with torch.no_grad():
        # 检查输入是否为批量状态
        if len(state.shape) == 2:
            # 批量状态，直接预测
            prediction = model(state)
            return prediction.squeeze().tolist()
        else:
            # 单个状态，添加批次维度
            state = state.unsqueeze(0)
            prediction = model(state)
            return prediction.item()


def a_star_search(initial_state, model, cube, lam=0.6, max_iterations=10000, N=10000, device='cpu'):
    """
    A*搜索算法求解魔方
    参数:
        initial_state: 初始魔方状态
        model: 启发函数模型
        cube: Cube对象，用于生成邻居状态
        lam: 路径长度权重
        max_iterations: 最大迭代次数
        N: 每次从堆中选取的最小状态数量
    返回:
        解决路径或None(如果未找到解决方案)
    """
    # 检查初始状态是否为目标状态
    if np.array_equal(initial_state, TARGET_STATE):
        return []

    # 打开列表(优先队列)和关闭列表
    open_set = []
    closed_set = set()

    # g_score: 从初始状态到当前状态的实际路径长度
    # f_score: g_score + h_score
    initial_state_tensor = torch.tensor(initial_state, device=device).long()
    initial_state_tensor = torch.nn.functional.one_hot(initial_state_tensor, num_classes=6).float().view(-1)
    g_score = {tuple(initial_state): 0}
    h_score = {tuple(initial_state): h(initial_state_tensor, model)}
    f_score = {tuple(initial_state): lam * g_score[tuple(initial_state)] + h_score[tuple(initial_state)]}

    # 将初始状态添加到打开列表
    heapq.heappush(open_set, (f_score[tuple(initial_state)], tuple(initial_state)))

    # 记录每个状态的前驱状态和动作
    came_from = {}

    iterations = 0

    while open_set and iterations < max_iterations:
        print(f"迭代 {iterations}, 打开列表大小: {len(open_set)}")

        iterations += 1

        # 获取f_score最小的N个状态
        current_states = []
        for _ in range(min(N, len(open_set))):
            _, state_tuple = heapq.heappop(open_set)
            current_states.append(state_tuple)
        
        best_state_tuple = current_states[0]
        print("f_score:", f_score[best_state_tuple], "g_score:", g_score[best_state_tuple], "h_score:", h_score[best_state_tuple])

        # 收集所有邻居状态
        neighbor_states = []

        # 处理每个选中的状态
        for current_state_tuple in current_states:
            current_state = np.array(current_state_tuple)

            # 如果当前状态是目标状态，重建路径
            if np.array_equal(current_state, TARGET_STATE):
                action_path = []
                state_path = [current_state]
                while current_state_tuple in came_from:
                    current_state_tuple, action = came_from[current_state_tuple]
                    action_path.append(action)
                    state_path.append(current_state_tuple)
                return action_path[::-1], state_path[::-1]  # 反转路径

            # 如果状态已经在关闭列表中，则跳过
            if current_state_tuple in closed_set:
                continue

            # 将当前状态添加到关闭列表
            closed_set.add(current_state_tuple)

            # 生成所有可能的移动
            for action in cube.moves.keys():
                # 应用动作生成新状态
                next_state = cube.apply_action(current_state, action)
                next_state_tuple = tuple(next_state)

                # 计算从初始状态到新状态的路径长度
                tentative_g_score = g_score[current_state_tuple] + 1

                # 如果新状态已经在关闭列表中，判断当前路径是否更短，如果更短，将新状态从关闭列表中移除，否则跳过这个状态
                if next_state_tuple in closed_set:
                    if tentative_g_score < g_score.get(next_state_tuple, float('inf')):
                        closed_set.remove(next_state_tuple)
                    else:
                        continue
                    
                if next_state_tuple not in g_score or tentative_g_score < g_score[next_state_tuple]:
                    # 记录前驱状态和动作
                    came_from[next_state_tuple] = (current_state_tuple, action)
                    # 更新分数
                    g_score[next_state_tuple] = tentative_g_score
                    # 收集邻居状态用于批量预测h值
                    neighbor_states.append(next_state)

        # 批量调用h函数预测所有邻居状态的h值
        if neighbor_states:
            # 转换为PyTorch张量
            neighbor_states = np.stack(neighbor_states)
            neighbor_states = np.unique(neighbor_states, axis=0)
            neighbor_states_tensor = torch.tensor(neighbor_states, device=device).long()
            # 转换为one-hot编码
            neighbor_states_tensor = torch.nn.functional.one_hot(neighbor_states_tensor, num_classes=6).float().view(-1, 324)

            # 调用h函数进行预测
            neighbor_h_scores = h(neighbor_states_tensor, model)
            # 更新分数并添加到打开列表
            for i, state in enumerate(neighbor_states):
                state_tuple = tuple(state)
                h_score[state_tuple] = neighbor_h_scores[i]
                f_score[state_tuple] = lam * g_score[state_tuple] + neighbor_h_scores[i]
                heapq.heappush(open_set, (f_score[state_tuple], state_tuple))

    print(f"超过最大迭代次数({max_iterations})，未找到解决方案")
    return None, None

def generate_html(initial_state, solution_path):
    # 颜色映射 (0=白, 1=红, 2=绿, 3=黄, 4=橙, 5=蓝)
    COLOR_MAP = {
        0: "white",
        1: "red",
        2: "green",
        3: "yellow",
        4: "orange",
        5: "blue"
    }

    # 面顺序和索引范围
    FACE_ORDER = {
        'U': list(range(0, 9)),    # 顶面
        'R': list(range(9, 18)),   # 右面
        'F': list(range(18, 27)),  # 前面
        'D': list(range(27, 36)),  # 底面
        'L': list(range(36, 45)),  # 左面
        'B': list(range(45, 54))   # 后面
    }

    # 将初始状态转换为generate_rubiks_html需要的格式
    initial_state_dict = {}
    for face, indices in FACE_ORDER.items():
        initial_state_dict[face] = [COLOR_MAP[initial_state[i]] for i in indices]

    # 生成每一步的状态
    moves = []
    for state in solution_path:
        move_state = {}
        for face, indices in FACE_ORDER.items():
            move_state[face] = [COLOR_MAP[state[i]] for i in indices]
        moves.append(move_state)
        
    # 调用generate_rubiks_html生成网页
    output_file = "rubiks_solution.html"
    generate_rubiks_html(initial_state_dict, FACE_ORDER, moves, output_file)
    print(f"已生成解决方案网页: {output_file}")


def main():
    # 加载配置
    config = Config()
    args = config.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = load_model(args.model_path, device=device)

    # 创建Cube对象
    cube = Cube()

    # 初始状态为目标状态
    initial_state = TARGET_STATE.copy()
    state_path = []

    if args.actions:
        # 使用用户指定的动作序列
        print(f"使用指定的动作序列: {args.actions}")
        shuffle_actions = args.actions.split()
        # 验证动作是否有效
        valid_actions = set(cube.moves.keys())
        invalid_actions = [action for action in shuffle_actions if action not in valid_actions]
        if invalid_actions:
            raise ValueError(f"无效的动作: {invalid_actions}，有效的动作是: {valid_actions}")
    else:
        # 随机生成动作序列
        print("随机打乱魔方20次...")
        all_actions = list(cube.moves.keys())
        shuffle_actions = np.random.choice(all_actions, size=15, replace=True)

    # 应用动作序列
    for action in shuffle_actions:
        initial_state = cube.apply_action(initial_state, action)
        state_path.append(initial_state.copy())

    print("开始A*搜索...")

    # 执行A*搜索
    start_time = time.time()
    action_path, solution_state_path = a_star_search(initial_state, model, cube, device=device)
    end_time = time.time()
    solving_time = end_time - start_time

    # 保存为可视化结果
    if solution_state_path:
        # 合并打乱路径和解决方案路径以展示完整过程
        # full_state_path = state_path # + solution_state_path[1:]
        generate_html(TARGET_STATE.copy(), state_path)
        # generate_html(initial_state, solution_state_path[1:])
        print(f"找到解决方案，路径长度: {len(solution_state_path)}")
        print(f"求解时间: {solving_time:.4f} 秒")  # 打印求解时间
        print("打乱路径:", shuffle_actions)
        if action_path:
            print("解决方案路径:", action_path)
    else:
        print("未找到解决方案")


if __name__ == '__main__':
    main()