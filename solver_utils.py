import numpy as np
import torch
import heapq
import os
from model.Cube import TARGET_STATE
from model.DNN import DNN


def load_model(model_path, device):
    """
    加载预训练模型
    """
    input_dim = 54 * 6  # 魔方54个贴纸，每个6种颜色，使用one-hot编码
    model = DNN(input_dim, num_residual_blocks=4)

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        # 处理模型权重，移除_orig_mod.前缀
        model_weights = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items() if k.startswith('_orig_mod.')}
        if not model_weights:
            model_weights = checkpoint
        model.load_state_dict(model_weights)
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        raise

    model = model.to(device)
    model.eval()
    return model


def state_to_one_hot(state):
    """
    将魔方状态转换为one-hot编码
    """
    one_hot = np.zeros(54 * 6)
    for i, color in enumerate(state):
        one_hot[i * 6 + color] = 1
    return one_hot


def h(state, model):
    """
    启发函数，使用模型预测当前状态到目标状态的距离
    """
    with torch.no_grad():
        if len(state.shape) == 2:
            prediction = model(state)
            return prediction.squeeze().tolist()
        else:
            state = state.unsqueeze(0)
            prediction = model(state)
            return prediction.item()


def a_star_search(initial_state, model, cube, lam=0.6, max_iterations=200, N=1000):
    """
    A*搜索算法求解魔方
    """
    # 检查初始状态是否为目标状态
    if np.array_equal(initial_state, TARGET_STATE):
        return [], [initial_state]

    open_set = []
    closed_set = set()

    initial_state_tensor = torch.tensor(initial_state, device=next(model.parameters()).device).long()
    initial_state_tensor = torch.nn.functional.one_hot(initial_state_tensor, num_classes=6).float().view(-1)
    g_score = {tuple(initial_state): 0}
    h_score = {tuple(initial_state): h(initial_state_tensor, model)}
    f_score = {tuple(initial_state): lam * g_score[tuple(initial_state)] + h_score[tuple(initial_state)]}

    heapq.heappush(open_set, (f_score[tuple(initial_state)], tuple(initial_state)))

    came_from = {}
    iterations = 0

    while open_set and iterations < max_iterations:
        iterations += 1
        #print(f"当前迭代: {iterations}, 开放集大小: {len(open_set)}")

        current_states = []
        for _ in range(min(N, len(open_set))):
            _, state_tuple = heapq.heappop(open_set)
            current_states.append(state_tuple)

        # 收集所有邻居状态
        neighbor_states = []

        for current_state_tuple in current_states:
            current_state = np.array(current_state_tuple)

            if np.array_equal(current_state, TARGET_STATE):
                action_path = []
                state_path = [current_state]
                while current_state_tuple in came_from:
                    current_state_tuple, action = came_from[current_state_tuple]
                    action_path.append(action)
                    state_path.append(current_state_tuple)
                return action_path[::-1], state_path[::-1]

            if current_state_tuple in closed_set:
                continue

            closed_set.add(current_state_tuple)

            for action in cube.moves.keys():
                next_state = cube.apply_action(current_state, action)
                next_state_tuple = tuple(next_state)

                tentative_g_score = g_score[current_state_tuple] + 1

                if next_state_tuple in closed_set:
                    if tentative_g_score < g_score.get(next_state_tuple, float('inf')):
                        closed_set.remove(next_state_tuple)
                    else:
                        continue

                if next_state_tuple not in g_score or tentative_g_score < g_score[next_state_tuple]:
                    came_from[next_state_tuple] = (current_state_tuple, action)
                    g_score[next_state_tuple] = tentative_g_score
                    neighbor_states.append(next_state)

        if neighbor_states:
            neighbor_states = np.stack(neighbor_states)
            neighbor_states = np.unique(neighbor_states, axis=0)
            neighbor_states_tensor = torch.tensor(neighbor_states, device=next(model.parameters()).device).long()
            neighbor_states_tensor = torch.nn.functional.one_hot(neighbor_states_tensor, num_classes=6).float().view(-1, 324)

            neighbor_h_scores = h(neighbor_states_tensor, model)
            for i, state in enumerate(neighbor_states):
                state_tuple = tuple(state)
                h_score[state_tuple] = neighbor_h_scores[i]
                f_score[state_tuple] = lam * g_score[state_tuple] + neighbor_h_scores[i]
                heapq.heappush(open_set, (f_score[state_tuple], state_tuple))

    return None, None  # 未找到解决方案