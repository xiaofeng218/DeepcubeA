import numpy as np

# Target State
# 用整数表示颜色（0=白, 1=红, 2=绿, 3=黄, 4=橙, 5=蓝）
TARGET_STATE = np.array([
    0,0,0, 0,0,0, 0,0,0,  # U 面
    1,1,1, 1,1,1, 1,1,1,  # R 面
    2,2,2, 2,2,2, 2,2,2,  # F 面
    3,3,3, 3,3,3, 3,3,3,  # D 面
    4,4,4, 4,4,4, 4,4,4,  # L 面
    5,5,5, 5,5,5, 5,5,5   # B 面
], dtype=np.int32)

TARGET_STATE_ONE_HOT = np.eye(6)[TARGET_STATE]

# Function to print state as a cube unfolded diagram
def print_cube_state(state, title=None):
    print(title)
    # U face
    print(" " * 6 + " ".join(map(str, state[0:3])))
    print(" " * 6 + " ".join(map(str, state[3:6])))
    print(" " * 6 + " ".join(map(str, state[6:9])))
    # L, F, R, B faces
    for i in range(3):
        print(" ".join(map(str, state[36+i*3:39+i*3])) + " " + 
              " ".join(map(str, state[18+i*3:21+i*3])) + " " + 
              " ".join(map(str, state[9+i*3:12+i*3])) + " " + 
              " ".join(map(str, state[51+i*3:48+i*3])))
    # D face
    print(" " * 6 + " ".join(map(str, state[27:30])))
    print(" " * 6 + " ".join(map(str, state[30:33])))
    print(" " * 6 + " ".join(map(str, state[33:36])))
    print()

def invert_mapping(mapping):
        """生成逆映射"""
        inv = np.empty(len(mapping), dtype=int)
        inv[mapping] = np.arange(len(mapping))
        return inv

def make_moves():
    """
    生成 3x3x3 魔方的贴纸索引映射（逆时针 + 顺时针）
    贴纸编号顺序：
      U: 0-8,  R: 9-17,  F: 18-26,
      D: 27-35, L: 36-44, B: 51-47
    """

    moves = {}

    def cycle(mapping, positions):
        """按循环位置更新映射"""
        temp = mapping.copy()
        for cycle_pos in positions:
            cycle = np.append(cycle_pos, cycle_pos[0])
            mapping[cycle[:-1]] = temp[cycle[1:]]

    # 初始化基础状态（映射到自身）
    identity = np.arange(54)

    # 定义每个面的顺时针旋转
    face_cycles = {
        'U': [
            # 上面自身旋转
            [2, 8, 6, 0], [5, 7, 3, 1],
            # 侧面环
            [20, 9, 53, 36], [19, 10, 52, 37], [18, 11, 51, 38] 
        ],
        'D': [
            [29, 35, 33, 27], [28, 32, 34, 30],
            [24, 44, 45, 17], [25, 43, 46, 16], [26, 42, 47, 15]
        ],
        'F': [
            [18, 24, 26, 20], [19, 21, 25, 23],
            [17, 2, 36, 33], [14, 1, 39, 34], [11, 0, 42, 35]
        ],
        'B': [
            [51, 45, 47, 53], [52, 48, 46, 50],
            [9, 29, 44, 6], [12, 28, 41, 7], [15, 27, 38, 8]
        ],
        'L': [
            [36, 38, 44, 42], [37, 41, 43, 39],
            [33, 18, 6, 47], [30, 21, 3, 50], [27, 24, 0, 53]
        ],
        'R': [
            [17, 15, 9, 11], [16, 12, 10, 14],
            [45, 8, 20, 35], [48, 5, 23, 32], [51, 2, 26, 29]
        ]
    }

    # 生成顺时针和逆时针映射
    for face, cycles in face_cycles.items():
        mapping = identity.copy()
        cycle(mapping, cycles)
        moves[face] = mapping
        moves[face + "_inv"] = invert_mapping(mapping)

    return moves

class Cube:
    def __init__(self):
        # 初始化移动映射
        self.moves = make_moves()
        
    def apply_action(self, state, action):
        """
        根据输入的action和state得到新的state
        参数:
            state: 当前魔方状态，np.array
            action: 要执行的动作，如 'U', 'R_inv', 等
        返回:
            新的魔方状态
        """
        if action not in self.moves.keys():
            raise ValueError(f'不支持的动作: {action}')
        return state[self.moves[action]]
    
    def get_neibor_state(self, state):
        """
        获取当前state的所有邻居状态
        参数:
            state: 当前魔方状态，np.array
        返回:
            所有邻居状态的列表，np.array
        """
        neibor_states = []
        for action in self.moves.keys():
            neibor_states.append(self.apply_action(state, action))
        return np.stack(neibor_states, axis=0)

    def is_solved(self, state):
        """
        判断当前state是否是魔方被还原后的state
        参数:
            state: 当前魔方状态，np.array
        返回:
            是否还原的布尔值
        """
        return np.array_equal(state, TARGET_STATE)
    
    def view_state(self, state):
        pass



if __name__ == "__main__":
    # Initialize the Cube object
    cube = Cube()
    
    # Get the initial solved state
    initial_state = np.arange(54)

    # Define the rotation action
    action = 'F'
    # Print the initial state
    print_cube_state(initial_state, "Initial Cube State:")
    
    # Print the applied action
    print(f"Applied action: {action}")

    # Apply the rotation action
    new_state = cube.apply_action(initial_state, action)
    
    # Print the new state
    print_cube_state(new_state, "Cube State after Rotation:")

    action = 'U'

    new_state = cube.apply_action(new_state, action)
    
    # Print the applied action
    print(f"Applied action: {action}")
    print_cube_state(new_state, "Cube State after Rotation:")






