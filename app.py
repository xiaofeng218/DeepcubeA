import flask
from flask import request, jsonify
import torch
import numpy as np
import os
from config import Config
from model.DNN import DNN
from model.Cube import Cube, TARGET_STATE
from solver_utils import *

# 初始化Flask应用
app = flask.Flask(__name__, static_folder=None)
app.config['JSON_AS_ASCII'] = False
app.config['DEBUG'] = True

# 加载配置
config = Config()
args = config.parse_args()
# 设置默认模型路径
model_path = args.model_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和创建Cube对象
model = load_model(model_path, device)
cube = Cube()

# 初始化状态接口
@app.route('/initState', methods=['POST'])
def init_state():
    # 初始状态设置为目标状态
    initial_state = TARGET_STATE.copy()

    # 生成旋转索引和状态映射
    rotateIdxs_old = {}
    rotateIdxs_new = {}
    for move_name in cube.moves.keys():
        # 使用Cube类中的实际移动映射
        move_mapping = cube.moves[move_name]
        # 构建old到new的映射
        rotateIdxs_old[move_name] = move_mapping.tolist()
        rotateIdxs_new[move_name] = list(range(54))

    # 定义状态到特征提取和反向的映射
    # 这里假设状态和特征提取使用相同的顺序
    stateToFE = list(range(54))
    FEToState = list(range(54))
    legalMoves = list(cube.moves.keys())

    response = {
        'state': initial_state.tolist(),
        'rotateIdxs_old': rotateIdxs_old,
        'rotateIdxs_new': rotateIdxs_new,
        'stateToFE': stateToFE,
        'FEToState': FEToState,
        'legalMoves': legalMoves
    }

    return jsonify(response)

# 求解魔方接口
@app.route('/solve', methods=['POST'])
def solve():
    try:
        data = request.json
        if not data or 'state' not in data:
            return jsonify({'error': '请求参数错误，缺少state字段'}), 400

        state = np.array(data['state'])
        if state.shape != (54,):
            return jsonify({'error': 'state参数格式错误，应为长度为54的数组'}), 400

        print("开始求解魔方...")
        action_path, solution_state_path = a_star_search(state, model, cube)

        if action_path is None:
            return jsonify({'error': '未能找到解决方案'}), 404

        # 生成反向动作路径
        solveMoves_rev = []
        for action in action_path:
            rev_action = action[:]
            # 反转动作方向
            if "inv" in rev_action:
                rev_action = rev_action[0]
            else:
                rev_action += "_inv"
            solveMoves_rev.append(rev_action)
            
        print(action_path)
        print(solveMoves_rev)

        response = {
            'moves': [action for action in action_path],
            'moves_rev': solveMoves_rev,
            'solve_text': action_path
        }

        return jsonify(response)
    except Exception as e:
        print(f"求解过程中发生错误: {str(e)}")
        return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500

# 静态文件服务
@app.route('/static/<path:path>')
def send_static(path):
    print("Serving static file:", path)
    return flask.send_from_directory('web/deepcube.igb.uci.edu/static', path)

# 主页
@app.route('/')
def home():
    return flask.send_from_directory('web/deepcube.igb.uci.edu', 'index.html')

# 处理缺失的heapq模块
import heapq

if __name__ == '__main__':
    # 确保checkpoint目录存在
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
        print("创建checkpoint目录，请将模型文件放入该目录")

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"警告：未找到模型文件 {model_path}")
        print("请确保模型文件存在于checkpoint目录中")

    # 启动服务器
    # 修改为仅监听本地主机
    app.run(host='127.0.0.1', port=5000)