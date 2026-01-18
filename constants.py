#!/usr/bin/env python3
"""
全局常量定义（关节映射、视角参数、预设等）
"""
import numpy as np

# ====================== 全局参数 ======================
DEVICE = "cpu"

# ====================== 视角相关全局变量 ======================
# 默认视角参数（第三方观察视角，能清晰看到全身）
DEFAULT_ELEV = 20
DEFAULT_AZIM = 45
DEFAULT_DIST = 10

# 全局视角变量（供多模块共享）
current_view_elev = DEFAULT_ELEV
current_view_azim = DEFAULT_AZIM
current_view_dist = DEFAULT_DIST
saved_views = {}

# ====================== SMPLX关节字典 + 对应旋转轴 + 精准索引 ======================
SMPLX_JOINTS = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_knee": 4,
    "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "spine3": 9,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "left_collar": 13,
    "right_collar": 14,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}

JOINT_AXIS_MAP = {
    'global': 1,
    1: 2, 2: 2, 4: 2, 5: 2, 7: 2, 8: 2, 10: 2, 11: 2,
    3: 1, 6: 1, 9: 1, 12: 1, 15: 1,
    13: 0, 14: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0
}

GLOBAL_ROTATION = 'global'

# ====================== 视角预设配置 ======================
# 参照默认视角参数设置，确保能看清全身
VIEW_PRESETS = {
    "正前": {"elev": 0, "azim": 0, "desc": "正面视角"},
    "正后": {"elev": 0, "azim": 180, "desc": "背面视角"},
    "正左": {"elev": 0, "azim": 90, "desc": "左侧视角"},
    "正右": {"elev": 0, "azim": -90, "desc": "右侧视角"},
    "俯视": {"elev": 60, "azim": 0, "desc": "俯视视角（倾斜）"},
    "仰视": {"elev": -30, "azim": 0, "desc": "仰视视角（低角度）"},
    "默认": {"elev": DEFAULT_ELEV, "azim": DEFAULT_AZIM, "desc": "第三方视角"},
}

# ====================== 模型加载路径 ======================
MODEL_PATHS = [
    "./smplx_models",
    "../smplx_models",
    "./models/smplx",
    "./SMPLX",
    "/home/kyomoto/repo/python/smpl-render/smplx_models",
]

# ====================== 核心关节列表 ======================
CORE_JOINTS = [
    ("全局Y", GLOBAL_ROTATION, 0),
    ("左髋Y", SMPLX_JOINTS["spine1"], 0),
    ("右腰X", SMPLX_JOINTS["right_hip"], 0),
    ("左腰X", SMPLX_JOINTS["right_knee"], 0),
    ("脖子X", SMPLX_JOINTS["right_foot"], 0),
    ("左肩X", SMPLX_JOINTS["left_shoulder"], 0),
    ("右肩X", SMPLX_JOINTS["right_shoulder"], 0),
    ("左肘X", SMPLX_JOINTS["left_elbow"], 0),
    ("右肘X", SMPLX_JOINTS["right_elbow"], 0),
    ("右脚X", SMPLX_JOINTS["spine2"], 0),
]
