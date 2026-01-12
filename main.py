#!/usr/bin/env python3
"""
SMPL-X 3D人体动画控制与动画生成系统
- 使用正确的关节索引（基于你的 joint_mapper）
- 修复动画渲染空白问题
"""

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QSlider, QLabel, QGroupBox, QGridLayout,
    QSpinBox, QLineEdit, QProgressBar, QMessageBox,
    QTabWidget, QFormLayout, QCheckBox, QScrollArea, QComboBox,
    QFrame, QTextEdit
)
import matplotlib.pyplot as plt
import matplotlib
import sys
import torch
import smplx
import numpy as np
import os
from pathlib import Path

# 兼容NumPy 2.0
np.Inf = np.inf
np.NAN = np.nan
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans',
                                   'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 全局参数 ======================
device = torch.device("cpu")
body_model = None
shape_params = torch.zeros(1, 10, device=device)
pose_params = torch.zeros(1, 156, device=device)

# ====================== 基于你的 joint_mapper 的关节定义 ======================
# body_pose 控制前 21 个身体关节 (0-21)
# body_pose[k] 对应 pose_params[3 + k]

SMPLX_JOINTS = {
    # 身体主关节 (0-21)
    "pelvis": 0,           # 骨盆（根关节）
    "left_hip": 1,         # 左髋关节
    "right_hip": 2,        # 右髋关节
    "spine1": 3,           # 脊柱第一节（腰椎）
    "left_knee": 4,        # 左膝盖
    "right_knee": 5,       # 右膝盖
    "spine2": 6,           # 脊柱第二节（胸椎）
    "left_ankle": 7,       # 左脚踝
    "right_ankle": 8,      # 右脚踝
    "spine3": 9,           # 脊柱第三节（颈椎）
    "left_foot": 10,       # 左脚掌
    "right_foot": 11,      # 右脚掌
    "neck": 12,            # 脖子
    "left_collar": 13,     # 左锁骨
    "right_collar": 14,    # 右锁骨
    "head": 15,            # 头部
    "left_shoulder": 16,   # 左肩
    "right_shoulder": 17,  # 右肩
    "left_elbow": 18,      # 左肘
    "right_elbow": 19,     # 右肘
    "left_wrist": 20,      # 左手腕
    "right_wrist": 21,     # 右手腕

    # 注意：22-54 是下巴、眼睛、手指关节，
    # 这些不在 body_pose 的 63 个参数控制范围内
}

# 全局旋转标识
GLOBAL_ROTATION = 'global'


# ====================== 动画生成线程 ======================
class AnimationWorker(QThread):
    progress_update = pyqtSignal(int, str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, frames, output_path, parent=None):
        super().__init__(parent)
        self.frames = frames
        self.output_path = output_path

    def run(self):
        try:
            from matplotlib.figure import Figure

            # 确保输出目录存在
            Path(self.output_path).mkdir(parents=True, exist_ok=True)
            total_frames = self.frames

            self.progress_update.emit(0, "初始化...")

            # 使用模块级参数
            global shape_params, pose_params

            # 保存初始值用于插值
            params = getattr(self, '_anim_params', {})
            shape_start = params.get('shape_start', 0)
            shape_end = params.get('shape_end', 0)
            joint_configs = params.get('joints', [])

            for frame_idx in range(total_frames):
                # 计算插值因子 t (0 到 1)
                t = frame_idx / max(1, total_frames -
                                    1) if total_frames > 1 else 1.0

                # 清空参数张量
                current_shape = torch.zeros(1, 10, device=device)
                current_pose = torch.zeros(1, 156, device=device)

                # 体型插值
                current_shape[0, 0] = shape_start + \
                    (shape_end - shape_start) * t

                # 关节插值
                for joint_info in joint_configs:
                    idx = joint_info['idx']
                    start_val = joint_info['start_val']
                    end_val = joint_info['end_val']

                    # 角度转弧度
                    start_rad = start_val * np.pi / 180
                    end_rad = end_val * np.pi / 180
                    current_rad = start_rad + (end_rad - start_rad) * t

                    if idx == GLOBAL_ROTATION:
                        current_pose[0, 0] = current_rad
                    else:
                        pose_idx = 3 + idx
                        if 0 <= pose_idx < 156:
                            current_pose[0, pose_idx] = current_rad

                # 更新全局参数（用于渲染）
                shape_params = current_shape.clone()
                pose_params = current_pose.clone()

                # 进度更新
                if frame_idx % max(1, total_frames // 10) == 0:
                    progress = int(t * 100)
                    self.progress_update.emit(
                        progress, f"渲染帧 {frame_idx + 1}/{total_frames}")

                # 渲染当前帧
                self._render_frame(frame_idx)

            self.progress_update.emit(100, "完成!")
            self.finished_signal.emit(self.output_path)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(f"渲染失败: {str(e)}")

    def _render_frame(self, frame_idx):
        """渲染单帧并保存"""
        global shape_params, pose_params, body_model

        try:
            from matplotlib.figure import Figure

            fig = Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111, projection='3d')

            # 设置坐标轴
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(0, 2)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(f"Frame {frame_idx + 1}")
            ax.view_init(elev=20, azim=45)

            if body_model is None:
                ax.text(0, 0, 1, "模型未加载", ha="center",
                        va="center", fontsize=14)
            else:
                # 渲染模型
                body_output = body_model(
                    betas=shape_params,
                    body_pose=pose_params[:, 3:66],
                    global_orient=pose_params[:, 0:3],
                    left_hand_pose=pose_params[:, 66:111],
                    right_hand_pose=pose_params[:, 111:],
                )

                vertices = body_output.vertices.detach().cpu().numpy()[0]
                faces = body_model.faces

                ax.plot_trisurf(
                    vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    triangles=faces, alpha=0.7, color="#4682B4",
                    linewidth=0, antialiased=True
                )

            # 保存图片
            output_file = os.path.join(
                self.output_path, f"frame_{frame_idx:04d}.png")
            fig.savefig(output_file, dpi=100, bbox_inches='tight')

            # 关闭图形释放内存
            plt.close(fig)

        except Exception as e:
            print(f"渲染帧 {frame_idx} 失败: {e}")

    def set_params(self, shape_start, shape_end, joint_configs):
        """设置动画参数"""
        self._anim_params = {
            'shape_start': shape_start,
            'shape_end': shape_end,
            'joints': joint_configs
        }


# ====================== 主窗口类 ======================
class HumanAnimationSystem(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("SMPL-X 3D人体动画控制与动画生成系统")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1000, 700)

        self.generate_btn = None
        self.animation_thread = None

        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # 左侧画布
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self._init_axes()
        self.canvas = FigureCanvas(self.fig)
        left_layout.addWidget(self.canvas, 7)

        self.status_label = QLabel("状态: 等待加载模型")
        self.status_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.status_label, 0)

        main_layout.addWidget(left_container, 6)

        # 右侧控制面板（带滚动）
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        right_container = QWidget()
        right_container.setMinimumWidth(450)
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # 标签页
        self.tab_widget = QTabWidget()

        self.tab_single = QWidget()
        self._setup_single_frame_tab()
        self.tab_widget.addTab(self.tab_single, "单帧控制")

        self.tab_animation = QWidget()
        self._setup_animation_tab()
        self.tab_widget.addTab(self.tab_animation, "动画生成")

        self.tab_index = QWidget()
        self._setup_index_tab()
        self.tab_widget.addTab(self.tab_index, "关节索引")

        right_layout.addWidget(self.tab_widget)

        right_scroll.setWidget(right_container)
        main_layout.addWidget(right_scroll, 4)

        self._draw_empty_hint()

    def _init_axes(self):
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(0, 2)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("SMPL-X 人体模型预览")
        self.ax.view_init(elev=20, azim=45)

    def _setup_single_frame_tab(self):
        """单帧控制"""
        layout = QVBoxLayout(self.tab_single)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # 模型加载
        load_group = QGroupBox("模型加载")
        load_layout = QHBoxLayout(load_group)
        load_layout.setContentsMargins(5, 5, 5, 5)

        self.load_btn = QPushButton("加载模型")
        self.load_btn.setFixedHeight(30)
        self.load_btn.clicked.connect(self._load_smplx_model)

        self.model_label = QLabel("未加载")
        self.model_label.setWordWrap(True)
        self.model_label.setFrameStyle(QFrame.StyledPanel)

        load_layout.addWidget(self.load_btn, 1)
        load_layout.addWidget(self.model_label, 2)
        layout.addWidget(load_group)

        # 体型
        shape_group = QGroupBox("体型参数 β₀")
        shape_layout = QHBoxLayout(shape_group)
        shape_layout.setContentsMargins(5, 5, 5, 5)

        self.shape_slider = QSlider(Qt.Horizontal)
        self.shape_slider.setRange(-5, 5)
        self.shape_slider.setValue(0)
        self.shape_slider.setFixedHeight(20)
        self.shape_slider.valueChanged.connect(self._update_shape)

        self.shape_label = QLabel("0")
        self.shape_label.setFixedWidth(30)

        shape_layout.addWidget(QLabel("β₀:"))
        shape_layout.addWidget(self.shape_slider, 1)
        shape_layout.addWidget(self.shape_label)

        layout.addWidget(shape_group)

        # 核心关节 - 使用你的 joint_mapper 中的正确索引
        joint_group = QGroupBox("核心关节 (使用你的 joint_mapper 索引)")
        joint_layout = QGridLayout(joint_group)
        joint_layout.setContentsMargins(5, 5, 5, 5)
        joint_layout.setSpacing(3)

        # 根据你的 joint_mapper，核心关节索引为：
        # left_shoulder=16, right_shoulder=17
        # left_elbow=18, right_elbow=19
        # spine2=6 (腰部), left_hip=1, right_hip=2
        # left_knee=4, right_knee=5
        self.core_joints = [
            ("全局Y", GLOBAL_ROTATION, 0),
            ("左肩X", SMPLX_JOINTS["left_shoulder"], 0),    # 16
            ("右肩X", SMPLX_JOINTS["right_shoulder"], 0),   # 17
            ("左肘X", SMPLX_JOINTS["left_elbow"], 0),       # 18
            ("右肘X", SMPLX_JOINTS["right_elbow"], 0),      # 19
            ("腰部X", SMPLX_JOINTS["spine2"], 0),           # 6
            ("左髋X", SMPLX_JOINTS["left_hip"], 0),         # 1
            ("右髋X", SMPLX_JOINTS["right_hip"], 0),        # 2
            ("左膝X", SMPLX_JOINTS["left_knee"], 0),        # 4
            ("右膝X", SMPLX_JOINTS["right_knee"], 0),       # 5
        ]

        self.core_sliders = {}
        self.core_labels = {}

        for i, (name, idx, val) in enumerate(self.core_joints):
            row, col = i // 2, (i % 2) * 3

            name_label = QLabel(f"{name}:")
            name_label.setFixedWidth(45)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(-90, 90)
            slider.setValue(val)
            slider.setFixedHeight(18)
            slider.valueChanged.connect(
                lambda v, id=idx: self._update_joint(v, id))

            value_label = QLabel("0°")
            value_label.setFixedWidth(35)

            self.core_sliders[idx] = slider
            self.core_labels[idx] = value_label

            joint_layout.addWidget(name_label, row, col)
            joint_layout.addWidget(slider, row, col + 1)
            joint_layout.addWidget(value_label, row, col + 2)

        layout.addWidget(joint_group)

        # 重置按钮
        reset_btn = QPushButton("重置所有参数")
        reset_btn.setFixedHeight(30)
        reset_btn.clicked.connect(self._reset_all)
        layout.addWidget(reset_btn)

        layout.addStretch()

    def _setup_animation_tab(self):
        """动画生成"""
        layout = QVBoxLayout(self.tab_animation)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # 输出设置
        dir_group = QGroupBox("输出设置")
        dir_layout = QFormLayout()
        dir_layout.setContentsMargins(5, 5, 5, 5)
        dir_layout.setSpacing(5)

        # 输出目录
        dir_hbox = QHBoxLayout()
        self.output_dir_edit = QLineEdit("./output_frames")
        self.output_dir_edit.setFixedHeight(30)
        self.output_dir_edit.setPlaceholderText("输入输出目录路径")

        dir_btn = QPushButton("浏览")
        dir_btn.setFixedSize(60, 30)
        dir_btn.clicked.connect(self._browse_output_dir)

        dir_hbox.addWidget(self.output_dir_edit, 1)
        dir_hbox.addWidget(dir_btn, 0)
        dir_layout.addRow(QLabel("输出目录:"), dir_hbox)

        # 帧数
        self.frame_count = QSpinBox()
        self.frame_count.setRange(1, 500)
        self.frame_count.setValue(30)
        self.frame_count.setFixedHeight(30)
        dir_layout.addRow(QLabel("帧数:"), self.frame_count)

        layout.addWidget(dir_group)

        # 动画参数
        anim_group = QGroupBox("动画参数")
        anim_layout = QVBoxLayout(anim_group)
        anim_layout.setContentsMargins(5, 5, 5, 5)
        anim_layout.setSpacing(3)

        hint = QLabel("勾选需要动画的参数并设置开始/结束值:")
        hint.setWordWrap(True)
        hint.setStyleSheet("QLabel { color: #666; }")
        anim_layout.addWidget(hint)

        # 体型动画
        shape_hbox = QHBoxLayout()
        shape_hbox.addWidget(QLabel("体型β₀:"))

        self.anim_shape_start = QSpinBox()
        self.anim_shape_start.setRange(-5, 5)
        self.anim_shape_start.setValue(0)
        self.anim_shape_start.setFixedSize(60, 30)

        shape_hbox.addWidget(self.anim_shape_start)
        shape_hbox.addWidget(QLabel("→"))

        self.anim_shape_end = QSpinBox()
        self.anim_shape_end.setRange(-5, 5)
        self.anim_shape_end.setValue(0)
        self.anim_shape_end.setFixedSize(60, 30)

        shape_hbox.addWidget(self.anim_shape_end)
        shape_hbox.addStretch(1)

        shape_widget = QWidget()
        shape_widget.setLayout(shape_hbox)
        shape_widget.setFixedHeight(35)
        anim_layout.addWidget(shape_widget)

        # 关节动画
        self.anim_joint_widgets = {}

        for name, idx, val in self.core_joints:
            joint_hbox = QHBoxLayout()
            joint_hbox.setContentsMargins(0, 0, 0, 0)

            checkbox = QCheckBox()
            checkbox.setChecked(False)
            checkbox.setFixedSize(20, 30)

            name_lbl = QLabel(f"{name}:")
            name_lbl.setFixedWidth(40)

            start_box = QSpinBox()
            start_box.setRange(-180, 180)
            start_box.setValue(0)
            start_box.setFixedSize(60, 30)
            start_box.setSuffix("°")

            arrow_lbl = QLabel("→")
            arrow_lbl.setFixedWidth(20)

            end_box = QSpinBox()
            end_box.setRange(-180, 180)
            end_box.setValue(0)
            end_box.setFixedSize(60, 30)
            end_box.setSuffix("°")

            self.anim_joint_widgets[idx] = (start_box, end_box, checkbox)

            joint_hbox.addWidget(checkbox)
            joint_hbox.addWidget(name_lbl)
            joint_hbox.addWidget(start_box)
            joint_hbox.addWidget(arrow_lbl)
            joint_hbox.addWidget(end_box)
            joint_hbox.addStretch(1)

            joint_widget = QWidget()
            joint_widget.setLayout(joint_hbox)
            joint_widget.setFixedHeight(35)
            anim_layout.addWidget(joint_widget)

        layout.addWidget(anim_group)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(25)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        self.anim_status_label = QLabel("就绪")
        self.anim_status_label.setAlignment(Qt.AlignCenter)
        self.anim_status_label.setFixedHeight(25)
        self.anim_status_label.setFrameStyle(QFrame.StyledPanel)
        layout.addWidget(self.anim_status_label)

        # 生成按钮
        self.generate_btn = QPushButton("生成动画帧序列")
        self.generate_btn.setFixedHeight(40)
        self.generate_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666; }"
        )
        self.generate_btn.clicked.connect(self._generate_animation)
        layout.addWidget(self.generate_btn)

        layout.addStretch()

    def _setup_index_tab(self):
        """关节索引"""
        layout = QVBoxLayout(self.tab_index)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # 说明
        info_group = QGroupBox("你的 joint_mapper (已根据你的模型更新)")
        info_layout = QVBoxLayout(info_group)
        info_layout.setContentsMargins(5, 5, 5, 5)

        info_text = QLabel(
            "根据你提供的 joint_mapper，SMPL-X 关节索引如下：\n\n"
            "身体主关节 (0-21，控制方式：body_pose[k] → pose_params[3+k])：\n"
            "  0: pelvis(骨盆)    1: left_hip    2: right_hip\n"
            "  3: spine1          4: left_knee   5: right_knee\n"
            "  6: spine2(腰部)    7: left_ankle  8: right_ankle\n"
            "  9: spine3          10: left_foot  11: right_foot\n"
            "  12: neck           13: left_collar 14: right_collar\n"
            "  15: head           16: left_shoulder 17: right_shoulder\n"
            "  18: left_elbow     19: right_elbow\n"
            "  20: left_wrist     21: right_wrist\n\n"
            "注意：22-54 是手指关节等，不在 body_pose 的 63 个参数控制范围内"
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        layout.addWidget(info_group)

        # 快速参考
        ref_group = QGroupBox("常用关节索引速查")
        ref_layout = QVBoxLayout(ref_group)

        ref_text = QLabel(
            "关节名称          body_pose    pose_params\n"
            "───────────────────────────────────────\n"
            f"全局旋转          global        0\n"
            f"左肩              16            19\n"
            f"右肩              17            20\n"
            f"左肘              18            21\n"
            f"右肘              19            22\n"
            f"左腕              20            23\n"
            f"右腕              21            24\n"
            f"腰部(spine2)      6             9\n"
            f"左髋              1             4\n"
            f"右髋              2             5\n"
            f"左膝              4             7\n"
            f"右膝              5             8"
        )
        ref_text.setFont(QFont("Monospace", 9))
        ref_text.setWordWrap(False)
        ref_layout.addWidget(ref_text)
        layout.addWidget(ref_group)

        # joint_mapper 显示
        mapper_group = QGroupBox("joint_mapper 详细信息")
        mapper_layout = QVBoxLayout(mapper_group)

        self.mapper_text = QTextEdit()
        self.mapper_text.setReadOnly(True)
        self.mapper_text.setMaximumHeight(200)
        self.mapper_text.setText("请先加载模型以查看 joint_mapper")
        mapper_layout.addWidget(self.mapper_text)

        layout.addWidget(mapper_group)

        layout.addStretch()

    def _browse_output_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self, "选择输出目录", "./",
            QFileDialog.ShowDirsOnly
        )
        if directory:
            self.output_dir_edit.setText(directory)

    def _load_smplx_model(self):
        """加载SMPL-X模型"""
        global body_model

        try:
            possible_paths = [
                "./smplx_models",
                "../smplx_models",
                "./models/smplx",
                "./SMPLX",
                "/home/kyomoto/repo/python/smpl-render/smplx_models",
            ]

            model_loaded = False
            for model_path in possible_paths:
                if os.path.exists(model_path):
                    try:
                        body_model = smplx.create(
                            model_path=model_path,
                            model_type="smplx",
                            gender="neutral",
                            flat_hand_mean=True,
                            use_pca=False,
                            num_pca_comps=45,
                            device=device
                        )
                        self.model_label.setText("已加载")
                        print(f"✓ 模型加载成功: {model_path}")
                        model_loaded = True

                        # 显示 joint_mapper
                        if hasattr(body_model, 'joint_mapper'):
                            mapper = body_model.joint_mapper
                            mapper_info = "关节名称 -> body_pose索引 -> pose_params索引:\n"
                            mapper_info += "-" * 60 + "\n"
                            for name in sorted(mapper.keys(), key=lambda x: mapper[x]):
                                idx = mapper[name]
                                pose_idx = 3 + idx
                                mapper_info += f"  {name:20s} -> body_pose[{
                                    idx:2d}] -> pose_params[{pose_idx:2d}]\n"
                            self.mapper_text.setText(mapper_info)

                        break
                    except Exception as e:
                        print(f"尝试 {model_path} 失败: {e}")
                        continue

            if not model_loaded:
                model_path = QFileDialog.getExistingDirectory(
                    self, "选择SMPL-X模型目录", "./",
                    QFileDialog.ShowDirsOnly
                )
                if model_path:
                    body_model = smplx.create(
                        model_path=model_path,
                        model_type="smplx",
                        gender="neutral",
                        flat_hand_mean=True,
                        use_pca=False,
                        num_pca_comps=45,
                        device=device
                    )
                    self.model_label.setText("已加载(自定义)")
                    model_loaded = True

            if model_loaded:
                self.status_label.setText("状态: 模型已就绪")
                self._update_render()
            else:
                raise Exception("未找到模型")

        except Exception as e:
            error_info = f"加载失败"
            self.model_label.setText(error_info)
            self.status_label.setText(f"状态: {error_info}")
            print(f"✗ {e}")
            QMessageBox.warning(self, "错误", f"加载模型失败:\n{e}")

    def _update_shape(self, value):
        """更新体型参数"""
        global shape_params
        shape_params[0, 0] = value
        self.shape_label.setText(str(value))
        self._update_render()

    def _update_joint(self, value, idx):
        """更新关节角度"""
        global pose_params

        rad = value * np.pi / 180

        if idx == GLOBAL_ROTATION:
            pose_params[0, 0] = rad
        else:
            pose_idx = 3 + idx
            if 0 <= pose_idx < 156:
                pose_params[0, pose_idx] = rad

        if idx in self.core_labels:
            self.core_labels[idx].setText(f"{value}°")

        self._update_render()

    def _reset_all(self):
        """重置所有参数"""
        global shape_params, pose_params
        shape_params = torch.zeros(1, 10, device=device)
        pose_params = torch.zeros(1, 156, device=device)

        self.shape_slider.setValue(0)
        self.shape_label.setText("0")

        for idx in self.core_sliders:
            self.core_sliders[idx].setValue(0)
            self.core_labels[idx].setText("0°")

        self._update_render()
        self.status_label.setText("状态: 已重置")

    def _update_render(self):
        """更新3D渲染"""
        global shape_params, pose_params, body_model

        self.ax.clear()
        self._init_axes()

        if body_model is None:
            self.ax.text(
                0, 0, 1, "请先加载SMPL-X模型",
                ha="center", va="center", fontsize=14, color='red'
            )
            self.canvas.draw()
            return

        try:
            body_output = body_model(
                betas=shape_params,
                body_pose=pose_params[:, 3:66],
                global_orient=pose_params[:, 0:3],
                left_hand_pose=pose_params[:, 66:111],
                right_hand_pose=pose_params[:, 111:],
            )

            vertices = body_output.vertices.detach().cpu().numpy()[0]
            faces = body_model.faces

            self.ax.plot_trisurf(
                vertices[:, 0], vertices[:, 1], vertices[:, 2],
                triangles=faces, alpha=0.7, color="#4682B4",
                linewidth=0, antialiased=True
            )

            self.status_label.setText("状态: 渲染完成")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.ax.text(
                0, 0, 1, f"渲染错误: {e}",
                ha="center", va="center", fontsize=10, color='red'
            )

        self.canvas.draw()

    def _draw_empty_hint(self):
        self.ax.clear()
        self._init_axes()
        self.ax.text(
            0, 0, 1, "请先加载SMPL-X模型",
            ha="center", va="center", fontsize=14, color='red'
        )
        self.canvas.draw()

    def _generate_animation(self):
        """生成动画"""
        global body_model

        if body_model is None:
            QMessageBox.warning(self, "警告", "请先加载SMPL-X模型!")
            return

        output_path = self.output_dir_edit.text().strip()
        if not output_path:
            output_path = "./output_frames"

        frames = self.frame_count.value()
        if frames < 1:
            QMessageBox.warning(self, "警告", "帧数必须大于0!")
            return

        shape_start = self.anim_shape_start.value()
        shape_end = self.anim_shape_end.value()

        joint_configs = []
        for name, idx, val in self.core_joints:
            if idx in self.anim_joint_widgets:
                start_box, end_box, checkbox = self.anim_joint_widgets[idx]
                if checkbox.isChecked():
                    joint_configs.append({
                        'idx': idx,
                        'start_val': start_box.value(),
                        'end_val': end_box.value(),
                        'name': name
                    })

        if len(joint_configs) == 0:
            reply = QMessageBox.question(
                self, "确认",
                "没有选择任何关节进行动画，是否只生成体型动画?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        self.animation_thread = AnimationWorker(frames, output_path)
        self.animation_thread.set_params(shape_start, shape_end, joint_configs)

        self.animation_thread.progress_update.connect(
            self._on_animation_progress)
        self.animation_thread.finished_signal.connect(
            self._on_animation_finished)
        self.animation_thread.error_signal.connect(self._on_animation_error)

        if self.generate_btn:
            self.generate_btn.setEnabled(False)

        self.animation_thread.start()
        self.status_label.setText("状态: 动画生成中...")

    def _on_animation_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.anim_status_label.setText(message)
        QApplication.processEvents()

    def _on_animation_finished(self, output_path):
        self.progress_bar.setValue(100)
        self.anim_status_label.setText("完成!")

        if self.generate_btn:
            self.generate_btn.setEnabled(True)

        reply = QMessageBox.question(
            self, "完成",
            f"动画帧已保存到:\n{output_path}\n\n是否打开文件夹查看?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if sys.platform == 'win32':
                os.startfile(output_path)
            elif sys.platform == 'darwin':
                os.system(f'open "{output_path}"')
            else:
                os.system(f'xdg-open "{output_path}"')

        self.status_label.setText("状态: 动画已保存")

    def _on_animation_error(self, error_message):
        self.anim_status_label.setText("错误!")

        if self.generate_btn:
            self.generate_btn.setEnabled(True)

        QMessageBox.critical(self, "错误", error_message)
        self.status_label.setText(f"状态: {error_message}")


# ====================== 程序入口 ======================
if __name__ == "__main__":
    try:
        from PyQt5.QtGui import QFont

        app = QApplication(sys.argv)
        app.setStyle('Fusion')

        window = HumanAnimationSystem()
        window.show()

        print("=" * 60)
        print("SMPL-X 3D人体动画控制系统")
        print("=" * 60)
        print("根据你的 joint_mapper，关节索引已更新：")
        print("  左肩 16 -> pose_params[19]")
        print("  右肩 17 -> pose_params[20]")
        print("  左肘 18 -> pose_params[21]")
        print("  右肘 19 -> pose_params[22]")
        print("  腰部  6 -> pose_params[9]")
        print("=" * 60)

        sys.exit(app.exec_())

    except Exception as e:
        import traceback
        print(f"程序错误: {e}")
        traceback.print_exc()
        sys.exit(1)
