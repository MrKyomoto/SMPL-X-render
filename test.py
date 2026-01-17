#!/usr/bin/env python3
"""
SMPL-X 3D人体动画控制与动画生成系统 - 视角增强版
新增功能：
① 视角预设：一键切换 正前/正后/正左/正右/俯视/仰视
② 视角保存/加载：支持保存3-5个常用视角
③ 动画插值：线性插值/平滑插值可选
"""

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QSlider, QLabel, QGroupBox, QGridLayout,
    QSpinBox, QLineEdit, QProgressBar, QMessageBox,
    QTabWidget, QFormLayout, QCheckBox, QScrollArea, QComboBox,
    QFrame, QTextEdit, QListWidget, QListWidgetItem,
    QInputDialog, QRadioButton, QButtonGroup
)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib
import sys
import torch
import smplx
import numpy as np
import os
from pathlib import Path
from scipy.interpolate import interp1d

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

# ====================== 视角相关全局变量 ======================
# 默认视角参数（第三方观察视角，能清晰看到全身）
DEFAULT_ELEV = 20
DEFAULT_AZIM = 45
DEFAULT_DIST = 10  # 默认距离放大一些，确保看全

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


# ====================== 动画生成线程（增强版）======================
class AnimationWorker(QThread):
    progress_update = pyqtSignal(int, str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, frames, output_path, parent=None, interpolation="linear"):
        super().__init__(parent)
        self.frames = frames
        self.output_path = output_path
        self.interpolation = interpolation
        self._anim_params = {}
    
    def set_params(self, shape_start, shape_end, joint_configs):
        """设置动画参数"""
        self._anim_params = {
            'shape_start': shape_start,
            'shape_end': shape_end,
            'joints': joint_configs
        }
    
    def run(self):
        try:
            Path(self.output_path).mkdir(parents=True, exist_ok=True)
            total_frames = self.frames
            self.progress_update.emit(0, "初始化...")
            
            global shape_params, pose_params
            params = self._anim_params
            shape_start = params.get('shape_start', 0)
            shape_end = params.get('shape_end', 0)
            joint_configs = params.get('joints', [])
            
            if self.interpolation == "smooth":
                t_points = np.array([0, 0.5, 1])
                def smooth_interpolate(start, end, t):
                    if abs(end - start) < 0.01:
                        return start
                    v_points = np.array([start, (start + end) / 2, end])
                    f = interp1d(t_points, v_points, kind='quadratic')
                    return float(f(t))
            else:
                def smooth_interpolate(start, end, t):
                    return start + (end - start) * t
            
            for frame_idx in range(total_frames):
                t = frame_idx / max(1, total_frames - 1) if total_frames > 1 else 1.0
                current_shape = torch.zeros(1, 10, device=device)
                current_pose = torch.zeros(1, 156, device=device)
                
                if self.interpolation == "smooth":
                    current_shape[0, 0] = smooth_interpolate(shape_start, shape_end, t)
                else:
                    current_shape[0, 0] = shape_start + (shape_end - shape_start) * t
                
                for joint_info in joint_configs:
                    idx = joint_info['idx']
                    start_val = joint_info['start_val']
                    end_val = joint_info['end_val']
                    
                    if self.interpolation == "smooth":
                        current_rad = smooth_interpolate(
                            start_val * np.pi / 180, 
                            end_val * np.pi / 180, 
                            t
                        ) if idx != GLOBAL_ROTATION else smooth_interpolate(
                            start_val * np.pi / 180,
                            end_val * np.pi / 180,
                            t
                        )
                    else:
                        current_rad = start_val * np.pi / 180 + (end_val - start_val) * np.pi / 180 * t
                    
                    if idx == GLOBAL_ROTATION:
                        axis = JOINT_AXIS_MAP[idx]
                        current_pose[0, axis] = current_rad
                        current_pose[0, 0 if axis != 0 else 1] = 0.0
                        current_pose[0, 2 if axis != 2 else 1] = 0.0
                    else:
                        pose_start_idx = 3 + idx * 3
                        axis = JOINT_AXIS_MAP.get(idx, 0)
                        if 0 <= pose_start_idx + axis < 156:
                            current_pose[0, pose_start_idx] = 0.0
                            current_pose[0, pose_start_idx + 1] = 0.0
                            current_pose[0, pose_start_idx + 2] = 0.0
                            current_pose[0, pose_start_idx + axis] = current_rad
                
                shape_params = current_shape.clone()
                pose_params = current_pose.clone()
                
                if frame_idx % max(1, total_frames // 10) == 0:
                    progress = int(t * 100)
                    self.progress_update.emit(progress, f"渲染帧 {frame_idx + 1}/{total_frames}")
                
                self._render_frame(frame_idx)
            
            self.progress_update.emit(100, "完成!")
            self.finished_signal.emit(self.output_path)
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(f"渲染失败: {str(e)}")
    
    def _render_frame(self, frame_idx):
        global shape_params, pose_params, body_model
        try:
            fig = Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(0, 2)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(f"Frame {frame_idx + 1}")
            
            global current_view_elev, current_view_azim, current_view_dist
            ax.view_init(elev=current_view_elev, azim=current_view_azim)
            if current_view_dist is not None:
                ax.dist = current_view_dist
            
            if body_model is None:
                ax.text(0, 0, 1, "模型未加载", ha="center", va="center", fontsize=14)
            else:
                body_output = body_model(
                    betas=shape_params,
                    body_pose=pose_params[:, 3:66],
                    global_orient=pose_params[:, 0:3],
                    left_hand_pose=pose_params[:, 66:111],
                    right_hand_pose=pose_params[:, 111:],
                )
                vertices = body_output.vertices.detach().cpu().numpy()[0]
                faces = body_model.faces
                ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                               triangles=faces, alpha=0.7, color="#4682B4", linewidth=0, antialiased=True)
                joints = body_output.joints.detach().cpu().numpy()[0]
                ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='red', s=15, alpha=1.0)
                core_joint_ids = [2, 3, 5, 8, 11, 17, 19]
                for jid in core_joint_ids:
                    ax.text(joints[jid, 0], joints[jid, 1], joints[jid, 2], f'{jid}', fontsize=8, color='yellow')
            
            output_file = os.path.join(self.output_path, f"frame_{frame_idx:04d}.png")
            fig.savefig(output_file, dpi=100, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"渲染帧 {frame_idx} 失败: {e}")


# ====================== 主窗口类 ======================
class HumanAnimationSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SMPL-X 3D人体动画控制与动画生成系统 - 视角增强版")
        self.setGeometry(100, 100, 1500, 950)
        self.setMinimumSize(1100, 750)
        self.generate_btn = None
        self.animation_thread = None
        self.view_saved_count = 0
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # ====================== 左侧3D视图区域 ======================
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)
        
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self._init_axes()
        self.canvas = FigureCanvas(self.fig)
        left_layout.addWidget(self.canvas, 7)
        
        # 视角状态显示
        self.view_status_label = QLabel(f"视角: 默认 (elev={DEFAULT_ELEV}°, azim={DEFAULT_AZIM}°)")
        self.view_status_label.setAlignment(Qt.AlignCenter)
        self.view_status_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ccc; padding: 5px; }")
        left_layout.addWidget(self.view_status_label)
        
        # 手动视角调整滑条
        view_ctrl_group = QGroupBox("手动视角调整")
        view_ctrl_layout = QHBoxLayout(view_ctrl_group)
        view_ctrl_layout.setContentsMargins(5, 5, 5, 5)
        
        view_ctrl_layout.addWidget(QLabel("俯仰:"))
        self.elev_slider = QSlider(Qt.Horizontal)
        self.elev_slider.setRange(-90, 90)
        self.elev_slider.setValue(DEFAULT_ELEV)
        self.elev_slider.setFixedHeight(20)
        self.elev_slider.valueChanged.connect(self._on_view_change)
        view_ctrl_layout.addWidget(self.elev_slider)
        
        view_ctrl_layout.addWidget(QLabel("  水平:"))
        self.azim_slider = QSlider(Qt.Horizontal)
        self.azim_slider.setRange(-180, 180)
        self.azim_slider.setValue(DEFAULT_AZIM)
        self.azim_slider.setFixedHeight(20)
        self.azim_slider.valueChanged.connect(self._on_view_change)
        view_ctrl_layout.addWidget(self.azim_slider)
        
        view_ctrl_layout.addWidget(QLabel("  距离:"))
        self.dist_slider = QSlider(Qt.Horizontal)
        self.dist_slider.setRange(50, 200)
        self.dist_slider.setValue(int(DEFAULT_DIST))
        self.dist_slider.setFixedHeight(20)
        self.dist_slider.valueChanged.connect(self._on_view_change)
        view_ctrl_layout.addWidget(self.dist_slider)
        
        left_layout.addWidget(view_ctrl_group)
        
        self.status_label = QLabel("状态: 等待加载模型")
        self.status_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.status_label, 0)
        main_layout.addWidget(left_container, 6)
        
        # ====================== 右侧控制面板 ======================
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_container = QWidget()
        right_container.setMinimumWidth(500)
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)
        
        self.tab_widget = QTabWidget()
        
        self.tab_single = QWidget()
        self._setup_single_frame_tab()
        self.tab_widget.addTab(self.tab_single, "单帧控制")
        
        self.tab_animation = QWidget()
        self._setup_animation_tab()
        self.tab_widget.addTab(self.tab_animation, "动画生成")
        
        self.tab_view = QWidget()
        self._setup_view_tab()
        self.tab_widget.addTab(self.tab_view, "视角控制")
        
        self.tab_index = QWidget()
        self._setup_index_tab()
        self.tab_widget.addTab(self.tab_index, "关节索引")
        
        right_layout.addWidget(self.tab_widget)
        right_scroll.setWidget(right_container)
        main_layout.addWidget(right_scroll, 4)
        
        self._draw_empty_hint()
    
    def _init_axes(self):
        """初始化坐标轴，设置默认视角"""
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(0, 2)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("SMPL-X")
        # 使用全局默认视角参数
        self.ax.view_init(elev=DEFAULT_ELEV, azim=DEFAULT_AZIM)
        # 设置视角距离，确保加载模型后不会自动缩放
        self.ax.dist = DEFAULT_DIST
    
    def _on_view_change(self, value=None):
        """视角滑条变化处理"""
        global current_view_elev, current_view_azim, current_view_dist
        current_view_elev = self.elev_slider.value()
        current_view_azim = self.azim_slider.value()
        current_view_dist = self.dist_slider.value()  # 滑条值直接作为距离
        
        elev_str = f"{current_view_elev}°"
        azim_str = f"{current_view_azim}°"
        dist_str = f"{current_view_dist}"
        self.view_status_label.setText(f"视角: elev={elev_str}, azim={azim_str}, dist={dist_str}")
        
        self._update_render()
    
    def _set_view(self, elev, azim, dist=None):
        """设置视角"""
        global current_view_elev, current_view_azim, current_view_dist
        
        current_view_elev = elev
        current_view_azim = azim
        if dist is not None:
            current_view_dist = dist
        
        # 更新滑条
        self.elev_slider.blockSignals(True)
        self.azim_slider.blockSignals(True)
        self.dist_slider.blockSignals(True)
        
        self.elev_slider.setValue(int(elev))
        self.azim_slider.setValue(int(azim))
        if dist is not None:
            self.dist_slider.setValue(int(dist))
        
        self.elev_slider.blockSignals(False)
        self.azim_slider.blockSignals(False)
        self.dist_slider.blockSignals(False)
        
        self.view_status_label.setText(f"视角: elev={elev}°, azim={azim}°, dist={int(dist) if dist else DEFAULT_DIST}")
        
        self._update_render()
    
    def _reset_view(self):
        """重置视角到默认值"""
        self._set_view(DEFAULT_ELEV, DEFAULT_AZIM, DEFAULT_DIST)
    
    def _save_current_view(self):
        """保存当前视角"""
        view_name, ok = QInputDialog.getText(
            self, "保存视角", "请输入视角名称:",
            QLineEdit.Normal, f"视角{self.view_saved_count + 1}"
        )
        
        if ok and view_name.strip():
            view_name = view_name.strip()
            global saved_views, current_view_elev, current_view_azim, current_view_dist
            
            saved_views[view_name] = {
                'elev': current_view_elev,
                'azim': current_view_azim,
                'dist': current_view_dist if current_view_dist else DEFAULT_DIST,
                'timestamp': len(saved_views)
            }
            
            self.view_saved_count += 1
            self._refresh_saved_views_list()
            self.status_label.setText(f"视角 '{view_name}' 已保存")
    
    def _load_saved_view(self, view_name):
        """加载保存的视角"""
        if view_name not in saved_views:
            return
        
        view = saved_views[view_name]
        self._set_view(view['elev'], view['azim'], view['dist'])
        self.status_label.setText(f"视角 '{view_name}' 已加载")
    
    def _delete_saved_view(self, view_name):
        """删除保存的视角"""
        if view_name in saved_views:
            del saved_views[view_name]
            self._refresh_saved_views_list()
            self.status_label.setText(f"视角 '{view_name}' 已删除")
    
    def _refresh_saved_views_list(self):
        """刷新保存视角列表"""
        self.saved_views_list.clear()
        for name in sorted(saved_views.keys(), key=lambda x: saved_views[x]['timestamp']):
            item = QListWidgetItem(name)
            item.setToolTip(f"elev={saved_views[name]['elev']}°, azim={saved_views[name]['azim']}°, dist={saved_views[name]['dist']}")
            self.saved_views_list.addItem(item)
    
    def _setup_view_tab(self):
        """设置视角控制选项卡"""
        layout = QVBoxLayout(self.tab_view)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # ====================== 视角预设区域 ======================
        preset_group = QGroupBox("视角预设")
        preset_layout = QGridLayout(preset_group)
        preset_layout.setContentsMargins(5, 5, 5, 5)
        preset_layout.setSpacing(5)
        
        # 预设按钮排列：2行4列
        preset_names = ["正前", "正后", "正左", "正右", "俯视", "仰视", "默认"]
        for i, name in enumerate(preset_names):
            row, col = i // 4, i % 4
            btn = QPushButton(name)
            btn.setFixedHeight(35)
            if name in VIEW_PRESETS:
                btn.setToolTip(VIEW_PRESETS[name]['desc'])
            btn.clicked.connect(lambda checked, n=name: self._apply_preset_view(n))
            preset_layout.addWidget(btn, row, col)
        
        # 提示信息
        preset_hint = QLabel("💡 点击按钮快速切换标准视角")
        preset_hint.setWordWrap(True)
        preset_hint.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        preset_layout.addWidget(preset_hint, 2, 0, 1, 4)
        
        layout.addWidget(preset_group)
        
        # ====================== 视角保存/加载区域 ======================
        save_load_group = QGroupBox("视角保存 / 加载")
        save_load_layout = QVBoxLayout(save_load_group)
        save_load_layout.setContentsMargins(5, 5, 5, 5)
        save_load_layout.setSpacing(5)
        
        # 按钮行
        btn_row = QHBoxLayout()
        save_btn = QPushButton("💾 保存当前视角")
        save_btn.setFixedHeight(35)
        save_btn.setStyleSheet("QPushButton { background-color: #3498db; color: white; }")
        save_btn.clicked.connect(self._save_current_view)
        
        clear_btn = QPushButton("🗑️ 清空全部")
        clear_btn.setFixedHeight(35)
        clear_btn.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; }")
        clear_btn.clicked.connect(self._clear_all_views)
        
        btn_row.addWidget(save_btn)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        
        save_load_layout.addLayout(btn_row)
        
        # 已保存列表
        list_label = QLabel("已保存的视角:")
        save_load_layout.addWidget(list_label)
        
        self.saved_views_list = QListWidget()
        self.saved_views_list.setFixedHeight(150)
        self.saved_views_list.setSelectionMode(QListWidget.SingleSelection)
        self.saved_views_list.itemClicked.connect(
            lambda item: self._load_saved_view(item.text())
        )
        save_load_layout.addWidget(self.saved_views_list)
        
        # 列表操作按钮
        list_btn_row = QHBoxLayout()
        load_selected_btn = QPushButton("加载选中")
        load_selected_btn.setFixedHeight(30)
        load_selected_btn.clicked.connect(self._load_selected_view)
        
        delete_selected_btn = QPushButton("删除选中")
        delete_selected_btn.setFixedHeight(30)
        delete_selected_btn.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; }")
        delete_selected_btn.clicked.connect(self._delete_selected_view)
        
        list_btn_row.addWidget(load_selected_btn)
        list_btn_row.addWidget(delete_selected_btn)
        list_btn_row.addStretch()
        save_load_layout.addLayout(list_btn_row)
        
        layout.addWidget(save_load_group)
        
        # 添加伸缩
        layout.addStretch()
    
    def _apply_preset_view(self, preset_name):
        """应用预设视角"""
        if preset_name not in VIEW_PRESETS:
            return
        
        preset = VIEW_PRESETS[preset_name]
        self._set_view(preset['elev'], preset['azim'], DEFAULT_DIST)
        self.status_label.setText(f"已切换到预设视角: {preset_name}")
    
    def _load_selected_view(self):
        """加载选中的视角"""
        selected = self.saved_views_list.selectedItems()
        if selected:
            self._load_saved_view(selected[0].text())
    
    def _delete_selected_view(self):
        """删除选中的视角"""
        selected = self.saved_views_list.selectedItems()
        if selected:
            view_name = selected[0].text()
            reply = QMessageBox.question(
                self, "确认", f"确定要删除视角 '{view_name}' 吗?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._delete_saved_view(view_name)
    
    def _clear_all_views(self):
        """清空所有保存的视角"""
        global saved_views
        if not saved_views:
            return
        
        reply = QMessageBox.question(
            self, "确认", "确定要清空所有保存的视角吗?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            saved_views.clear()
            self.view_saved_count = 0
            self._refresh_saved_views_list()
            self.status_label.setText("已清空所有视角")
    
    def _setup_single_frame_tab(self):
        """单帧控制 + 精准关节映射"""
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
        
        joint_group = QGroupBox("核心关节")
        joint_layout = QGridLayout(joint_group)
        joint_layout.setContentsMargins(5, 5, 5, 5)
        joint_layout.setSpacing(3)
        
        self.core_joints = [
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
            slider.valueChanged.connect(lambda v, id=idx: self._update_joint(v, id))
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
        layout = QVBoxLayout(self.tab_animation)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        dir_group = QGroupBox("输出设置")
        dir_layout = QFormLayout()
        dir_layout.setContentsMargins(5, 5, 5, 5)
        dir_layout.setSpacing(5)
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
        
        self.frame_count = QSpinBox()
        self.frame_count.setRange(1, 500)
        self.frame_count.setValue(30)
        self.frame_count.setFixedHeight(30)
        dir_layout.addRow(QLabel("帧数:"), self.frame_count)
        layout.addWidget(dir_group)
        
        # 插值算法选择
        interp_group = QGroupBox("动画插值算法")
        interp_layout = QHBoxLayout(interp_group)
        interp_layout.setContentsMargins(5, 5, 5, 5)
        
        self.interp_button_group = QButtonGroup()
        linear_radio = QRadioButton("线性插值")
        linear_radio.setChecked(True)
        smooth_radio = QRadioButton("平滑插值")
        self.interp_button_group.addButton(linear_radio, 0)
        self.interp_button_group.addButton(smooth_radio, 1)
        
        interp_layout.addWidget(linear_radio)
        interp_layout.addWidget(smooth_radio)
        interp_layout.addStretch()
        
        interp_hint = QLabel("线性：匀速变化 | 平滑：缓入缓出效果")
        interp_hint.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        interp_layout.addWidget(interp_hint)
        
        layout.addWidget(interp_group)
        
        anim_group = QGroupBox("动画参数")
        anim_layout = QVBoxLayout(anim_group)
        anim_layout.setContentsMargins(5, 5, 5, 5)
        anim_layout.setSpacing(3)
        hint = QLabel("勾选需要动画的参数并设置开始/结束值:")
        hint.setWordWrap(True)
        hint.setStyleSheet("QLabel { color: #666; }")
        anim_layout.addWidget(hint)
        
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
        
        self.anim_joint_widgets = {}
        for name, idx, val in self.core_joints:
            joint_hbox = QHBoxLayout()
            joint_hbox.setContentsMargins(0, 0, 0, 0)
            checkbox = QCheckBox()
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
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(25)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)
        self.anim_status_label = QLabel("就绪")
        self.anim_status_label.setAlignment(Qt.AlignCenter)
        self.anim_status_label.setFixedHeight(25)
        self.anim_status_label.setFrameStyle(QFrame.StyledPanel)
        layout.addWidget(self.anim_status_label)
        
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
        layout = QVBoxLayout(self.tab_index)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        info_group = QGroupBox("SMPLX关节核心规则")
        info_layout = QVBoxLayout(info_group)
        info_layout.setContentsMargins(5, 5, 5, 5)
        info_text = QLabel(
            "每个关节占3个连续维度 → pose_params[3+ID*3 : 3+ID*3+3]\n"
            "下肢关节(髋/膝/脚) 只改 Z轴(+2)\n"
            "躯干关节(腰/胸/颈) 只改 Y轴(+1)\n"
            "上肢关节(肩/肘/腕) 只改 X轴(+0)\n"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("QLabel { color: #d63031; font-weight:bold; }")
        info_layout.addWidget(info_text)
        layout.addWidget(info_group)
        
        ref_group = QGroupBox("常用关节精准索引速查表")
        ref_layout = QVBoxLayout(ref_group)
        ref_text = QLabel(
            "关节名称          ID    pose起始位  旋转轴  运动效果\n"
            "──────────────────────────────────────────────────\n"
            f"全局旋转          global  1(Y)      Y      水平旋转\n"
            f"腰腹核心          spine1  3→12      Y      弯腰/扭腰 ✔️\n"
            f"右髋关节          right_hip 2→9     Z      抬腿/扭胯 ✔️\n"
            f"右膝关节          right_knee5→18    Z      屈膝/伸膝 ✔️\n"
            f"右脚掌            right_foot11→36   Z      脚面旋转 ✔️\n"
            f"右肩关节          right_shoulder17→54 X    抬肩/压肩 ✔️\n"
            f"右肘关节          right_elbow19→60   X    屈肘/伸肘 ✔️\n"
            f"胸椎              spine2  6→21     Y      挺胸/含胸\n"
        )
        ref_text.setFont(QFont("Monospace", 9))
        ref_text.setWordWrap(False)
        ref_layout.addWidget(ref_text)
        layout.addWidget(ref_group)
        
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
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录", "./", QFileDialog.ShowDirsOnly)
        if directory:
            self.output_dir_edit.setText(directory)
    
    def _load_smplx_model(self):
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
                        
                        if hasattr(body_model, 'joint_mapper'):
                            mapper = body_model.joint_mapper
                            mapper_info = "关节名称 -> ID -> pose起始位 -> 核心轴:\n"
                            mapper_info += "-" * 70 + "\n"
                            for name in sorted(mapper.keys(), key=lambda x: mapper[x]):
                                idx = mapper[name]
                                pose_idx = 3 + idx * 3
                                axis = JOINT_AXIS_MAP.get(idx, 0)
                                axis_name = {0: 'X', 1: 'Y', 2: 'Z'}[axis]
                                mapper_info += f"  {name:20s} -> {idx:2d} -> {pose_idx:2d} -> {axis_name}\n"
                            self.mapper_text.setText(mapper_info)
                        break
                    except Exception as e:
                        print(f"尝试 {model_path} 失败: {e}")
                        continue
            
            if not model_loaded:
                model_path = QFileDialog.getExistingDirectory(self, "选择SMPLX模型目录", "./", QFileDialog.ShowDirsOnly)
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
                self.status_label.setText("状态: 模型就绪")
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
        global shape_params
        shape_params[0, 0] = value
        self.shape_label.setText(str(value))
        self._update_render()
    
    def _update_joint(self, value, idx):
        global pose_params
        rad = value * np.pi / 180
        if idx == GLOBAL_ROTATION:
            pose_params[0, 0] = 0.0
            pose_params[0, 1] = rad
            pose_params[0, 2] = 0.0
        else:
            pose_start_idx = 3 + idx * 3
            axis = JOINT_AXIS_MAP.get(idx, 0)
            if 0 <= pose_start_idx + axis < 156:
                pose_params[0, pose_start_idx] = 0.0
                pose_params[0, pose_start_idx + 1] = 0.0
                pose_params[0, pose_start_idx + 2] = 0.0
                pose_params[0, pose_start_idx + axis] = rad
        
        if idx in self.core_labels:
            self.core_labels[idx].setText(f"{value}°")
        self._update_render()
    
    def _reset_all(self):
        """重置所有参数，包括视角"""
        global shape_params, pose_params
        shape_params = torch.zeros(1, 10, device=device)
        pose_params = torch.zeros(1, 156, device=device)
        self.shape_slider.setValue(0)
        self.shape_label.setText("0")
        for idx in self.core_sliders:
            self.core_sliders[idx].setValue(0)
            self.core_labels[idx].setText("0°")
        # 同时重置视角到默认值
        self._reset_view()
        self._update_render()
        self.status_label.setText("状态: 已重置所有参数和视角")
    
    def _update_render(self):
        global shape_params, pose_params, body_model
        self.ax.clear()
        self._init_axes()
        
        global current_view_elev, current_view_azim, current_view_dist
        self.ax.view_init(elev=current_view_elev, azim=current_view_azim)
        if current_view_dist is not None:
            self.ax.dist = current_view_dist
        
        if body_model is None:
            self.ax.text(0, 0, 1, "please load SMPLX model", ha="center", va="center", fontsize=14, color='red')
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
            self.ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                                triangles=faces, alpha=0.7, color="#4682B4", linewidth=0, antialiased=True)
            joints = body_output.joints.detach().cpu().numpy()[0]
            self.ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='red', s=20, alpha=1.0, label='joints')
            focus_joints = {3: '腰', 2: '右髋', 5: '右膝', 11: '右脚', 17: '右肩'}
            for jid, name in focus_joints.items():
                self.ax.text(joints[jid, 0], joints[jid, 1], joints[jid, 2], f'{name}\n{jid}', fontsize=9, color='yellow', ha='center')
            self.ax.legend(loc='upper right')
            self.status_label.setText("状态: 渲染完成")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.ax.text(0, 0, 1, f"渲染错误: {e}", ha="center", va="center", fontsize=10, color='red')
        self.canvas.draw()
    
    def _draw_empty_hint(self):
        self.ax.clear()
        self._init_axes()
        self.ax.text(0, 0, 1, "please load SMPLX model", ha="center", va="center", fontsize=14, color='red')
        self.canvas.draw()
    
    def _generate_animation(self):
        global body_model
        if body_model is None:
            QMessageBox.warning(self, "警告", "请先加载SMPLX模型!")
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
                self, "确认", "没有选择任何关节动画，是否只生成体型动画?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        selected_id = self.interp_button_group.checkedId()
        interpolation = "linear" if selected_id == 0 else "smooth"
        
        self.animation_thread = AnimationWorker(frames, output_path, interpolation=interpolation)
        self.animation_thread.set_params(shape_start, shape_end, joint_configs)
        self.animation_thread.progress_update.connect(self._on_animation_progress)
        self.animation_thread.finished_signal.connect(self._on_animation_finished)
        self.animation_thread.error_signal.connect(self._on_animation_error)
        
        if self.generate_btn:
            self.generate_btn.setEnabled(False)
        self.animation_thread.start()
        self.status_label.setText("状态: 动画生成中")
    
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
            self, "完成", f"动画帧已保存到:\n{output_path}\n是否打开文件夹?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            if sys.platform == 'win32':
                os.startfile(output_path)
            elif sys.platform == 'darwin':
                os.system(f'open "{output_path}"')
            else:
                os.system(f'xdg-open "{output_path}"')
        self.status_label.setText("状态: 动画保存")
    
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
        print("=" * 70)
        print("SMPL-X 3D人体动画控制系统 - 视角增强版")
        print("新增功能: 视角预设 / 视角保存 / 平滑插值")
        print("=" * 70)
        sys.exit(app.exec_())
    except Exception as e:
        import traceback
        print(f"程序错误: {e}")
        traceback.print_exc()
        sys.exit(1)
