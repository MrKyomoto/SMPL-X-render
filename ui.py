# ui.py
"""
SMPL-X 3D人体动画控制系统 - 主界面和逻辑
"""

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QSlider, QLabel, QGroupBox, QGridLayout,
    QSpinBox, QLineEdit, QProgressBar, QMessageBox,
    QTabWidget, QFormLayout, QCheckBox, QScrollArea,
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

# 导入配置模块
from config import (
    device, SMPLX_JOINTS, JOINT_AXIS_MAP, GLOBAL_ROTATION,
    DEFAULT_ELEV, DEFAULT_AZIM, DEFAULT_DIST, VIEW_PRESETS,
    body_model, shape_params, pose_params,
    current_view_elev, current_view_azim, current_view_dist, saved_views
)

# 导入动画线程
from animation_worker import AnimationWorker, set_globals

# 设置matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False


class HumanAnimationSystem(QMainWindow):
    """SMPL-X 3D人体动画控制与动画生成系统主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SMPL-X 3D人体动画控制与动画生成系统")
        self.setGeometry(100, 100, 1500, 950)
        self.setMinimumSize(1100, 750)
        self.generate_btn = None
        self.animation_thread = None
        self.view_saved_count = 0
        
        # 初始化UI
        self._init_ui()
        
        # 绘制空提示
        self._draw_empty_hint()
    
    def _init_ui(self):
        """初始化用户界面"""
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
        self.view_status_label = QLabel(
            f"视角: 默认 (elev={DEFAULT_ELEV}°, azim={DEFAULT_AZIM}°)"
        )
        self.view_status_label.setAlignment(Qt.AlignCenter)
        self.view_status_label.setStyleSheet(
            "QLabel { background-color: #f0f0f0; border: 1px solid #ccc; padding: 5px; }"
        )
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
    
    def _init_axes(self):
        """初始化坐标轴基础设置（不包含视角）"""
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(0, 2)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("SMPL-X")
        # 注意：这里不再设置默认视角，视角由调用者决定
    
    def _on_view_change(self, value=None):
        """视角滑条变化处理"""
        global current_view_elev, current_view_azim, current_view_dist
        
        current_view_elev = self.elev_slider.value()
        current_view_azim = self.azim_slider.value()
        current_view_dist = self.dist_slider.value()
        
        elev_str = f"{current_view_elev}°"
        azim_str = f"{current_view_azim}°"
        dist_str = f"{current_view_dist}"
        self.view_status_label.setText(
            f"视角: elev={elev_str}, azim={azim_str}, dist={dist_str}"
        )
        
        # 重新渲染整个场景（包括模型）
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
        
        dist_display = int(dist) if dist else DEFAULT_DIST
        self.view_status_label.setText(
            f"视角: elev={elev}°, azim={azim}°, dist={dist_display}"
        )
        
        # 重新渲染整个场景（包括模型）
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
        global saved_views
        
        if view_name not in saved_views:
            return
        
        view = saved_views[view_name]
        self._set_view(view['elev'], view['azim'], view['dist'])
        self.status_label.setText(f"视角 '{view_name}' 已加载")
    
    def _delete_saved_view(self, view_name):
        """删除保存的视角"""
        global saved_views
        
        if view_name in saved_views:
            del saved_views[view_name]
            self._refresh_saved_views_list()
            self.status_label.setText(f"视角 '{view_name}' 已删除")
    
    def _refresh_saved_views_list(self):
        """刷新保存视角列表"""
        global saved_views
        
        self.saved_views_list.clear()
        for name in sorted(saved_views.keys(), key=lambda x: saved_views[x]['timestamp']):
            item = QListWidgetItem(name)
            tooltip = (
                f"elev={saved_views[name]['elev']}°, "
                f"azim={saved_views[name]['azim']}°, "
                f"dist={saved_views[name]['dist']}"
            )
            item.setToolTip(tooltip)
            self.saved_views_list.addItem(item)
    
    def _setup_view_tab(self):
        """设置视角控制选项卡"""
        layout = QVBoxLayout(self.tab_view)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # 视角预设区域
        preset_group = QGroupBox("视角预设")
        preset_layout = QGridLayout(preset_group)
        preset_layout.setContentsMargins(5, 5, 5, 5)
        preset_layout.setSpacing(5)
        
        preset_names = ["正前", "正后", "正左", "正右", "俯视", "仰视", "默认"]
        for i, name in enumerate(preset_names):
            row, col = i // 4, i % 4
            btn = QPushButton(name)
            btn.setFixedHeight(35)
            if name in VIEW_PRESETS:
                btn.setToolTip(VIEW_PRESETS[name]['desc'])
            btn.clicked.connect(lambda checked, n=name: self._apply_preset_view(n))
            preset_layout.addWidget(btn, row, col)
        
        preset_hint = QLabel("💡 点击按钮快速切换标准视角")
        preset_hint.setWordWrap(True)
        preset_hint.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        preset_layout.addWidget(preset_hint, 2, 0, 1, 4)
        
        layout.addWidget(preset_group)
        
        # 视角保存/加载区域
        save_load_group = QGroupBox("视角保存 / 加载")
        save_load_layout = QVBoxLayout(save_load_group)
        save_load_layout.setContentsMargins(5, 5, 5, 5)
        save_load_layout.setSpacing(5)
        
        btn_row = QHBoxLayout()
        save_btn = QPushButton("💾 保存当前视角")
        save_btn.setFixedHeight(35)
        save_btn.setStyleSheet(
            "QPushButton { background-color: #3498db; color: white; }"
        )
        save_btn.clicked.connect(self._save_current_view)
        
        clear_btn = QPushButton("🗑️ 清空全部")
        clear_btn.setFixedHeight(35)
        clear_btn.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; }"
        )
        clear_btn.clicked.connect(self._clear_all_views)
        
        btn_row.addWidget(save_btn)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        
        save_load_layout.addLayout(btn_row)
        
        list_label = QLabel("已保存的视角:")
        save_load_layout.addWidget(list_label)
        
        self.saved_views_list = QListWidget()
        self.saved_views_list.setFixedHeight(150)
        self.saved_views_list.setSelectionMode(QListWidget.SingleSelection)
        self.saved_views_list.itemClicked.connect(
            lambda item: self._load_saved_view(item.text())
        )
        save_load_layout.addWidget(self.saved_views_list)
        
        list_btn_row = QHBoxLayout()
        load_selected_btn = QPushButton("加载选中")
        load_selected_btn.setFixedHeight(30)
        load_selected_btn.clicked.connect(self._load_selected_view)
        
        delete_selected_btn = QPushButton("删除选中")
        delete_selected_btn.setFixedHeight(30)
        delete_selected_btn.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; }"
        )
        delete_selected_btn.clicked.connect(self._delete_selected_view)
        
        list_btn_row.addWidget(load_selected_btn)
        list_btn_row.addWidget(delete_selected_btn)
        list_btn_row.addStretch()
        save_load_layout.addLayout(list_btn_row)
        
        layout.addWidget(save_load_group)
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
        
        # 体型参数
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
        
        # 核心关节
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
            slider.valueChanged.connect(
                lambda v, id=idx: self._update_joint(v, id)
            )
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
        """设置动画生成选项卡"""
        layout = QVBoxLayout(self.tab_animation)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # 输出设置
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
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; font-size: 14px; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666; }"
        )
        self.generate_btn.clicked.connect(self._generate_animation)
        layout.addWidget(self.generate_btn)
        layout.addStretch()
    
    def _setup_index_tab(self):
        """设置关节索引选项卡"""
        from PyQt5.QtGui import QFont
        
        layout = QVBoxLayout(self.tab_index)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # SMPLX关节核心规则
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
        
        # 常用关节精准索引速查表
        ref_group = QGroupBox("常用关节精准索引速查表")
        ref_layout = QVBoxLayout(ref_group)
        ref_text = QLabel(
            "关节名称          ID    pose起始位  旋转轴  运动效果\n"
            "────────────────────────────────────────────────--\n"
            "全局旋转          global  1(Y)      Y      水平旋转\n"
            "腰腹核心          spine1  3→12      Y      弯腰/扭腰 ✔️\n"
            "右髋关节          right_hip 2→9     Z      抬腿/扭胯 ✔️\n"
            "右膝关节          right_knee5→18    Z      屈膝/伸膝 ✔️\n"
            "右脚掌            right_foot11→36   Z      脚面旋转 ✔️\n"
            "右肩关节          right_shoulder17→54 X    抬肩/压肩 ✔️\n"
            "右肘关节          right_elbow19→60   X    屈肘/伸肘 ✔️\n"
            "胸椎              spine2  6→21     Y      挺胸/含胸\n"
        )
        ref_text.setFont(QFont("Monospace", 9))
        ref_text.setWordWrap(False)
        ref_layout.addWidget(ref_text)
        layout.addWidget(ref_group)
        
        # joint_mapper 详细信息
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
        """浏览选择输出目录"""
        directory = QFileDialog.getExistingDirectory(
            self, "选择输出目录", "./", QFileDialog.ShowDirsOnly
        )
        if directory:
            self.output_dir_edit.setText(directory)
    
    def _load_smplx_model(self):
        """加载SMPLX模型"""
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
                            mapper_info = (
                                "关节名称 -> ID -> pose起始位 -> 核心轴:\n"
                            )
                            mapper_info += "-" * 70 + "\n"
                            for name in sorted(
                                mapper.keys(), key=lambda x: mapper[x]
                            ):
                                idx = mapper[name]
                                pose_idx = 3 + idx * 3
                                axis = JOINT_AXIS_MAP.get(idx, 0)
                                axis_name = {0: 'X', 1: 'Y', 2: 'Z'}[axis]
                                mapper_info += (
                                    f"  {name:20s} -> {idx:2d} -> "
                                    f"{pose_idx:2d} -> {axis_name}\n"
                                )
                            self.mapper_text.setText(mapper_info)
                        break
                    except Exception as e:
                        print(f"尝试 {model_path} 失败: {e}")
                        continue
            
            if not model_loaded:
                model_path = QFileDialog.getExistingDirectory(
                    self, "选择SMPLX模型目录", "./", QFileDialog.ShowDirsOnly
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
        """更新体型参数"""
        global shape_params
        
        shape_params[0, 0] = value
        self.shape_label.setText(str(value))
        self._update_render()
    
    def _update_joint(self, value, idx):
        """更新关节参数"""
        global pose_params, JOINT_AXIS_MAP
        
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
        self._reset_view()
        self._update_render()
        self.status_label.setText("状态: 已重置所有参数和视角")
    
    def _update_render(self):
        """更新渲染（包含视角设置）"""
        global body_model, shape_params, pose_params
        global current_view_elev, current_view_azim, current_view_dist
        
        # 清除并重新初始化坐标轴
        self.ax.clear()
        self._init_axes()
        
        # 在初始化之后再设置视角，确保使用当前的视角值
        self.ax.view_init(elev=current_view_elev, azim=current_view_azim)
        if current_view_dist is not None:
            self.ax.dist = current_view_dist
        
        if body_model is None:
            self.ax.text(
                0, 0, 1, "please load SMPLX model",
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
            joints = body_output.joints.detach().cpu().numpy()[0]
            self.ax.scatter(
                joints[:, 0], joints[:, 1], joints[:, 2],
                c='red', s=20, alpha=1.0, label='joints'
            )
            focus_joints = {3: '腰', 2: '右髋', 5: '右膝', 11: '右脚', 17: '右肩'}
            for jid, name in focus_joints.items():
                self.ax.text(
                    joints[jid, 0], joints[jid, 1], joints[jid, 2],
                    f'{name}\n{jid}', fontsize=9, color='yellow', ha='center'
                )
            self.ax.legend(loc='upper right')
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
        """绘制空提示"""
        global current_view_elev, current_view_azim, current_view_dist
        
        self.ax.clear()
        self._init_axes()
        # 设置初始视角
        self.ax.view_init(elev=current_view_elev, azim=current_view_azim)
        self.ax.dist = current_view_dist if current_view_dist else DEFAULT_DIST
        self.ax.text(
            0, 0, 1, "please load SMPLX model",
            ha="center", va="center", fontsize=14, color='red'
        )
        self.canvas.draw()
    
    def _generate_animation(self):
        """生成动画"""
        global body_model, shape_params, pose_params
        global current_view_elev, current_view_azim, current_view_dist
        
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
        
        # 创建动画线程
        self.animation_thread = AnimationWorker(
            frames, output_path, interpolation=interpolation
        )
        self.animation_thread.set_params(shape_start, shape_end, joint_configs)
        
        # 传递当前状态给动画线程
        self.animation_thread.set_state(shape_params, pose_params)
        
        # 设置全局变量供动画线程使用
        set_globals(
            body_model,
            current_view_elev,
            current_view_azim,
            current_view_dist
        )
        
        # 连接信号
        self.animation_thread.progress_update.connect(
            self._on_animation_progress
        )
        self.animation_thread.finished_signal.connect(
            self._on_animation_finished
        )
        self.animation_thread.error_signal.connect(
            self._on_animation_error
        )
        
        if self.generate_btn:
            self.generate_btn.setEnabled(False)
        self.animation_thread.start()
        self.status_label.setText("状态: 动画生成中")
    
    def _on_animation_progress(self, value, message):
        """动画进度回调"""
        self.progress_bar.setValue(value)
        self.anim_status_label.setText(message)
        QApplication.processEvents()
    
    def _on_animation_finished(self, output_path):
        """动画完成回调"""
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
        """动画错误回调"""
        self.anim_status_label.setText("错误!")
        if self.generate_btn:
            self.generate_btn.setEnabled(True)
        QMessageBox.critical(self, "错误", error_message)
        self.status_label.setText(f"状态: {error_message}")
