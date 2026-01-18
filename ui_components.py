#!/usr/bin/env python3
"""
UI组件封装（视角控制、关节控制、标签页等）
"""
from PyQt5.QtWidgets import (
    QGroupBox, QGridLayout, QHBoxLayout, QVBoxLayout, QPushButton,
    QLabel, QSlider, QListWidget, QInputDialog, QMessageBox, QSpinBox,
    QLineEdit, QFormLayout, QCheckBox, QRadioButton, QButtonGroup,
    QTextEdit, QFrame, QFont
)
from PyQt5.QtCore import Qt

from .constants import (
    DEFAULT_ELEV, DEFAULT_AZIM, DEFAULT_DIST, VIEW_PRESETS,
    CORE_JOINTS, SMPLX_JOINTS
)

class ViewControlComponent:
    """视角控制组件"""
    def __init__(self, parent):
        self.parent = parent  # 主窗口实例
        self.view_saved_count = 0
        self.saved_views_list = None
    
    def setup_view_tab(self, tab_view):
        """设置视角控制标签页"""
        layout = QVBoxLayout(tab_view)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # 视角预设区域
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
            btn.clicked.connect(lambda checked, n=name: self.parent._apply_preset_view(n))
            preset_layout.addWidget(btn, row, col)
        
        # 提示信息
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
        
        # 按钮行
        btn_row = QHBoxLayout()
        save_btn = QPushButton("💾 保存当前视角")
        save_btn.setFixedHeight(35)
        save_btn.setStyleSheet("QPushButton { background-color: #3498db; color: white; }")
        save_btn.clicked.connect(self.parent._save_current_view)
        
        clear_btn = QPushButton("🗑️ 清空全部")
        clear_btn.setFixedHeight(35)
        clear_btn.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; }")
        clear_btn.clicked.connect(self.parent._clear_all_views)
        
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
            lambda item: self.parent._load_saved_view(item.text())
        )
        save_load_layout.addWidget(self.saved_views_list)
        
        # 列表操作按钮
        list_btn_row = QHBoxLayout()
        load_selected_btn = QPushButton("加载选中")
        load_selected_btn.setFixedHeight(30)
        load_selected_btn.clicked.connect(self.parent._load_selected_view)
        
        delete_selected_btn = QPushButton("删除选中")
        delete_selected_btn.setFixedHeight(30)
        delete_selected_btn.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; }")
        delete_selected_btn.clicked.connect(self.parent._delete_selected_view)
        
        list_btn_row.addWidget(load_selected_btn)
        list_btn_row.addWidget(delete_selected_btn)
        list_btn_row.addStretch()
        save_load_layout.addLayout(list_btn_row)
        layout.addWidget(save_load_group)
        layout.addStretch()
    
    def refresh_saved_views_list(self, saved_views):
        """刷新保存的视角列表"""
        self.saved_views_list.clear()
        for name in sorted(saved_views.keys(), key=lambda x: saved_views[x]['timestamp']):
            item = QListWidgetItem(name)
            item.setToolTip(f"elev={saved_views[name]['elev']}°, azim={saved_views[name]['azim']}°, dist={saved_views[name]['dist']}")
            self.saved_views_list.addItem(item)

class SingleFrameComponent:
    """单帧控制组件"""
    def __init__(self, parent):
        self.parent = parent
        self.core_sliders = {}
        self.core_labels = {}
    
    def setup_single_frame_tab(self, tab_single):
        """设置单帧控制标签页"""
        layout = QVBoxLayout(tab_single)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # 模型加载
        load_group = QGroupBox("模型加载")
        load_layout = QHBoxLayout(load_group)
        load_layout.setContentsMargins(5, 5, 5, 5)
        self.parent.load_btn = QPushButton("加载模型")
        self.parent.load_btn.setFixedHeight(30)
        self.parent.load_btn.clicked.connect(self.parent._load_smplx_model)
        self.parent.model_label = QLabel("未加载")
        self.parent.model_label.setWordWrap(True)
        self.parent.model_label.setFrameStyle(QFrame.StyledPanel)
        load_layout.addWidget(self.parent.load_btn, 1)
        load_layout.addWidget(self.parent.model_label, 2)
        layout.addWidget(load_group)
        
        # 体型
        shape_group = QGroupBox("体型参数 β₀")
        shape_layout = QHBoxLayout(shape_group)
        shape_layout.setContentsMargins(5, 5, 5, 5)
        self.parent.shape_slider = QSlider(Qt.Horizontal)
        self.parent.shape_slider.setRange(-5, 5)
        self.parent.shape_slider.setValue(0)
        self.parent.shape_slider.setFixedHeight(20)
        self.parent.shape_slider.valueChanged.connect(self.parent._update_shape)
        self.parent.shape_label = QLabel("0")
        self.parent.shape_label.setFixedWidth(30)
        shape_layout.addWidget(QLabel("β₀:"))
        shape_layout.addWidget(self.parent.shape_slider, 1)
        shape_layout.addWidget(self.parent.shape_label)
        layout.addWidget(shape_group)
        
        # 核心关节控制
        joint_group = QGroupBox("核心关节")
        joint_layout = QGridLayout(joint_group)
        joint_layout.setContentsMargins(5, 5, 5, 5)
        joint_layout.setSpacing(3)
        
        for i, (name, idx, val) in enumerate(CORE_JOINTS):
            row, col = i // 2, (i % 2) * 3
            name_label = QLabel(f"{name}:")
            name_label.setFixedWidth(45)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-90, 90)
            slider.setValue(val)
            slider.setFixedHeight(18)
            slider.valueChanged.connect(lambda v, id=idx: self.parent._update_joint(v, id))
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
        reset_btn.clicked.connect(self.parent._reset_all)
        layout.addWidget(reset_btn)
        layout.addStretch()
    
    def update_joint_label(self, idx, value):
        """更新关节标签显示"""
        if idx in self.core_labels:
            self.core_labels[idx].setText(f"{value}°")

class AnimationComponent:
    """动画生成组件"""
    def __init__(self, parent):
        self.parent = parent
        self.anim_joint_widgets = {}
    
    def setup_animation_tab(self, tab_animation):
        """设置动画生成标签页"""
        layout = QVBoxLayout(tab_animation)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # 输出设置
        dir_group = QGroupBox("输出设置")
        dir_layout = QFormLayout()
        dir_layout.setContentsMargins(5, 5, 5, 5)
        dir_layout.setSpacing(5)
        dir_hbox = QHBoxLayout()
        self.parent.output_dir_edit = QLineEdit("./output_frames")
        self.parent.output_dir_edit.setFixedHeight(30)
        self.parent.output_dir_edit.setPlaceholderText("输入输出目录路径")
        dir_btn = QPushButton("浏览")
        dir_btn.setFixedSize(60, 30)
        dir_btn.clicked.connect(self.parent._browse_output_dir)
        dir_hbox.addWidget(self.parent.output_dir_edit, 1)
        dir_hbox.addWidget(dir_btn, 0)
        dir_layout.addRow(QLabel("输出目录:"), dir_hbox)
        
        self.parent.frame_count = QSpinBox()
        self.parent.frame_count.setRange(1, 500)
        self.parent.frame_count.setValue(30)
        self.parent.frame_count.setFixedHeight(30)
        dir_layout.addRow(QLabel("帧数:"), self.parent.frame_count)
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)
        
        # 插值算法选择
        interp_group = QGroupBox("动画插值算法")
        interp_layout = QHBoxLayout(interp_group)
        interp_layout.setContentsMargins(5, 5, 5, 5)
        
        self.parent.interp_button_group = QButtonGroup()
        linear_radio = QRadioButton("线性插值")
        linear_radio.setChecked(True)
        smooth_radio = QRadioButton("平滑插值")
        self.parent.interp_button_group.addButton(linear_radio, 0)
        self.parent.interp_button_group.addButton(smooth_radio, 1)
        
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
        
        # 体型参数
        shape_hbox = QHBoxLayout()
        shape_hbox.addWidget(QLabel("体型β₀:"))
        self.parent.anim_shape_start = QSpinBox()
        self.parent.anim_shape_start.setRange(-5, 5)
        self.parent.anim_shape_start.setValue(0)
        self.parent.anim_shape_start.setFixedSize(60, 30)
        shape_hbox.addWidget(self.parent.anim_shape_start)
        shape_hbox.addWidget(QLabel("→"))
        self.parent.anim_shape_end = QSpinBox()
        self.parent.anim_shape_end.setRange(-5, 5)
        self.parent.anim_shape_end.setValue(0)
        self.parent.anim_shape_end.setFixedSize(60, 30)
        shape_hbox.addWidget(self.parent.anim_shape_end)
        shape_hbox.addStretch(1)
        shape_widget = QWidget()
        shape_widget.setLayout(shape_hbox)
        shape_widget.setFixedHeight(35)
        anim_layout.addWidget(shape_widget)
        
        # 关节动画参数
        for name, idx, val in CORE_JOINTS:
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
        
        # 进度条和状态
        self.parent.progress_bar = QProgressBar()
        self.parent.progress_bar.setFixedHeight(25)
        self.parent.progress_bar.setTextVisible(True)
        layout.addWidget(self.parent.progress_bar)
        
        self.parent.anim_status_label = QLabel("就绪")
        self.parent.anim_status_label.setAlignment(Qt.AlignCenter)
        self.parent.anim_status_label.setFixedHeight(25)
        self.parent.anim_status_label.setFrameStyle(QFrame.StyledPanel)
        layout.addWidget(self.parent.anim_status_label)
        
        # 生成按钮
        self.parent.generate_btn = QPushButton("生成动画帧序列")
        self.parent.generate_btn.setFixedHeight(40)
        self.parent.generate_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666; }"
        )
        self.parent.generate_btn.clicked.connect(self.parent._generate_animation)
        layout.addWidget(self.parent.generate_btn)
        layout.addStretch()
    
    def get_anim_joint_widgets(self):
        """获取关节动画控件字典"""
        return self.anim_joint_widgets

class IndexComponent:
    """关节索引组件"""
    def __init__(self, parent):
        self.parent = parent
    
    def setup_index_tab(self, tab_index):
        """设置关节索引标签页"""
        layout = QVBoxLayout(tab_index)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # 核心规则
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
        
        # 索引速查表
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
        
        # joint_mapper 信息
        mapper_group = QGroupBox("joint_mapper 详细信息")
        mapper_layout = QVBoxLayout(mapper_group)
        self.parent.mapper_text = QTextEdit()
        self.parent.mapper_text.setReadOnly(True)
        self.parent.mapper_text.setMaximumHeight(200)
        self.parent.mapper_text.setText("请先加载模型以查看 joint_mapper")
        mapper_layout.addWidget(self.parent.mapper_text)
        layout.addWidget(mapper_group)
        layout.addStretch()
