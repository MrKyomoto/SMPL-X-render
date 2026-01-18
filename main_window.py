#!/usr/bin/env python3
"""
主窗口类（整合所有UI组件，处理核心业务逻辑）
"""
import sys
import os
import torch
import smplx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QTabWidget, QFileDialog, QInputDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QThread

from .constants import (
    DEFAULT_ELEV, DEFAULT_AZIM, DEFAULT_DIST, VIEW_PRESETS,
    saved_views, current_view_elev, current_view_azim, current_view_dist,
    CORE_JOINTS, MODEL_PATHS, JOINT_AXIS_MAP, GLOBAL_ROTATION
)
from .ui_components import (
    ViewControlComponent, SingleFrameComponent,
    AnimationComponent, IndexComponent
)
from .animation_worker import AnimationWorker, update_globals

# 兼容NumPy 2.0
np.Inf = np.inf
np.NAN = np.nan
matplotlib.use('Qt5Agg')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

class HumanAnimationSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SMPL-X 3D人体动画控制与动画生成系统")
        self.setGeometry(100, 100, 1500, 950)
        self.setMinimumSize(1100, 750)
        
        # 全局变量
        self.body_model = None
        self.shape_params = torch.zeros(1, 10, device=torch.device("cpu"))
        self.pose_params = torch.zeros(1, 156, device=torch.device("cpu"))
        self.animation_thread = None
        self.view_saved_count = 0
        
        # UI组件实例
        self.view_component = ViewControlComponent(self)
        self.single_frame_component = SingleFrameComponent(self)
        self.animation_component = AnimationComponent(self)
        self.index_component = IndexComponent(self)
        
        # 初始化UI
        self._init_ui()
    
    def _init_ui(self):
        """初始化主界面"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # ====================== 左侧3D视图区域 ======================
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)
        
        # 3D画布
        self.fig = Figure(figsize=(10, 8))
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
        view_ctrl_group = QWidget()
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
        
        # 状态标签
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
        
        # 标签页
        self.tab_widget = QTabWidget()
        self.tab_single = QWidget()
        self.single_frame_component.setup_single_frame_tab(self.tab_single)
        self.tab_widget.addTab(self.tab_single, "单帧控制")
        
        self.tab_animation = QWidget()
        self.animation_component.setup_animation_tab(self.tab_animation)
        self.tab_widget.addTab(self.tab_animation, "动画生成")
        
        self.tab_view = QWidget()
        self.view_component.setup_view_tab(self.tab_view)
        self.tab_widget.addTab(self.tab_view, "视角控制")
        
        self.tab_index = QWidget()
        self.index_component.setup_index_tab(self.tab_index)
        self.tab_widget.addTab(self.tab_index, "关节索引")
        
        right_layout.addWidget(self.tab_widget)
        right_scroll.setWidget(right_container)
        main_layout.addWidget(right_scroll, 4)
        
        # 初始绘制空提示
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
        self.ax.view_init(elev=DEFAULT_ELEV, azim=DEFAULT_AZIM)
        self.ax.dist = DEFAULT_DIST
    
    def _draw_empty_hint(self):
        """绘制空模型提示"""
        self.ax.clear()
        self._init_axes()
        self.ax.text(0, 0, 1, "模型未加载", ha="center", va="center", fontsize=14)
        self.canvas.draw()
    
    def _on_view_change(self, value=None):
        """视角滑条变化处理"""
        global current_view_elev, current_view_azim, current_view_dist
        current_view_elev = self.elev_slider.value()
        current_view_azim = self.azim_slider.value()
        current_view_dist = self.dist_slider.value()
        
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
    
    def _apply_preset_view(self, preset_name):
        """应用预设视角"""
        if preset_name not in VIEW_PRESETS:
            return
        preset = VIEW_PRESETS[preset_name]
        self._set_view(preset['elev'], preset['azim'], DEFAULT_DIST)
        self.status_label.setText(f"已切换到预设视角: {preset_name}")
    
    def _save_current_view(self):
        """保存当前视角"""
        view_name, ok = QInputDialog.getText(
            self, "保存视角", "请输入视角名称:",
            QLineEdit.Normal, f"视角{self.view_saved_count + 1}"
        )
        
        if ok and view_name.strip():
            view_name = view_name.strip()
            global saved_views
            saved_views[view_name] = {
                'elev': current_view_elev,
                'azim': current_view_azim,
                'dist': current_view_dist,
                'timestamp': len(saved_views)
            }
            self.view_saved_count += 1
            self.view_component.refresh_saved_views_list(saved_views)
            self.status_label.setText(f"视角 '{view_name}' 已保存")
    
    def _load_saved_view(self, view_name):
        """加载保存的视角"""
        if view_name not in saved_views:
            return
        view = saved_views[view_name]
        self._set_view(view['elev'], view['azim'], view['dist'])
        self.status_label.setText(f"视角 '{view_name}' 已加载")
    
    def _load_selected_view(self):
        """加载选中的视角"""
        selected = self.view_component.saved_views_list.selectedItems()
        if selected:
            self._load_saved_view(selected[0].text())
    
    def _delete_selected_view(self):
        """删除选中的视角"""
        selected = self.view_component.saved_views_list.selectedItems()
        if selected:
            view_name = selected[0].text()
            reply = QMessageBox.question(
                self, "确认", f"确定要删除视角 '{view_name}' 吗?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._delete_saved_view(view_name)
    
    def _delete_saved_view(self, view_name):
        """删除指定视角"""
        if view_name in saved_views:
            del saved_views[view_name]
            self.view_component.refresh_saved_views_list(saved_views)
            self.status_label.setText(f"视角 '{view_name}' 已删除")
    
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
            self.view_component.refresh_saved_views_list(saved_views)
            self.status_label.setText("已清空所有视角")
    
    def _load_smplx_model(self):
        """加载SMPL-X模型"""
        try:
            model_loaded = False
            for model_path in MODEL_PATHS:
                if os.path.exists(model_path):
                    try:
                        self.body_model = smplx.create(
                            model_path=model_path,
                            model_type="smplx",
                            gender="neutral",
                            flat_hand_mean=True,
                            device=torch.device("cpu")
                        )
                        model_loaded = True
                        self.model_label.setText(f"已加载: {model_path}")
                        self.status_label.setText("状态: 模型加载完成")
                        # 更新joint_mapper信息
                        if hasattr(self.body_model, 'joint_mapper'):
                            self.mapper_text.setText(str(self.body_model.joint_mapper))
                        self._update_render()
                        break
                    except Exception as e:
                        self.model_label.setText(f"加载失败: {str(e)}")
                        continue
            if not model_loaded:
                self.model_label.setText("未找到模型路径")
                self.status_label.setText("状态: 模型加载失败")
        except Exception as e:
            self.status_label.setText(f"加载模型出错: {str(e)}")
            self.model_label.setText(f"错误: {str(e)}")
    
    def _update_shape(self, value):
        """更新体型参数"""
        self.shape_params[0, 0] = value
        self.shape_label.setText(str(value))
        self._update_render()
    
    def _update_joint(self, value, idx):
        """更新关节参数"""
        self.single_frame_component.update_joint_label(idx, value)
        rad = value * np.pi / 180
        
        if idx == GLOBAL_ROTATION:
            axis = JOINT_AXIS_MAP[idx]
            self.pose_params[0, axis] = rad
            self.pose_params[0, 0 if axis != 0 else 1] = 0.0
            self.pose_params[0, 2 if axis != 2 else 1] = 0.0
        else:
            pose_start_idx = 3 + idx * 3
            axis = JOINT_AXIS_MAP.get(idx, 0)
            if 0 <= pose_start_idx + axis < 156:
                self.pose_params[0, pose_start_idx] = 0.0
                self.pose_params[0, pose_start_idx + 1] = 0.0
                self.pose_params[0, pose_start_idx + 2] = 0.0
                self.pose_params[0, pose_start_idx + axis] = rad
        
        self._update_render()
    
    def _reset_all(self):
        """重置所有参数"""
        # 重置体型
        self.shape_slider.setValue(0)
        self.shape_params = torch.zeros(1, 10, device=torch.device("cpu"))
        # 重置关节
        for idx in self.single_frame_component.core_sliders:
            self.single_frame_component.core_sliders[idx].setValue(0)
            self.single_frame_component.core_labels[idx].setText("0°")
        self.pose_params = torch.zeros(1, 156, device=torch.device("cpu"))
        # 重置视角
        self._reset_view()
        self.status_label.setText("状态: 已重置所有参数")
        self._update_render()
    
    def _reset_view(self):
        """重置视角到默认值"""
        self._set_view(DEFAULT_ELEV, DEFAULT_AZIM, DEFAULT_DIST)
    
    def _update_render(self):
        """更新3D渲染"""
        if self.body_model is None:
            self._draw_empty_hint()
            return
        
        self.ax.clear()
        self._init_axes()
        self.ax.view_init(elev=current_view_elev, azim=current_view_azim)
        self.ax.dist = current_view_dist
        
        # 渲染模型
        body_output = self.body_model(
            betas=self.shape_params,
            body_pose=self.pose_params[:, 3:66],
            global_orient=self.pose_params[:, 0:3],
            left_hand_pose=self.pose_params[:, 66:111],
            right_hand_pose=self.pose_params[:, 111:],
        )
        vertices = body_output.vertices.detach().cpu().numpy()[0]
        faces = self.body_model.faces
        self.ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                           triangles=faces, alpha=0.7, color="#4682B4", linewidth=0, antialiased=True)
        joints = body_output.joints.detach().cpu().numpy()[0]
        self.ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='red', s=15, alpha=1.0)
        core_joint_ids = [2, 3, 5, 8, 11, 17, 19]
        for jid in core_joint_ids:
            self.ax.text(joints[jid, 0], joints[jid, 1], joints[jid, 2], f'{jid}', fontsize=8, color='yellow')
        
        self.canvas.draw()
    
    def _browse_output_dir(self):
        """浏览输出目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录", "./", QFileDialog.ShowDirsOnly)
        if directory:
            self.output_dir_edit.setText(directory)
    
    def _generate_animation(self):
        """生成动画帧序列"""
        if self.body_model is None:
            QMessageBox.warning(self, "警告", "请先加载SMPL-X模型！")
            return
        
        # 禁用生成按钮
        self.generate_btn.setDisabled(True)
        self.anim_status_label.setText("正在生成动画...")
        
        # 获取参数
        output_path = self.output_dir_edit.text()
        frame_count = self.frame_count.value()
        interpolation = "smooth" if self.interp_button_group.checkedId() == 1 else "linear"
        
        # 收集动画参数
        shape_start = self.anim_shape_start.value()
        shape_end = self.anim_shape_end.value()
        joint_configs = []
        
        anim_joint_widgets = self.animation_component.get_anim_joint_widgets()
        for idx, (start_box, end_box, checkbox) in anim_joint_widgets.items():
            if checkbox.isChecked():
                joint_configs.append({
                    'idx': idx,
                    'start_val': start_box.value(),
                    'end_val': end_box.value()
                })
        
        # 更新动画线程的全局变量
        update_globals(
            self.body_model, self.shape_params, self.pose_params,
            current_view_elev, current_view_azim, current_view_dist
        )
        
        # 创建动画线程
        self.animation_thread = AnimationWorker(frame_count, output_path, interpolation=interpolation)
        self.animation_thread.set_params(shape_start, shape_end, joint_configs)
        
        # 连接信号
        self.animation_thread.progress_update.connect(self._on_animation_progress)
        self.animation_thread.finished_signal.connect(self._on_animation_finished)
        self.animation_thread.error_signal.connect(self._on_animation_error)
        
        # 启动线程
        self.animation_thread.start()
    
    def _on_animation_progress(self, progress, msg):
        """更新动画进度"""
        self.progress_bar.setValue(progress)
        self.anim_status_label.setText(msg)
    
    def _on_animation_finished(self, output_path):
        """动画生成完成"""
        self.generate_btn.setDisabled(False)
        self.anim_status_label.setText(f"完成！帧已保存到: {output_path}")
        self.status_label.setText("状态: 动画生成完成")
    
    def _on_animation_error(self, error_msg):
        """动画生成出错"""
        self.generate_btn.setDisabled(False)
        self.anim_status_label.setText(f"错误: {error_msg}")
        self.status_label.setText("状态: 动画生成失败")
        QMessageBox.critical(self, "错误", error_msg)
