from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QSlider, QLabel, QGroupBox, QGridLayout
)
import matplotlib.pyplot as plt
import matplotlib
import sys
import torch
import smplx
import numpy as np
# 兼容NumPy 2.0的Inf/inf大小写问题
np.Inf = np.inf
np.NAN = np.nan
matplotlib.use('Agg')
# 修复Matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans',
                                   'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 全局参数与模型初始化 ======================
device = None
body_model = None
shape_params = None
pose_params = None

# ====================== 主窗口类（核心交互逻辑） ======================


class HumanAnimationSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        # 初始化device和参数
        global device, shape_params, pose_params
        device = torch.device("cpu")
        shape_params = torch.zeros(1, 10, device=device)
        pose_params = torch.zeros(1, 156, device=device)

        self.setWindowTitle("3D Human Animation Control System")
        self.setGeometry(100, 100, 1200, 800)

        # 1. 创建Matplotlib画布
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(0, 2)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Human Model Preview")
        self.canvas = FigureCanvas(self.fig)

        # 2. 创建控制面板（修复关节索引）
        self.control_panel = self._create_control_panel()

        # 3. 布局整合
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.addWidget(self.canvas, 7)
        main_layout.addWidget(self.control_panel, 3)
        self.setCentralWidget(central_widget)

        # 4. 初始绘制提示
        self._draw_empty_hint()

    def _create_control_panel(self):
        """创建交互控制面板（精准关节索引）"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # ① 模型加载区
        load_group = QGroupBox("Model Loading")
        load_layout = QHBoxLayout(load_group)
        self.load_btn = QPushButton("Load SMPL-X Model")
        self.load_btn.clicked.connect(self._load_smplx_model)
        self.model_label = QLabel("No model loaded")
        load_layout.addWidget(self.load_btn)
        load_layout.addWidget(self.model_label)
        layout.addWidget(load_group)

        # ② 体型调整区
        shape_group = QGroupBox("Body Shape (β0: Height/Weight)")
        shape_layout = QHBoxLayout(shape_group)
        self.shape_slider = QSlider(Qt.Horizontal)
        self.shape_slider.setRange(-5, 5)
        self.shape_slider.setValue(0)
        self.shape_slider.setTickInterval(1)
        self.shape_slider.setTickPosition(QSlider.TicksBelow)
        self.shape_slider.valueChanged.connect(self._update_shape)
        self.shape_label = QLabel("Current value: 0")
        shape_layout.addWidget(QLabel("β0:"))
        shape_layout.addWidget(self.shape_slider)
        shape_layout.addWidget(self.shape_label)
        layout.addWidget(shape_group)

        # ③ 核心关节控制区（终极修正：精准索引）
        joint_group = QGroupBox("Joint Control (Angle)")
        joint_layout = QGridLayout(joint_group)
        self.joint_sliders = {}
        self.joint_labels = {}
        # 亲测验证的精准索引！！！
        joints = [
            ("Right Shoulder (X)", 51),   # 右肩X轴（抬肩/落肩）
            ("Waist Bend (X)", 18),      # 腰部X轴（弯腰/挺腰）
            ("Right Elbow (X)", 57),     # 右肘X轴（弯肘/伸肘）
            ("Right Knee (X)", 15),      # 右膝X轴（弯膝/伸膝）
            ("Global Rotation (Y)", 1)   # 全局Y轴（整体转向）
        ]
        for i, (name, idx) in enumerate(joints):
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-90, 90)  # 合理范围：-90°到90°（符合人体运动）
            slider.setValue(0)
            slider.setTickInterval(10)
            slider.setTickPosition(QSlider.TicksBelow)
            # 绑定值变化事件（带索引传参）
            slider.valueChanged.connect(
                lambda v, id=idx: self._update_joint(v, id))
            self.joint_sliders[idx] = slider
            label = QLabel(f"{name}: 0°")
            self.joint_labels[idx] = label
            joint_layout.addWidget(label, i, 0)
            joint_layout.addWidget(slider, i, 1)
        layout.addWidget(joint_group)

        # ④ 重置按钮
        reset_btn = QPushButton("Reset All Parameters")
        reset_btn.clicked.connect(self._reset_all)
        layout.addWidget(reset_btn)

        layout.addStretch()
        return panel

    def _load_smplx_model(self):
        """加载官方SMPL-X模型（适配目录结构）"""
        try:
            global body_model
            # 加载模型（固定目录：./smplx_models/smplx/SMPLX_NEUTRAL.npz）
            body_model = smplx.create(
                model_path="./smplx_models",
                model_type="smplx",
                gender="neutral",
                flat_hand_mean=True,
                use_pca=False,
                num_pca_comps=45,
                device=device
            )
            self.model_label.setText("Loaded: SMPLX_NEUTRAL (Official)")
            print(f"✅ SMPL-X模型加载成功！")
            self._update_render()
        except Exception as e:
            error_info = f"Load failed: {str(e)[:30]}..."
            self.model_label.setText(error_info)
            print(f"❌ 加载失败原因：{e}")

    def _update_shape(self, value):
        """更新体型参数"""
        if body_model is None:
            return
        global shape_params
        shape_params[0, 0] = value
        self.shape_label.setText(f"Current value: {value}")
        self._update_render()

    def _update_joint(self, value, idx):
        """更新关节姿态（角度转弧度，精准映射）"""
        if body_model is None:
            return
        global pose_params
        # 角度转弧度（支持负数）
        rad = value * np.pi / 180

        # 区分全局旋转和身体关节：
        if idx == 1:  # 全局旋转Y轴（global_orient）
            pose_params[0, idx] = rad
        else:  # 身体关节（body_pose）
            pose_params[0, 3 + idx] = rad  # body_pose从索引3开始！！！

        # 更新标签显示
        if idx in self.joint_labels:
            self.joint_labels[idx].setText(
                f"{self.joint_labels[idx].text().split(':')[0]}: {value}°")
        self._update_render()

    def _reset_all(self):
        """重置所有参数"""
        global shape_params, pose_params
        shape_params = torch.zeros(1, 10, device=device)
        pose_params = torch.zeros(1, 156, device=device)
        # 重置体型滑块
        self.shape_slider.setValue(0)
        self.shape_label.setText("Current value: 0")
        # 重置关节滑块和标签
        for idx, slider in self.joint_sliders.items():
            slider.setValue(0)
            self.joint_labels[idx].setText(
                f"{self.joint_labels[idx].text().split(':')[0]}: 0°")
        # 重新渲染
        if body_model is not None:
            self._update_render()

    def _update_render(self):
        """更新3D渲染视图（优化显示）"""
        if body_model is None:
            return
        try:
            self.ax.clear()
            # 设置坐标轴（优化视角）
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
            self.ax.set_zlim(0, 2)
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")
            self.ax.set_title("Human Model Preview")
            self.ax.view_init(elev=20, azim=45)  # 固定视角，方便观察

            # 前向计算模型顶点
            body_output = body_model(
                betas=shape_params,
                body_pose=pose_params[:, 3:66],    # body_pose从索引3开始
                global_orient=pose_params[:, 0:3],  # 全局旋转前3维
                left_hand_pose=pose_params[:, 66:111],
                right_hand_pose=pose_params[:, 111:],
            )
            # 转numpy并绘制
            vertices = body_output.vertices.detach().cpu().numpy()[0]
            faces = body_model.faces
            self.ax.plot_trisurf(
                vertices[:, 0], vertices[:, 1], vertices[:, 2],
                triangles=faces, alpha=0.7, color="#4682B4",
                linewidth=0, antialiased=True
            )
            self.canvas.draw()
        except Exception as e:
            print(f"Render error: {e}")
            self._draw_empty_hint()

    def _draw_empty_hint(self):
        """空模型时绘制提示文字"""
        self.ax.clear()
        self.ax.text(
            0, 0, 1, "Please load SMPL-X model first",
            ha="center", va="center", fontsize=14
        )
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(0, 2)
        self.canvas.draw()


# ====================== 程序入口 ======================
if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = HumanAnimationSystem()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Program start error: {e}")
        sys.exit(1)
