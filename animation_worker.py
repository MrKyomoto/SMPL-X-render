# animation_worker.py
"""
SMPL-X 3D人体动画控制系统 - 动画生成线程
"""

from PyQt5.QtCore import QThread, pyqtSignal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch
import os
from pathlib import Path

# 全局变量引用（从config导入的全局状态）
_body_model = None
_shape_params = None
_pose_params = None
_current_view_elev = 20
_current_view_azim = 45
_current_view_dist = 10


def set_globals(body_model, shape_params, pose_params, view_elev, view_azim, view_dist):
    """设置渲染所需的全局变量"""
    global _body_model, _shape_params, _pose_params
    global _current_view_elev, _current_view_azim, _current_view_dist
    _body_model = body_model
    _shape_params = shape_params
    _pose_params = pose_params
    _current_view_elev = view_elev
    _current_view_azim = view_azim
    _current_view_dist = view_dist


class AnimationWorker(QThread):
    """动画生成线程（增强版）"""
    progress_update = pyqtSignal(int, str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, frames, output_path, parent=None, interpolation="linear"):
        super().__init__(parent)
        self.frames = frames
        self.output_path = output_path
        self.interpolation = interpolation
        self._anim_params = {}
        self._shape_params = None
        self._pose_params = None
    
    def set_params(self, shape_start, shape_end, joint_configs):
        """设置动画参数"""
        self._anim_params = {
            'shape_start': shape_start,
            'shape_end': shape_end,
            'joints': joint_configs
        }
    
    def set_state(self, shape_params, pose_params):
        """设置当前的形状和姿态参数"""
        self._shape_params = shape_params.clone()
        self._pose_params = pose_params.clone()
    
    def run(self):
        try:
            Path(self.output_path).mkdir(parents=True, exist_ok=True)
            total_frames = self.frames
            self.progress_update.emit(0, "初始化...")
            
            params = self._anim_params
            shape_start = params.get('shape_start', 0)
            shape_end = params.get('shape_end', 0)
            joint_configs = params.get('joints', [])
            
            # 定义插值函数
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
            
            # 预先计算所有帧的参数
            frame_params = []
            for frame_idx in range(total_frames):
                t = frame_idx / max(1, total_frames - 1) if total_frames > 1 else 1.0
                
                # 计算当前帧的形状参数
                if self.interpolation == "smooth":
                    current_shape_0 = smooth_interpolate(shape_start, shape_end, t)
                else:
                    current_shape_0 = shape_start + (shape_end - shape_start) * t
                
                # 计算当前帧的姿态参数
                current_pose = torch.zeros(1, 156, device=torch.device("cpu"))
                
                for joint_info in joint_configs:
                    idx = joint_info['idx']
                    start_val = joint_info['start_val']
                    end_val = joint_info['end_val']
                    
                    if self.interpolation == "smooth":
                        current_rad = smooth_interpolate(
                            start_val * np.pi / 180, 
                            end_val * np.pi / 180, 
                            t
                        )
                    else:
                        current_rad = start_val * np.pi / 180 + (end_val - start_val) * np.pi / 180 * t
                    
                    if idx == 'global':
                        # 全局旋转
                        current_pose[0, 0] = 0.0
                        current_pose[0, 1] = current_rad
                        current_pose[0, 2] = 0.0
                    else:
                        # 局部关节旋转
                        pose_start_idx = 3 + idx * 3
                        axis_map = {
                            0: 0, 1: 1, 2: 2, 4: 2, 5: 2, 7: 2, 8: 2, 10: 2, 11: 2,
                            3: 1, 6: 1, 9: 1, 12: 1, 15: 1,
                            13: 0, 14: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0
                        }
                        axis = axis_map.get(idx, 0)
                        if 0 <= pose_start_idx + axis < 156:
                            current_pose[0, pose_start_idx] = 0.0
                            current_pose[0, pose_start_idx + 1] = 0.0
                            current_pose[0, pose_start_idx + 2] = 0.0
                            current_pose[0, pose_start_idx + axis] = current_rad
                
                frame_params.append((current_shape_0, current_pose))
            
            # 渲染所有帧
            for frame_idx, (shape_0, pose) in enumerate(frame_params):
                progress = int((frame_idx / total_frames) * 100)
                self.progress_update.emit(progress, f"渲染帧 {frame_idx + 1}/{total_frames}")
                
                self._render_frame(frame_idx, shape_0, pose)
            
            self.progress_update.emit(100, "完成!")
            self.finished_signal.emit(self.output_path)
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(f"渲染失败: {str(e)}")
    
    def _render_frame(self, frame_idx, shape_0, pose):
        """渲染单帧"""
        try:
            # 创建图形
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
            
            # 设置视角
            ax.view_init(elev=_current_view_elev, azim=_current_view_azim)
            if _current_view_dist is not None:
                ax.dist = _current_view_dist
            
            # 检查模型
            if _body_model is None:
                ax.text(0, 0, 1, "模型未加载", ha="center", va="center", fontsize=14)
            else:
                # 创建形状参数张量
                shape_tensor = torch.zeros(1, 10)
                shape_tensor[0, 0] = shape_0
                
                # 调用模型
                body_output = _body_model(
                    betas=shape_tensor,
                    body_pose=pose[:, 3:66],
                    global_orient=pose[:, 0:3],
                    left_hand_pose=pose[:, 66:111],
                    right_hand_pose=pose[:, 111:],
                )
                
                vertices = body_output.vertices.detach().cpu().numpy()[0]
                faces = _body_model.faces
                
                # 绘制人体网格
                ax.plot_trisurf(
                    vertices[:, 0], 
                    vertices[:, 1], 
                    vertices[:, 2],
                    triangles=faces, 
                    alpha=0.7, 
                    color="#4682B4", 
                    linewidth=0, 
                    antialiased=True
                )
                
                # 绘制关节
                joints = body_output.joints.detach().cpu().numpy()[0]
                ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='red', s=15, alpha=1.0)
                
                # 标注核心关节
                core_joint_ids = [2, 3, 5, 8, 11, 17, 19]
                for jid in core_joint_ids:
                    ax.text(
                        joints[jid, 0], 
                        joints[jid, 1], 
                        joints[jid, 2], 
                        f'{jid}', 
                        fontsize=8, 
                        color='yellow'
                    )
            
            # 保存图像
            output_file = os.path.join(self.output_path, f"frame_{frame_idx:04d}.png")
            fig.savefig(output_file, dpi=100, bbox_inches='tight')
            
            # 关闭图形释放内存
            plt.close(fig)
            
        except Exception as e:
            print(f"渲染帧 {frame_idx} 失败: {e}")
            import traceback
            traceback.print_exc()
