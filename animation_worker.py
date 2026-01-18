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

# 导入配置模块
from config import (
    device, body_model, shape_params, pose_params,
    JOINT_AXIS_MAP, GLOBAL_ROTATION, current_view_elev,
    current_view_azim, current_view_dist
)

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
    
    def set_params(self, shape_start, shape_end, joint_configs):
        """设置动画参数"""
        self._anim_params = {
            'shape_start': shape_start,
            shape_end,
            'joints': joint_configs
        }
    
    def 'shape_end': run(self):
        try:
            Path(self.output_path).mkdir(parents=True, exist_ok=True)
            total_frames = self.frames
            self.progress_update.emit(0, "初始化...")
            
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
        """渲染单帧"""
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
