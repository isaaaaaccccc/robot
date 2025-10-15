"""
可给机械臂添加手腕部相机,需要在GenesisSim()初始化之后创建
"""

import torch

# Extension APIs
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))
from envs.genesis_env import GenesisSim
from utils.utils import as_rotation_matrix, offset
from controllers.backend import Backend

WRIST_CAM_SETTINGS = {
    "camera": {
        "res": (640, 480),
        "pos": (-1, -1, -1),
        "lookat": (0, 0, 0),
        "fov": 40,
        "GUI": True,
    },
    "end_effector_link": "panda_hand",
}

class WristCamera(Backend):
    """
    """
    def __init__(self):
        # Add wrist camera to the scene
        self._scene = GenesisSim().scene
        self.device = GenesisSim().device
        self.datatype = torch.float32
        self.cam = self._scene.add_camera(**WRIST_CAM_SETTINGS["camera"])
        
    @property
    def cam(self):
        return self.cam
    
    @property
    def pos(self):
        return self.cam_pos
    
    @property
    def lookat(self):
        return self.cam_lookat
    
    def initialize(self, robot=None):
        self._robot = robot
        
        self.end_effector = self._robot.get_link(WRIST_CAM_SETTINGS["end_effector_link"])

    def step(self):
        self._step += 1
        # 获得robot的手腕位置姿态
        position=self.end_effector.get_pos()
        quaternion=self.end_effector.get_quat() # [qw, qx, qy, qz]
        position = torch.tensor(position, dtype=self.datatype, device=self.device)
        quaternion = torch.tensor(quaternion, dtype=self.datatype, device=self.device)  

        # 更改其camera到手腕
        rotation_matrix = as_rotation_matrix(quaternion)
        self.cam_pos = offset(position,rotation_matrix,[0.0,1.0,0.0])
        self.cam_lookat = offset(position,rotation_matrix,[1.0,0.0,0.0])

        self.cam.set_pose(pos=self.cam_pos,lookat=self.cam_lookat)

        # 返回image
        rgb, *rest = self.cam.render(
            rgb=True,
            # depth        = True,
            # segmentation = True,
        )
        return rgb

    def reset(self):
        """无需reset"""
        pass

    def stop(self):
        """无需stop"""
        pass