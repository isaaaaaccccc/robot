import os
import math
import datetime
import torch
from typing import Dict, Any

#
def parse_asset_config(config_dict, config_key):
    if config_key not in config_dict:
        print(f"Key '{config_key}' not found in assets dictionary")
        return None, None
    
    entity_config = config_dict[config_key]

    if hasattr(entity_config, '__iter__') and isinstance(entity_config, list):
        if len(entity_config) == 1:
            return entity_config[0], None
        elif len(entity_config) == 2:
            return entity_config[0], entity_config[1]
        else:
            print(f"Value for key '{config_key}' has {len(entity_config)} elements, expected 1 or 2")
            return None, None
    else:
        return entity_config, None

#
def convert_dict_to_tensors(nested_dict: Dict[str, Any], 
                            dtype: torch.dtype = torch.float32,
                            device: str = None) -> Dict[str, Any]:
    """
    将嵌套字典中的所有列表转换为PyTorch张量
    
    Args:
        nested_dict: 包含列表的嵌套字典
        device: 目标设备（如'cpu', 'cuda:0'），如果为None则使用默认设备
        dtype: 目标数据类型，默认为torch.float32
    
    Returns:
        转换后的字典，其中所有列表都变为PyTorch张量
    """
    if nested_dict is None:
        return None
    
    result = {}
    
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            # 递归处理嵌套字典
            result[key] = convert_dict_to_tensors(value, dtype, device)
        elif isinstance(value, (list, tuple)):
            # 转换列表或元组为PyTorch张量
            tensor = torch.tensor(value, dtype=dtype)
            if device is not None:
                tensor = tensor.to(device)
            result[key] = tensor
        elif isinstance(value, torch.Tensor):
            # 如果已经是张量，确保设备和数据类型正确
            tensor = value
            if dtype is not None and tensor.dtype != dtype:
                tensor = tensor.to(dtype)
            if device is not None and tensor.device != torch.device(device):
                tensor = tensor.to(device)
            result[key] = tensor
        else:
            # 保持其他类型不变（数字、字符串等）
            result[key] = value
    
    return result

#
def reorder_quaternion(quat: torch.Tensor, to_format: str = "xyzw") -> torch.Tensor:
    """
    Convert quaternion between [w, x, y, z] and [x, y, z, w] formats.
    Args:
        quat: Input quaternion tensor
        to_format: Target format - "xyzw" (x, y, z, w) or "wxyz" (w, x, y, z)
    Returns:
        Quaternion in the specified format
    Raises:
        ValueError: If quaternion doesn't have 4 elements or invalid format specified
    """
    if quat.shape[-1] != 4:
        raise ValueError(f"Quaternion must have 4 elements, got {quat.shape[-1]}")
    
    if to_format not in ["xyzw", "wxyz"]:
        raise ValueError(f"Target format must be 'xyzw' or 'wxyz', got '{to_format}'")
    
    if to_format == "xyzw":
        # Rearrange from [w, x, y, z] to [x, y, z, w]
        return torch.roll(quat, shifts=-1, dims=-1)
    elif to_format == "wxyz":
        # Rearrange from [x, y, z, w] to [w, x, y, z]
        return torch.roll(quat, shifts=1, dims=-1)
    else:
        raise ValueError(f"Target format must be 'xyzw' or 'wxyz', got '{to_format}'")

#
def with_camera(cam_settings):
    """
    Check whether CAM_SETTINGS contains camera configuration
    Args:
        cam_settings (dict): camera configurations
    Returns:
        bool: return True if containing any camera configuration, otherwise return False
    """
    if not isinstance(cam_settings, dict):
        return False
    
    # Check all keys. Look for keys starting with "camera" (e.g., "camera", "camera0", "camera1")
    camera_keys = [key for key in cam_settings.keys() 
                  if isinstance(key, str) and key.startswith("camera")]
    
    return len(camera_keys) > 0

#
def generate_filename(prefix="video", extension="mp4", folder_path=""):
    """
    生成带时间戳的文件名
    Args:
        prefix (str): 文件名前缀，默认为"video"
        extension (str): 文件扩展名，默认为"mp4"
        folder_path (str): 文件夹路径，如果提供会检查并创建目录
    Returns:
        str: 完整的文件路径
    """
    # 获取当前时间
    now = datetime.datetime.now()
    
    # 格式化时间字符串
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    # 组合文件名
    filename = f"{prefix}_{timestamp}.{extension}"
    
    # 如果提供了文件夹路径，确保目录存在并返回完整路径
    if folder_path:
        # 创建目录（如果不存在）
        os.makedirs(folder_path, exist_ok=True)
        return os.path.join(folder_path, filename)
    
    return filename

#
def angle_difference(target, current):
    """
    使用PyTorch计算角度差值, 处理-π到π的跨越问题
    Args:
        target: 目标角度张量
        current: 当前角度张量
    Returns:
        最短路径的角度差值张量
    """
    diff = target - current
    # Limit the difference in [-π, π]
    if isinstance(diff, torch.Tensor):
        # 使用PyTorch操作确保梯度可计算
        diff = torch.remainder(diff + math.pi, 2 * math.pi) - math.pi
    else:
        # 处理标量
        diff = ((diff + math.pi) % (2 * math.pi)) - math.pi
    return diff

def angular_velocity(current_qpos, prev_qpos, dt=None):
    """
    基于当前位置和上一次位置计算角速度
    Args:
        current_qpos: 当前关节位置张量
        prev_qpos: 上一次关节位置张量  
        dt: 持续时间
    Returns:
        目标角速度张量
    """
    if prev_qpos is None:
        return torch.zeros_like(current_qpos)
    
    if dt is None:
        dt = 1.0
    
    # 计算位置差值，处理角度跨越边界的情况
    angle_diff = angle_difference(current_qpos, prev_qpos)
    
    # 应用折扣系数计算目标角速度
    target_velocity = angle_diff / dt
    
    return target_velocity


def as_rotation_matrix(quaternion, order="wxyz"):
    """
    根据姿态四元数，计算旋转矩阵
    参数:
        quaternion: torch.Tensor, 末端姿态四元数 
        order: string, 定义四元数的元素顺序
    返回:
        rotation_matrix: 旋转矩阵
    """
    # 提取四元数分量 (顺序为 [qw, qx, qy, qz])
    if order == "wxyz":
        qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    
    # 计算旋转矩阵（从四元数到旋转矩阵的转换）
    # 第一列 (x轴方向)
    r00 = 1 - 2*(qy*qy + qz*qz)
    r10 = 2*(qx*qy + qw*qz)
    r20 = 2*(qx*qz - qw*qy)
    
    # 第二列 (y轴方向)
    r01 = 2*(qx*qy - qw*qz)
    r11 = 1 - 2*(qx*qx + qz*qz)
    r21 = 2*(qy*qz + qw*qx)
    
    # 第三列 (z轴方向)
    r02 = 2*(qx*qz + qw*qy)
    r12 = 2*(qy*qz - qw*qx)
    r22 = 1 - 2*(qx*qx + qy*qy)
    
    # 构建旋转矩阵
    rotation_matrix = torch.tensor([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
    ], device=quaternion.device, dtype=quaternion.dtype)
    
    return rotation_matrix

def offset(position, rotation_matrix, offset=[0.0,0.0,0.0]):
    """
    根据机械臂末端的位置和姿态旋转矩阵，计算沿着局部坐标系方向偏移后的点
    
    参数:
        position: torch.Tensor, 末端位置 [x, y, z]
        rotation_matrix: torch.Tensor, 末端姿态旋转矩阵
        offset: list, 沿着末端x,y,z轴方向的偏移量        
    返回:
        torch.Tensor: 偏移后的位置 [x_new, y_new, z_new]
    """    
    
    # 局部偏移向量
    local_offset = torch.tensor(offset, device=position.device, dtype=position.dtype)
    
    # 将局部偏移转换到世界坐标系
    world_offset = torch.matmul(rotation_matrix, local_offset)
    
    # 计算最终位置
    new_position = position + world_offset
    
    return new_position
