# Adapted from VLMaps (https://github.com/vlmaps/vlmaps).
# Copyright (c) 2023 Tom-Huang. MIT license; see THIRD_PARTY_LICENSES/VLMaps-LICENSE.

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from omegaconf import DictConfig
import hydra

from instance_nav.utils.time_utils import Tic
from instance_nav.utils.mapping_utils import cvt_pose_vec2tf, base_pos2grid_id_3d, grid_id2base_pos_3d, base_rot_mat2theta
from instance_nav.map.map import Map

# from utils.clip_mapping_utils import *
# from utils.clip_utils import *
# from utils.data_collect_utils import *
# from utils.utils import *

from typing import Tuple, List, Dict, Optional, Union


class HabitatSceneLoader:
    """
    A pose converter that bridge the pose from habitat simulator to the pose in the map.
    The full map pose is (row, col, angle_deg) where angle_deg is 0 when pointing negative
    row direction (up).
    The cropped map pose is (row, col, angle_deg) whose row and col are rowmin and colmin
    smaller than full map pose's row and col.
    """

    def __init__(
        self,
        data_dir: Union[Path, str],
        map_config: DictConfig,
        map: Map = None,
        load_gt_map: bool = False,
    ):
        """
        Initialize the pose converter

        Args:
            data_dir (Union[Path, str]): the dir to the data of one scene
            map_config (DictConfig): the config of the map which you can check in config/map_config
            map (Map, optional): The reference to the map if it is already created. When None is provided,
                                    the converter will load the map based on the data_dir and map_config.
                                    Defaults to None.
            load_gt_map (bool, optional): Boolean that controls whether to load GT map. Defaults to False.
        """
        # 初始化数据目录
        self.data_dir = data_dir
        # 初始化地图配置
        self.map_config = map_config
        # 初始化地图对象
        self.map = map
        # 初始化单元格大小
        self.cs = map_config.cell_size
        # 初始化网格大小
        self.gs = map_config.grid_size
        # 初始化相机高度
        self.camera_height = map_config.pose_info.camera_height

        # 如果未提供地图对象，则加载地图
        if map is None:
            # 设置地图目录
            self.map_dir = os.path.join(data_dir, map_config.map_type)

            # 创建地图对象
            self.map = Map.create(map_config)

            # 加载地图
            load_success = self.map.load_map(data_dir)
            assert (
                load_success == True
            ), f"Map loading fails. It could be because the map hasn't been created at {self.map_dir}."
            # 生成障碍物地图
            self.map.generate_obstacle_map()

        # 初始化障碍物地图
        self.obstacles = self.map.obstacles_map
        # 初始化裁剪后的障碍物地图
        self.obstacles_cropped = self.map.obstacles_cropped
        # 初始化裁剪地图的行偏移量
        self.rmin = self.map.rmin
        # 初始化裁剪地图的最大行索引
        self.xmax = self.map.rmax
        # 初始化裁剪地图的列偏移量
        self.cmin = self.map.cmin
        # 初始化裁剪地图的最大列索引
        self.ymax = self.map.cmax

        # 初始化从基座到相机的变换矩阵和基座变换矩阵
        self.base2cam_tf, self.base_transform = self.map.base2cam_tf, self.map.base_transform
        # 初始化基座姿态
        self.base_poses = np.loadtxt(self.map.pose_path)
        # 初始化初始基座变换矩阵
        self.init_base_tf = (
            self.base_transform @ cvt_pose_vec2tf(self.base_poses[0]) @ np.linalg.inv(self.base_transform)
        )
        # 初始化初始基座变换矩阵的逆矩阵
        self.inv_init_base_tf = np.linalg.inv(self.init_base_tf)

        # 初始化完整地图姿态（默认为None）
        self.full_map_pose = None  # (row, col, theta_deg)
        self.full_map_3d_pose = None

        self.tran_map_3d_pose = None
        # TODO: implement loading GT map option

    def get_obstacles_cropped(self) -> np.array:
        """
        获取裁剪后的障碍物地图
        """
        return self.obstacles_cropped

    def get_obstacles_cropped_no_floor(self) -> np.array:
        """
        获取不包含地面的裁剪障碍物地图
        """
        floor_mask = self.gt_cropped == 2
        obstacles_cropped_no_floor = self.obstacles_cropped.copy()
        obstacles_cropped_no_floor[floor_mask] = 1
        return obstacles_cropped_no_floor

    def get_color_topdown_bgr_cropped(self, times: float = 1) -> np.array:
        """
        获取裁剪后的彩色顶视图（BGR格式）
        """
        color_top_down = self.map.generate_rgb_topdown_map()
        color_top_down = color_top_down[self.rmin : self.xmax + 1, self.cmin : self.ymax + 1]
        color_top_down_bgr = cv2.cvtColor(color_top_down, cv2.COLOR_RGB2BGR)
        color_top_down_bgr = cv2.resize(
            color_top_down_bgr,
            (color_top_down_bgr.shape[1] * times, color_top_down_bgr.shape[1] * times),
        )
        return color_top_down_bgr

    def get_gt_semantic_cropped(self) -> np.array:
        """
        获取裁剪后的真值语义地图
        """
        return self.gt_cropped

    def from_cropped_map_pose(self, row: int, col: int, theta_deg: float):
        """
        从裁剪地图姿态转换到完整地图姿态
        """
        self.full_map_pose = [row + self.rmin, col + self.cmin, theta_deg]

    def from_full_map_pose(self, row: int, col: int, theta_deg: float):
        """
        设置完整地图姿态
        """
        self.full_map_pose = [row, col, theta_deg]

    def from_habitat_tf(self, tf_hab: np.ndarray):
        """
        从Habitat变换矩阵转换到完整地图姿态
        """
        tf = self.inv_init_base_tf @ self.base_transform @ tf_hab @ np.linalg.inv(self.base_transform)
        theta = base_rot_mat2theta(tf[:3, :3])
        theta_deg = np.rad2deg(theta)
        x, y, z = tf[:3, 3]
        row, col, height = base_pos2grid_id_3d(self.gs, self.cs, x, y, z)
        self.full_map_3d_pose = [row, col, height, theta_deg]
        self.full_map_pose = [row, col, theta_deg]
        z_tran = z + self.camera_height
        row_tran, col_tran, height_tran = base_pos2grid_id_3d(self.gs, self.cs, x, y, z_tran)
        height_tran -= int(self.map.pcd_min[2] / self.cs)
        self.tran_map_3d_pose = [row_tran, col_tran, height_tran, theta_deg]

    def conver_tf_from_habitat_tf(self, tf_hab: np.ndarray):
        """
        从Habitat变换矩阵转换到完整地图姿态
        """
        tf = self.inv_init_base_tf @ self.base_transform @ tf_hab @ np.linalg.inv(self.base_transform)
        theta = base_rot_mat2theta(tf[:3, :3])
        theta_deg = np.rad2deg(theta)
        x, y, z = tf[:3, 3]
        row, col, height = base_pos2grid_id_3d(self.gs, self.cs, x, y, z)
        return row, col, height, theta_deg

    def from_camera_tf(self, tf_cam: np.ndarray):
        """
        从相机变换矩阵转换到Habitat变换矩阵，再转换到完整地图姿态
        """
        tf_hab = self.base_transform @ self.inv_init_base_tf @ self.base2cam_tf @ tf_cam
        self.from_habitat_tf(tf_hab)

    def to_cropped_map_pose(self) -> Tuple[int, int, float]:
        """
        从完整地图姿态转换到裁剪地图姿态
        """
        assert self.full_map_pose is not None, "Please call from_xx() first."
        return [self.full_map_pose[0] - self.rmin, self.full_map_pose[1] - self.cmin, self.full_map_pose[2]]

    def to_full_map_pose(self) -> Tuple[int, int, float]:
        """
        获取完整地图姿态
        """
        assert self.full_map_pose is not None, "Please call from_xx() first."
        return self.full_map_pose

    def to_habitat_tf(self) -> np.ndarray:
        """
        从完整地图姿态转换到Habitat变换矩阵
        """
        assert self.full_map_pose is not None, "Please call from_xx() first."
        row, col, theta_deg = self.full_map_pose
        x, y, z = grid_id2base_pos_3d(row, col, 0, self.cs, self.gs)
        theta = np.deg2rad(theta_deg)
        tf = np.eye(4)
        tf[:3, 3] = [x, y, z]
        tf[0, 0] = np.cos(theta)
        tf[1, 1] = np.cos(theta)
        tf[0, 1] = -np.sin(theta)
        tf[1, 0] = np.sin(theta)
        tf_hab = np.linalg.inv(self.base_transform) @ self.init_base_tf @ tf @ self.base_transform
        return tf_hab


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="test_config.yaml",
)
def main(config: DictConfig) -> None:
    """
    主函数，用于测试HabitatSceneLoader类
    """
    # 设置数据目录
    data_dir = Path(config.data_paths.scene_data_dir) / "scene_dataset"
    # 获取所有数据目录
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    # 创建HabitatSceneLoader对象
    dataloader = HabitatSceneLoader(data_dirs[0], config.map_config)
    # 随机选择10个基座姿态索引
    ids_list = np.random.randint(0, len(dataloader.base_poses), 10)
    # 遍历选择的基座姿态索引
    for test_i, i in enumerate(ids_list):
        # 获取基座姿态变换矩阵
        base_hab_tf = cvt_pose_vec2tf(dataloader.base_poses[i])
        # 从Habitat变换矩阵转换到完整地图姿态
        dataloader.from_habitat_tf(base_hab_tf)
        # 获取完整地图姿态
        full_map_pose = dataloader.to_full_map_pose()
        print("full map pose: ", full_map_pose)
        # 设置完整地图姿态
        dataloader.from_full_map_pose(*full_map_pose)
        # 从完整地图姿态转换回Habitat变换矩阵
        cvt_hab_tf = dataloader.to_habitat_tf()
        print("The correct habitat tf is: \n", base_hab_tf)
        print("The converted habitat tf is: \n", cvt_hab_tf)

        # 计算误差
        err = np.linalg.norm(base_hab_tf - cvt_hab_tf)
        print(f"Error: {err}")
        # 断言误差小于1
        assert err < 1, "[TEST RESULTS]: FAIL! The converted habitat tf is not correct."
        print(f"[TEST RESULTS]: PASS {test_i+1}/{len(ids_list)}!")


if __name__ == "__main__":
    main()
