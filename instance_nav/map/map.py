from __future__ import annotations

# Adapted from VLMaps (https://github.com/vlmaps/vlmaps).
# Copyright (c) 2023 Tom-Huang. MIT license; see THIRD_PARTY_LICENSES/VLMaps-LICENSE.

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import logging
import cv2
import numpy as np
from omegaconf import DictConfig
from scipy.ndimage import binary_dilation, gaussian_filter
from shapely.geometry import Point, Polygon

from instance_nav.utils.navigation_utils import get_dist_to_bbox_2d

# from instance_nav.utils.mapping_utils import load_map


class Map:
    def __init__(self, map_config: DictConfig, data_dir: str = ""):
        self.map_config = map_config
        # obstacles_path = os.path.join(map_dir, "obstacles.npy")
        # self.obstacles = load_map(obstacles_path)
        self.gs = map_config["grid_size"]
        self.cs = map_config["cell_size"]

        self.mapped_iter_list = None
        self.grid_feat = None
        self.grid_pos = None
        self.weight = None
        self.occupied_ids = None
        self.grid_rgb = None

        self.obstacles_map = None
        self.obstacles_cropped = None

        self._setup_transforms()
        if data_dir:
            self._setup_paths(data_dir)
        # self.obstacles_new_cropped = None

    def _setup_paths(self, data_dir: Union[Path, str]) -> None:
        self.data_dir = Path(data_dir)
        self.rgb_dir = self.data_dir / "rgb"
        self.depth_dir = self.data_dir / "depth"
        self.semantic_dir = self.data_dir / "semantic"
        self.pose_path = self.data_dir / "poses.txt"
        try:
            self.rgb_paths = sorted(self.rgb_dir.glob("*.png"))
            self.depth_paths = sorted(self.depth_dir.glob("*.npy"))
            self.semantic_paths = sorted(self.semantic_dir.glob("*.npy"))
        except FileNotFoundError as e:
            print(e)

    def _setup_transforms(self) -> np.ndarray:
        """
        Setup the transformation from base frame to camera frame: self.base2cam_tf
        Setup the transformation from the standard mobile base coordinate
        (x forward, y left, z up) to base frame"""
        self.base2cam_tf = np.eye(4)
        self.base2cam_tf[:3, :3] = np.array([self.map_config.pose_info.base2cam_rot]).reshape((3, 3))
        # 加了会影响绕y轴的旋转，具体来说就是有绕y轴的旋转时，会将这个位移叠加到变换中
        # self.base2cam_tf[1, 3] = self.map_config.pose_info.camera_height
        # transform the base coordinate such that x is forward, y is leftward, z is upward
        self.base_transform = np.eye(4)
        self.base_transform[0, :3] = self.map_config.pose_info.base_forward_axis
        self.base_transform[1, :3] = self.map_config.pose_info.base_left_axis
        self.base_transform[2, :3] = self.map_config.pose_info.base_up_axis

        return self.base2cam_tf, self.base_transform

    def load_map(self, map_dir: str):
        return NotImplementedError

    def index_map(self, language_desc: str, with_init_cat: bool = True):
        return NotImplementedError

    def generate_obstacle_map(self, h_min: float = 0, h_max: float = 1.5) -> np.ndarray:
        """Generate topdown obstacle map from loaded 3D map

        Args:
            h_min (float, optional): The minimum height (m) of voxels considered
                as obstacles. Defaults to 0.
            h_max (float, optional): The maximum height (m) of voxels considered
                as obstacles. Defaults to 1.5.
        Return:
            obstacles_map (np.ndarray): (gs, gs) 1 is free, 0 is occupied
        """
        assert self.occupied_ids is not None, "map not loaded"
        heights = np.arange(0, self.occupied_ids.shape[-1]) * self.cs
        logging.info(f"heights min: {np.min(heights)}, max: {np.max(heights)}")
        height_mask = np.logical_and(heights > h_min, heights < h_max)
        self.obstacles_map = np.sum(self.occupied_ids[..., height_mask] > 0, axis=2) == 0
        self.envelope_map = np.sum(self.occupied_ids > 0, axis=2) == 0
        self.generate_cropped_obstacle_map(self.obstacles_map, self.envelope_map)
        self.min_height = h_min
        self.max_height = h_max
        # obs_map_vis = (self.obstacles_map[:, :, None] * 255).astype(np.uint8)
        # obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])
        # cv2.imshow("#obs#", obs_map_vis)
        # cv2.waitKey()
        return self.obstacles_map

    def generate_3d_obstacle_map(self, unique_heights: list[float], potential_obstacle_names: List[str], obstacle_names: List[str], h_min_rel: float = 0, h_max_rel: float = 1.5, vis: bool = False) -> list[np.ndarray]:
        assert self.occupied_ids is not None, "map not loaded"
        self.obstacle_3d_maps = []
        self.obstacles_3d_cropped = []
        self.min_3d_heights = []
        self.max_3d_heights = []
        self.passable_3d_maps = []
        for uh in unique_heights:
            h_min = (uh * self.cs) + h_min_rel
            h_max = (uh * self.cs) + h_max_rel
            self.min_3d_heights.append(h_min)
            self.max_3d_heights.append(h_max)
            heights = np.arange(0, self.occupied_ids.shape[-1]) * self.cs
            logging.info(f"generate_3d_obstacle_map: heights min: {h_min}, max: {h_max}, h_min_rel: {h_min_rel}, h_max_rel: {h_max_rel}")
            height_mask = np.logical_and(heights > h_min, heights < h_max)
            obstacle_map = np.sum(self.occupied_ids[..., height_mask] > 0, axis=2) == 0
            obstacles_cropped = self.generate_cropped_3d_obstacle_map(self.obstacles_map, obstacle_map)

            self.obstacle_3d_maps.append(obstacle_map)
            self.obstacles_3d_cropped.append(obstacles_cropped)

            passable_map = self.customize_obstacle_3d_map(
                obstacles_cropped,
                potential_obstacle_names,
                obstacle_names,
                h_min,
                h_max,
                vis=vis,
            )
            self.passable_3d_maps.append(passable_map)

            if vis:
                obs_map_vis = (passable_map[:, :, None] * 255).astype(np.uint8)
                obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])
                title = f"3d_obs_floor_{uh}"
                cv2.imshow(title, obs_map_vis)
                cv2.waitKey()
        return self.obstacle_3d_maps, self.obstacles_3d_cropped, self.passable_3d_maps

    def generate_cropped_obstacle_map(self, obstacle_map: np.ndarray, envelope_map: np.ndarray) -> np.ndarray:
        x_indices, y_indices = np.where(obstacle_map == 0)
        self.rmin = np.min(x_indices)
        self.rmax = np.max(x_indices)
        self.cmin = np.min(y_indices)
        self.cmax = np.max(y_indices)
        self.obstacles_cropped = obstacle_map[self.rmin : self.rmax + 1, self.cmin : self.cmax + 1]
        self.envelope_cropped = envelope_map[self.rmin : self.rmax + 1, self.cmin : self.cmax + 1]
        return self.obstacles_cropped

    def generate_cropped_3d_obstacle_map(self, max_obstacle_map: np.ndarray, obstacle_map: np.ndarray) -> np.ndarray:
        x_indices, y_indices = np.where(max_obstacle_map == 0)
        self.rmin = np.min(x_indices)
        self.rmax = np.max(x_indices)
        self.cmin = np.min(y_indices)
        self.cmax = np.max(y_indices)
        obstacles_cropped = obstacle_map[self.rmin : self.rmax + 1, self.cmin : self.cmax + 1]
        return obstacles_cropped

    def generate_rgb_topdown_map(self) -> np.ndarray:
        assert self.grid_rgb is not None, "map not loaded"
        assert self.grid_pos is not None
        rgb_topdown = np.zeros((self.gs, self.gs, 3))
        for ci, (rgb, pos) in enumerate(zip(self.grid_rgb, self.grid_pos)):
            row, col, _ = pos
            rgb_topdown[row, col, :] = rgb.flatten()
        return rgb_topdown.astype(np.uint8)

    def init_categories(self, categories: List[str]) -> np.ndarray:
        return NotImplementedError
    # Implemented by PrebuiltMap.
    def customize_obstacle_map(self, potential_obstacle_names: List[str], obstacle_names: List[str], passable_names: List[str], vis: bool = False) -> np.ndarray:
        return NotImplementedError

    def customize_obstacle_3d_map(self, obstacles_cropped, potential_obstacle_names: List[str], obstacle_names: List[str], h_min: float = 0.0, h_max: float = 1.5, vis: bool = False) -> np.ndarray:
        return NotImplementedError
    @staticmethod
    def create(map_config: DictConfig) -> Map:
        from instance_nav.map import PrebuiltMap

        if map_config.map_type == "vlmap":
            return PrebuiltMap(map_config)

        raise ValueError(f"Unsupported prebuilt map type: {map_config.map_type}")
    def get_pos(self, name: str) -> Tuple[List[List[int]], List[List[float]], List[np.ndarray], Any]:
        """
        Get the contours, centers, and bbox list of a certain category
        on a full map
        """
        return NotImplementedError

    def get_pos_and_color(self, name: str, vis: bool) -> Tuple[List[List[int]], List[List[float]], List[np.ndarray], List[Dict]]:
        """
        Get the contours, centers, and bbox list of a certain category
        on a full map
        """
        return NotImplementedError

    def get_pos_and_color_quantization(self, name: str, vis: bool) -> Tuple[List[List[int]], List[List[float]], List[np.ndarray], List[Dict]]:
        """
        Get the contours, centers, and bbox list of a certain category
        on a full map
        """
        return NotImplementedError

    def get_pos_color_and_height(self, name: str, floor: int, vis: bool) -> Tuple[List[List[int]], List[List[float]], List[np.ndarray], List[Dict], List[int]]:
        """
        Get the contours, centers, and bbox list of a certain category
        on a full map
        """
        return NotImplementedError

    def get_distribution_map(self, name: str) -> np.ndarray:
        return NotImplementedError

    def get_predict_mask(self, name: str) -> np.ndarray:
        return NotImplementedError

    def get_obstacle_cropped(self):
        return self.obstacles_cropped

    def get_customized_obstacle_cropped(self):
        return self.obstacles_new_cropped

    def get_rgb_topdown_map_cropped(self) -> np.ndarray:
        rgb_map = self.generate_rgb_topdown_map()
        return rgb_map[self.rmin : self.rmax, self.cmin : self.cmax]

    @staticmethod
    def _dilate_map(binary_map: np.ndarray, dilate_iter: int = 0, gaussian_sigma: float = 1.0, use_dilation=True):
        h, w = binary_map.shape
        binary_map = cv2.resize(binary_map.astype(float), (w * 2, h * 2))
        binary_map = gaussian_filter((binary_map).astype(float), sigma=gaussian_sigma, truncate=3)
        binary_map = (binary_map > 0.5).astype(np.uint8)
        if use_dilation:
            binary_map = binary_dilation(
                binary_map,
                structure=np.ones((3, 3)),
                iterations=dilate_iter * 2,
            )
        binary_map = cv2.resize(binary_map.astype(float), (w, h))
        return binary_map

    def get_nearest_pos(self, curr_pos: List[float], name: str, vis: bool = False) -> List[float]:
        # 获取目标位置信息
        contours, centers, bbox_list = self.get_pos(name)
        logging.debug(f"contours: {len(contours)}")
        # 过滤掉面积小于阈值的物体
        ids_list = self.filter_small_objects(bbox_list, name, area_thres=10)
        logging.debug(f"ids_list: {len(ids_list)}")
        # 根据过滤后的id列表，重新整理contours、centers和bbox_list
        contours = [contours[i] for i in ids_list]
        centers = [centers[i] for i in ids_list]
        bbox_list = [bbox_list[i] for i in ids_list]
        # 如果没有符合条件的物体，则返回当前位置
        if len(centers) == 0:
            logging.warning("没有符合条件的物体")
            return curr_pos
        # 选择最近的物体
        id = self.select_nearest_obj(centers, bbox_list, curr_pos)
        # 显示选中物体的轮廓
        if vis:
            # 获取障碍物地图
            obs_map = self.get_customized_obstacle_cropped()
            if obs_map is None or obs_map.size == 0:
                print("Error: obs_map is empty or invalid!")
                return curr_pos
            # 创建三通道 BGR 图像
            obs_map_vis = (obs_map * 255).astype(np.uint8)
            obs_map_vis = np.stack([obs_map_vis] * 3, axis=-1)

            # 绘制选中轮廓
            contour = contours[id]
            contour_cv2 = contour[:, [1, 0]]  # [row, col] → [col, row]
            # 将全图坐标转换为裁剪图像坐标
            contour_cv2[:, 0] -= self.cmin  # col - cmin
            contour_cv2[:, 1] -= self.rmin  # row - rmin
            cv2.drawContours(obs_map_vis, [contour_cv2], 0, (0, 255, 0), 3)  # 绿色轮廓

            # 绘制选中物体的中心点（红点）
            selected_center = centers[id]
            center_col = int(selected_center[1]) - self.cmin  # col - cmin
            center_row = int(selected_center[0]) - self.rmin  # row - rmin
            cv2.circle(
                obs_map_vis,
                (center_col, center_row),
                radius=5,
                color=(0, 0, 255),
                thickness=-1
            )

            # 保存图像以供调试
            # cv2.imwrite("debug_output.png", obs_map_vis)
            print(f"######nav object name : {name} #######")
            # 显示最终图像
            cv2.imshow("Selected Contour and Center", obs_map_vis)
            cv2.waitKey()
        # 返回当前位置到目标物体轮廓上的最近点
        return self.nearest_point_on_polygon(curr_pos, contours[id])

    def get_nearest_and_similarity_color_pos(
        self,
        curr_pos: List[float],
        name: str,
        target_colors: List[List[int]],
        color_weight: float = 0.5,
        area_thres: int = 5,
        vis: bool = False
    ) -> List[float]:
        """
        Get nearest position considering color information
        :param curr_pos: Current [row, col] position
        :param name: Object category name
        :param target_colors: List of target RGB colors (e.g., [[255,0,0]] or [[255,0,0], [255,255,255]])
        :param color_weight: Weight of color similarity (0-1)
        :param vis: Whether to visualize
        """
        # 获取目标位置信息和颜色分布
        contours, centers, bbox_list, color_dists = self.get_pos_and_color(name, vis)

        # 过滤小物体
        ids_list = self.filter_small_objects(bbox_list, name, area_thres=area_thres)
        contours = [contours[i] for i in ids_list]
        centers = [centers[i] for i in ids_list]
        bbox_list = [bbox_list[i] for i in ids_list]
        color_dists = [color_dists[i] for i in ids_list]

        if len(centers) == 0:
            return curr_pos

        # 计算颜色匹配分数
        color_scores = np.ones(len(centers))

        if target_colors:
            for i, color_dist in enumerate(color_dists):
                main_colors = color_dist["main_colors"]
                score = 0.0

                if len(target_colors) == 1:
                    # 单个目标颜色：只考虑相似度最高的值
                    best_match_score = 0.0
                    for color_info in main_colors:
                        # 使用余弦相似度计算颜色匹配度
                        obj_color = np.array(color_info["color"])
                        tgt_color = np.array(target_colors[0])

                        # 计算余弦相似度
                        # dot_product = np.dot(obj_color, tgt_color)
                        # norm_obj = np.linalg.norm(obj_color)
                        # norm_tgt = np.linalg.norm(tgt_color)
                        # similarity = dot_product / (norm_obj * norm_tgt + 1e-6)
                        distance = np.linalg.norm(obj_color - tgt_color)
                        similarity = 1 / (1 + distance/441)
                        # 更新最佳匹配
                        if similarity > best_match_score:
                            best_match_score = similarity

                    # 单个颜色直接使用最佳匹配分数
                    color_scores[i] = best_match_score

                elif len(target_colors) >= 2:
                    # 两个目标颜色：分别匹配主色和次主色
                    primary_match = 0.0
                    secondary_match = 0.0

                    # 匹配第一个目标颜色（主色）
                    best_primary_match = 0.0
                    primary_idx = -1
                    for j, color_info in enumerate(main_colors):
                        obj_color = np.array(color_info["color"])
                        tgt_color = np.array(target_colors[0])
                        # 余弦相似度
                        # dot_product = np.dot(obj_color, tgt_color)
                        # norm_obj = np.linalg.norm(obj_color)
                        # norm_tgt = np.linalg.norm(tgt_color)
                        # similarity = dot_product / (norm_obj * norm_tgt + 1e-6)
                        distance = np.linalg.norm(obj_color - tgt_color)
                        similarity = 1 / (1 + distance/441)
                        if similarity > best_primary_match:
                            best_primary_match = similarity
                            primary_idx = j

                    primary_match = best_primary_match

                    # 匹配第二个目标颜色（剩余颜色）
                    best_secondary_match = 0.0
                    for j, color_info in enumerate(main_colors):
                        # 跳过已匹配的主色
                        if j == primary_idx:
                            continue

                        obj_color = np.array(color_info["color"])
                        tgt_color = np.array(target_colors[1])
                        # 余弦相似度
                        # dot_product = np.dot(obj_color, tgt_color)
                        # norm_obj = np.linalg.norm(obj_color)
                        # norm_tgt = np.linalg.norm(tgt_color)
                        # similarity = dot_product / (norm_obj * norm_tgt + 1e-6)
                        distance = np.linalg.norm(obj_color - tgt_color)
                        similarity = 1 / (1 + distance/441)
                        if similarity > best_secondary_match:
                            best_secondary_match = similarity

                    # 如果只有一个主色，使用它匹配第二个目标颜色
                    if best_secondary_match == 0 and len(main_colors) > 0:
                        obj_color = np.array(main_colors[0]["color"])
                        tgt_color = np.array(target_colors[1])

                        dot_product = np.dot(obj_color, tgt_color)
                        norm_obj = np.linalg.norm(obj_color)
                        norm_tgt = np.linalg.norm(tgt_color)
                        best_secondary_match = dot_product / (norm_obj * norm_tgt + 1e-6)

                    secondary_match = best_secondary_match

                    # 计算平均分数
                    # 多颜色匹配 两种
                    # color_scores[i] = (primary_match + secondary_match) / 2.0
                    #颜色匹配 1种
                    color_scores[i] = primary_match

        # 计算距离分数
        dist_scores = np.zeros(len(centers))
        for i, center in enumerate(centers):
            dist = np.sqrt((center[0]-curr_pos[0])**2 + (center[1]-curr_pos[1])**2)
            dist_scores[i] = 1 / (dist + 1e-6)  # 避免除以零

        if np.max(dist_scores) > 0:
            dist_scores = dist_scores / np.max(dist_scores)

        # 综合评分 = 颜色匹配分数 * 颜色权重 + 距离分数 * (1 - 颜色权重)
        combined_scores = (color_scores * color_weight) + (dist_scores * (1 - color_weight))
        best_id = np.argmax(combined_scores)
        if vis:
            # 获取障碍物地图
            obs_map = self.get_customized_obstacle_cropped()
            if obs_map is None or obs_map.size == 0:
                print("Error: obs_map is empty or invalid!")
                return curr_pos
            # 创建三通道 BGR 图像
            obs_map_vis = (obs_map * 255).astype(np.uint8)
            obs_map_vis = np.stack([obs_map_vis] * 3, axis=-1)

            # 绘制选中轮廓
            contour = contours[best_id]
            contour_cv2 = contour[:, [1, 0]]  # [row, col] → [col, row]
            # 将全图坐标转换为裁剪图像坐标
            contour_cv2[:, 0] -= self.cmin  # col - cmin
            contour_cv2[:, 1] -= self.rmin  # row - rmin
            cv2.drawContours(obs_map_vis, [contour_cv2], 0, (0, 255, 0), 3)  # 绿色轮廓

            # 绘制选中物体的中心点（红点）
            selected_center = centers[best_id]
            center_col = int(selected_center[1]) - self.cmin  # col - cmin
            center_row = int(selected_center[0]) - self.rmin  # row - rmin
            cv2.circle(
                obs_map_vis,
                (center_col, center_row),
                radius=5,
                color=(0, 0, 255),
                thickness=-1
            )

            # 保存图像以供调试
            # cv2.imwrite("debug_output.png", obs_map_vis)
            print(f"######nav object name : {name} #######")
            # 显示最终图像
            cv2.imshow(f"Selected Contour and Center {name}", obs_map_vis)
            cv2.waitKey()

        return self.nearest_point_on_polygon(curr_pos, contours[best_id])

    def nearest_point_on_polygon(self, coord: List[float], polygon: List[List[float]]):
        # Create a Shapely Point from the given coordinate
        point = Point(coord)

        # Create a Shapely Polygon from the polygon's coordinates
        poly = Polygon(polygon)

        # Find the nearest point on the polygon's boundary to the given point
        nearest = poly.exterior.interpolate(poly.exterior.project(point))

        # Extract the nearest point's coordinates as a tuple
        nearest_coords = [int(nearest.x), int(nearest.y)]

        return nearest_coords

    def get_forward_pos(self, curr_pos: List[float], curr_angle_deg: float, meters: float) -> List[float]:
        i, j = curr_pos
        rad = np.deg2rad(curr_angle_deg)
        pix = meters / self.cs

        new_i = i - pix * np.cos(rad)
        new_j = j + pix * np.sin(rad)
        return [new_i, new_j]

    def filter_small_objects(self, bbox_list: List[List[int]], name, area_thres: int = 50) -> List[int]:
        results_ids = []
        for bbox_i, bbox in enumerate(bbox_list):
            dx = bbox[1] - bbox[0]
            dy = bbox[3] - bbox[2]
            area = dx * dy
            if area > area_thres:
                results_ids.append(bbox_i)
            else:
                logging.info(f"{name} {bbox_i} is too small : {area}")
        return results_ids

    #  TODO: there has some change should be recover
    def select_nearest_obj(
        self,
        centers: List[List[float]],
        bbox_list: List[List[float]],
        curr_pos: Tuple[float, float],
    ) -> int:
        dist_list = []
        for c, bbox in zip(centers, bbox_list):
            size = np.array([bbox[1] - bbox[0], bbox[3] - bbox[2]])
            dist = get_dist_to_bbox_2d(np.array(c), size, np.array(curr_pos))
            dist_list.append(dist)
        # id = np.argmax(dist_list)
        dist_array = np.array(dist_list)
        sorted_indices = np.argsort(dist_array)
        second_farthest_index = sorted_indices[0]

        return second_farthest_index

    def _get_left_pos(self, curr_pos: List[float], tar_pos: List[float], tar_bbox: List[float]) -> List[float]:
        di = tar_pos[0] - curr_pos[0]
        dj = tar_pos[1] - curr_pos[1]
        # angle = np.arctan2(dj, -di)
        angle = np.arctan2(-dj, -di)

        d1 = 0.5 * (tar_bbox[3] - tar_bbox[2]) / np.abs(np.cos(angle))
        d2 = 0.5 * (tar_bbox[1] - tar_bbox[0]) / np.abs(np.sin(angle))
        d = min(d1, d2) + 0.5
        h = tar_bbox[1] - tar_bbox[0]
        w = tar_bbox[3] - tar_bbox[2]
        d = 0.5 * np.sqrt(h * h + w * w) + 2

        # i_final = tar_pos[0] - np.sin(angle) * d
        i_final = tar_pos[0] + np.sin(angle) * d
        j_final = tar_pos[1] - np.cos(angle) * d

        return [i_final, j_final]

    def _get_right_pos(self, curr_pos: List[float], tar_pos: List[float], tar_bbox: List[float]) -> List[float]:
        di = tar_pos[0] - curr_pos[0]
        dj = tar_pos[1] - curr_pos[1]
        angle = np.arctan2(-dj, -di)

        d1 = 0.5 * (tar_bbox[3] - tar_bbox[2]) / np.abs(np.cos(angle))
        d2 = 0.5 * (tar_bbox[1] - tar_bbox[0]) / np.abs(np.sin(angle))
        h = tar_bbox[1] - tar_bbox[0]
        w = tar_bbox[3] - tar_bbox[2]
        d = 0.5 * np.sqrt(h * h + w * w)

        i_final = tar_pos[0] - np.sin(angle) * d
        j_final = tar_pos[1] + np.cos(angle) * d

        return [i_final, j_final]

    def get_front_nearest_obj_pos(self, curr_pos: List[float], curr_angle_deg: float, name: str) -> List[float]:
        contours, centers, bbox_list = self.get_pos(name)
        ids_list = self.select_front_objs(centers, curr_pos, curr_angle_deg)
        if not ids_list:
            return None

        front_centers = [centers[i] for i in ids_list]
        nearest_id = self.select_nearest_obj(front_centers, bbox_list, curr_pos)
        nearest_center = front_centers[nearest_id]
        return nearest_center

    def get_front_nearest_obj_pos_box(
        self, curr_pos: List[float], curr_angle_deg: float, name: str
    ) -> Tuple[List[float], List[float]]:
        contours, centers, bbox_list = self.get_pos(name)
        print("centers: ", centers)
        centers_cropped = [[x[0] - self.rmin, x[1] - self.cmin] for x in centers]
        ids_list = self.select_front_objs(centers, curr_pos, curr_angle_deg)
        print("ids_list: ", ids_list)
        if not ids_list:
            return None, None

        front_centers = [centers[i] for i in ids_list]
        front_bboxes = [bbox_list[i] for i in ids_list]
        front_centers_cropped = [[x[0] - self.rmin, x[1] - self.cmin] for x in front_centers]
        nearest_id = self.select_nearest_obj(front_centers, front_bboxes, curr_pos)
        nearest_center = front_centers[nearest_id]
        nearest_box = front_bboxes[nearest_id]
        return nearest_center, nearest_box

    def select_front_objs(
        self,
        centers: List[List[float]],
        curr_pos: List[float],
        curr_angle_deg: float,
        fov_deg: float = 90,
    ) -> List[int]:
        """
        Return a list of indices indicating the objects in the front
        Input:
            centers: a list of 2-element lists
            curr_pos: the robot's current position in array index space
            curr_angle_deg: the robot's current orientation in degree, up is zero, clock-wise
            fov_deg: the field of view of the camera
        """
        ids_list = []
        # i_org, j_org = curr_pos
        theta = curr_angle_deg * np.pi / 180
        fov_rad_2 = fov_deg * np.pi / 360
        pi_2 = np.pi / 2
        # change to normal x, y coordinate y up, x right
        # theta = np.pi / 2 - theta
        # y_org, x_org = curr_pos
        row_org, col_org = curr_pos
        # y_org = -y_org

        for c_i, c in enumerate(centers):
            row, col = c
            # y, x = c
            # y = -y

            # center_angle = np.arctan2(y - y_org, x - x_org)
            center_angle = np.arctan2(-col + col_org, -row + row_org)
            if (
                np.abs(center_angle - theta) < fov_rad_2
                or (theta > pi_2 and center_angle < -pi_2 and np.abs(2 * np.pi - theta + center_angle) < fov_rad_2)
                or (theta < -pi_2 and center_angle > pi_2 and np.abs(2 * np.pi - center_angle + theta) < fov_rad_2)
            ):
                print(f"select_front_objs: {theta}, {center_angle}, {fov_rad_2}")
                ids_list.append(c_i)

        return ids_list

    def find_middle_bewteen_contours(self, cona: List[List[float]], conb: List[List[float]]):
        min_pos = np.array([[self.rmin, self.cmin]])
        cona_np = np.array(cona).reshape((-1, 1, 2))
        conb_np = np.array(conb).reshape((1, -1, 2))
        dist_mat = cona_np - conb_np
        dist_mat = np.linalg.norm(dist_mat, axis=2)
        id = np.argmin(dist_mat)
        row, col = np.unravel_index(id, dist_mat.shape)
        pa = cona[row]
        pb = conb[col]
        middle = (pa + pb) / 2
        cona_cropped = cona - min_pos
        conb_cropped = conb - min_pos
        return middle

    def get_pos_in_between(
        self,
        curr_pos: List[float],
        curr_angle_deg: float,
        obj_a_name: str,
        obj_b_name: str,
    ) -> List[float]:
        contours_a, centers_a, bbox_list_a = self.get_pos(obj_a_name)
        contours_b, centers_b, bbox_list_b = self.get_pos(obj_b_name)

        ids_a_list = self.select_front_objs(centers_a, curr_pos, curr_angle_deg)
        ids_b_list = self.select_front_objs(centers_b, curr_pos, curr_angle_deg)
        if not ids_a_list or not ids_b_list:
            print(f"Can't find the middle point betwen {obj_a_name} and {obj_b_name}")
            return None

        contours_a = [contours_a[i] for i in ids_a_list]
        contours_b = [contours_b[i] for i in ids_b_list]
        front_bbox_list_a = [bbox_list_a[i] for i in ids_a_list]
        front_bbox_list_b = [bbox_list_b[i] for i in ids_b_list]
        ids_a_list = self.filter_small_objects(front_bbox_list_a)
        ids_b_list = self.filter_small_objects(front_bbox_list_b)
        if not ids_a_list or not ids_b_list:
            print(f"Can't find the middle point between {obj_a_name} and {obj_b_name}")
            return None

        ca = [x for i, x in enumerate(centers_a) if i in ids_a_list]
        cb = [x for i, x in enumerate(centers_b) if i in ids_b_list]
        cona = [x for i, x in enumerate(contours_a) if i in ids_a_list]
        conb = [x for i, x in enumerate(contours_b) if i in ids_b_list]

        ca_np = np.array(ca).reshape((-1, 1, 2))
        cb_np = np.array(cb).reshape((1, -1, 2))
        dist_mat = ca_np - cb_np
        dist_mat = np.linalg.norm(dist_mat, axis=2)

        middle_pos_mat = (ca_np + cb_np) / 2
        middle_to_curr_dist_mat = middle_pos_mat - np.array(curr_pos).reshape((1, 1, 2))
        middle_to_curr_dist_mat = np.linalg.norm(middle_to_curr_dist_mat, axis=-1)

        # dist_mat = ca_np @ cb_np.T
        id = np.argmin(middle_to_curr_dist_mat)
        row, col = np.unravel_index(id, middle_to_curr_dist_mat.shape)
        dist_to_curr = middle_to_curr_dist_mat[row, col]
        dist = dist_mat[row, col]

        pos = self.find_middle_bewteen_contours(cona[row], conb[col])
        return pos

    def get_left_pos(self, curr_pos: List[float], curr_angle_deg: float, name: str) -> List[float]:
        nearest_center, nearest_bbox = self.get_front_nearest_obj_pos_box(curr_pos, curr_angle_deg, name)
        if nearest_center is None:
            print("nearest center is None.")
            return [None, None]

        left_pos = self._get_left_pos(curr_pos, nearest_center, nearest_bbox)
        return left_pos

    def get_right_pos(self, curr_pos: List[float], curr_angle_deg: float, name: str) -> List[float]:
        nearest_center, nearest_bbox = self.get_front_nearest_obj_pos_box(curr_pos, curr_angle_deg, name)
        if nearest_center is None:
            return [None, None]

        right_pos = self._get_right_pos(curr_pos, nearest_center, nearest_bbox)
        return right_pos

    def get_delta_angle_to(self, curr_pos: List[float], curr_angle_deg: float, name: str):
        contours, centers, bbox_list = self.get_pos(name)
        nearest_id = self.select_nearest_obj(centers, bbox_list, curr_pos)
        nearest_center = centers[nearest_id]
        dx = nearest_center[0] - curr_pos[0]  # down
        dy = nearest_center[1] - curr_pos[1]  # right

        angle = np.arctan2(dy, -dx)  # upward zero, turn right positive
        angle = angle * 180.0 / np.pi

        turn_right_angle = angle - curr_angle_deg
        turn_right_angle = np.mod(turn_right_angle, 360)
        if turn_right_angle < -180:
            turn_right_angle += 360
        elif turn_right_angle > 180:
            turn_right_angle -= 360

        return turn_right_angle

    def get_north_pos(self, curr_pos: List[float], curr_angle_deg: float, name: str):
        dist = 10
        nearest_center, nearest_box = self.get_front_nearest_obj_pos_box(curr_pos, curr_angle_deg, name)
        if nearest_center is None:
            return ["stop"]

        pos = [nearest_box[0] - dist, nearest_center[1]]
        return pos

    def get_south_pos(self, curr_pos: List[float], curr_angle_deg: float, name: str):
        dist = 10
        nearest_center, nearest_box = self.get_front_nearest_obj_pos_box(curr_pos, curr_angle_deg, name)
        if nearest_center is None:
            return ["stop"]

        pos = [nearest_box[1] + dist, nearest_center[1]]
        return pos

    def get_west_pos(self, curr_pos: List[float], curr_angle_deg: float, name: str):
        dist = 10
        nearest_center, nearest_box = self.get_front_nearest_obj_pos_box(curr_pos, curr_angle_deg, name)
        if nearest_center is None:
            return ["stop"]

        pos = [nearest_center[0], nearest_box[2] - dist]
        return pos

    def get_east_pos(self, curr_pos: List[float], curr_angle_deg: float, name: str):
        dist = 10
        nearest_center, nearest_box = self.get_front_nearest_obj_pos_box(curr_pos, curr_angle_deg, name)
        if nearest_center is None:
            return ["stop"]

        pos = [nearest_center[0], nearest_box[3] + dist]
        return pos

    def get_3d_nearest_and_similarity_color_pos(
        self,
        curr_pos: List[float],
        name: str,
        target_colors: List[List[int]],
        floor: int=None,
        color_weight: float = 0.5,
        vis: bool = False
    ) -> List[float]:
        """
        Get nearest position considering color information
        :param curr_pos: Current [row, col] position
        :param name: Object category name
        :param target_colors: List of target RGB colors (e.g., [[255,0,0]] or [[255,0,0], [255,255,255]])
        :param color_weight: Weight of color similarity (0-1)
        :param vis: Whether to visualize
        """
        # 获取目标位置信息和颜色分布
        contours, centers, bbox_list, color_dists, avg_heights = self.get_pos_color_and_height(name, floor, vis)

        # 过滤小物体
        ids_list = self.filter_small_objects(bbox_list, name, area_thres=5)
        contours = [contours[i] for i in ids_list]
        centers = [centers[i] for i in ids_list]
        bbox_list = [bbox_list[i] for i in ids_list]
        color_dists = [color_dists[i] for i in ids_list]
        avg_heights = [avg_heights[i] for i in ids_list]

        if len(centers) == 0:
            return [curr_pos, None]

        # 计算颜色匹配分数
        color_scores = np.ones(len(centers))

        if target_colors:
            for i, color_dist in enumerate(color_dists):
                main_colors = color_dist["main_colors"]
                score = 0.0

                if len(target_colors) == 1:
                    # 单个目标颜色：只考虑相似度最高的值
                    best_match_score = 0.0
                    for color_info in main_colors:
                        # 使用余弦相似度计算颜色匹配度
                        obj_color = np.array(color_info["color"])
                        tgt_color = np.array(target_colors[0])

                        # 计算余弦相似度
                        # dot_product = np.dot(obj_color, tgt_color)
                        # norm_obj = np.linalg.norm(obj_color)
                        # norm_tgt = np.linalg.norm(tgt_color)
                        # similarity = dot_product / (norm_obj * norm_tgt + 1e-6)
                        distance = np.linalg.norm(obj_color - tgt_color)
                        similarity = 1 / (1 + distance/441)
                        # 更新最佳匹配
                        if similarity > best_match_score:
                            best_match_score = similarity

                    # 单个颜色直接使用最佳匹配分数
                    color_scores[i] = best_match_score

                elif len(target_colors) >= 2:
                    # 两个目标颜色：分别匹配主色和次主色
                    primary_match = 0.0
                    secondary_match = 0.0

                    # 匹配第一个目标颜色（主色）
                    best_primary_match = 0.0
                    primary_idx = -1
                    for j, color_info in enumerate(main_colors):
                        obj_color = np.array(color_info["color"])
                        tgt_color = np.array(target_colors[0])
                        # 余弦相似度
                        # dot_product = np.dot(obj_color, tgt_color)
                        # norm_obj = np.linalg.norm(obj_color)
                        # norm_tgt = np.linalg.norm(tgt_color)
                        # similarity = dot_product / (norm_obj * norm_tgt + 1e-6)
                        distance = np.linalg.norm(obj_color - tgt_color)
                        similarity = 1 / (1 + distance/441)
                        if similarity > best_primary_match:
                            best_primary_match = similarity
                            primary_idx = j

                    primary_match = best_primary_match

                    # 匹配第二个目标颜色（剩余颜色）
                    best_secondary_match = 0.0
                    for j, color_info in enumerate(main_colors):
                        # 跳过已匹配的主色
                        if j == primary_idx:
                            continue
                        obj_color = np.array(color_info["color"])
                        tgt_color = np.array(target_colors[1])
                        # 余弦相似度
                        # dot_product = np.dot(obj_color, tgt_color)
                        # norm_obj = np.linalg.norm(obj_color)
                        # norm_tgt = np.linalg.norm(tgt_color)
                        # similarity = dot_product / (norm_obj * norm_tgt + 1e-6)
                        distance = np.linalg.norm(obj_color - tgt_color)
                        similarity = 1 / (1 + distance/441)
                        if similarity > best_secondary_match:
                            best_secondary_match = similarity

                    # 如果只有一个主色，使用它匹配第二个目标颜色
                    if best_secondary_match == 0 and len(main_colors) > 0:
                        obj_color = np.array(main_colors[0]["color"])
                        tgt_color = np.array(target_colors[1])

                        dot_product = np.dot(obj_color, tgt_color)
                        norm_obj = np.linalg.norm(obj_color)
                        norm_tgt = np.linalg.norm(tgt_color)
                        best_secondary_match = dot_product / (norm_obj * norm_tgt + 1e-6)

                    secondary_match = best_secondary_match

                    # 计算平均分数
                    color_scores[i] = (primary_match + secondary_match) / 2.0

        # 计算距离分数
        dist_scores = np.zeros(len(centers))
        for i, center in enumerate(centers):
            dist = np.sqrt((center[0]-curr_pos[0])**2 + (center[1]-curr_pos[1])**2)
            dist_scores[i] = 1 / (dist + 1e-6)  # 避免除以零

        if np.max(dist_scores) > 0:
            dist_scores = dist_scores / np.max(dist_scores)

        # 综合评分 = 颜色匹配分数 * 颜色权重 + 距离分数 * (1 - 颜色权重)
        combined_scores = (color_scores * color_weight) + (dist_scores * (1 - color_weight))
        best_id = np.argmax(combined_scores)
        if vis:
            # 获取障碍物地图
            obs_map = self.get_customized_obstacle_cropped()
            if obs_map is None or obs_map.size == 0:
                print("Error: obs_map is empty or invalid!")
                return curr_pos
            # 创建三通道 BGR 图像
            obs_map_vis = (obs_map * 255).astype(np.uint8)
            obs_map_vis = np.stack([obs_map_vis] * 3, axis=-1)

            # 绘制选中轮廓
            contour = contours[best_id]
            contour_cv2 = contour[:, [1, 0]]  # [row, col] → [col, row]
            # 将全图坐标转换为裁剪图像坐标
            contour_cv2[:, 0] -= self.cmin  # col - cmin
            contour_cv2[:, 1] -= self.rmin  # row - rmin
            cv2.drawContours(obs_map_vis, [contour_cv2], 0, (0, 255, 0), 3)  # 绿色轮廓

            # 绘制选中物体的中心点（红点）
            selected_center = centers[best_id]
            center_col = int(selected_center[1]) - self.cmin  # col - cmin
            center_row = int(selected_center[0]) - self.rmin  # row - rmin
            cv2.circle(
                obs_map_vis,
                (center_col, center_row),
                radius=5,
                color=(0, 0, 255),
                thickness=-1
            )

            # 保存图像以供调试
            # cv2.imwrite("debug_output.png", obs_map_vis)
            print(f"######nav object name : {name} #######")
            # 显示最终图像
            cv2.imshow("Selected Contour and Center", obs_map_vis)
            cv2.waitKey()

        return self.nearest_point_on_polygon(curr_pos, contours[best_id]), avg_heights[best_id]
