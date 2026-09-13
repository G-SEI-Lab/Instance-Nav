from __future__ import annotations

# Adapted from VLMaps (https://github.com/vlmaps/vlmaps).
# Copyright (c) 2023 Tom-Huang. MIT license; see THIRD_PARTY_LICENSES/VLMaps-LICENSE.

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import logging
import clip
import cv2
from skimage import color
import numpy as np
from omegaconf import DictConfig
from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter
import torch
from instance_nav.utils.visualize_utils import pool_3d_label_to_2d, pool_filter_by_height_3d_label_to_2d

# from utils.ai2thor_constant import ai2thor_class_list
# from utils.clip_mapping_utils import load_map
# from utils.planning_utils import (
#     find_similar_category_id,
#     get_dynamic_obstacles_map,
#     get_lseg_score,
#     get_segment_islands_pos,
#     mp3dcat,
#     segment_lseg_map,
# )
from instance_nav.utils.mapping_utils import load_3d_map
from instance_nav.map.map import Map
from instance_nav.utils.index_utils import find_similar_category_id, get_segment_islands_pos, get_dynamic_obstacles_map_3d, get_main_colors
from instance_nav.utils.clip_utils import get_lseg_score


class PrebuiltMap(Map):
    def __init__(self, map_config: DictConfig, data_dir: str = ""):
        super().__init__(map_config, data_dir=data_dir)
        self.scores_mat = None
        self.categories = None

    def load_map(self, data_dir: str) -> bool:
        self._setup_paths(data_dir)
        print(self.data_dir)
        if self.map_config.pose_info.pose_type in ("mobile_base", "mobile_base_2"):
            self.map_save_path = Path(data_dir) / "vlmap" / "vlmaps.h5df"
            print(self.map_save_path)
            if not self.map_save_path.exists():
                raise FileNotFoundError(f"Prebuilt map not found: {self.map_save_path}")
            (
                self.mapped_iter_list,
                self.grid_feat,
                self.grid_pos,
                self.weight,
                self.occupied_ids,
                self.grid_rgb,
                self.pcd_min,
                self.pcd_max
            ) = load_3d_map(self.map_save_path)
        else:
            raise ValueError("Navigation release supports prebuilt mobile_base maps only")

        return True

    def _init_clip(self, clip_version="ViT-B/32"):
        if hasattr(self, "clip_model"):
            print("clip model is already initialized")
            return
        if torch.cuda.is_available():
            self.device = "cuda:0"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.clip_version = clip_version
        self.clip_feat_dim = {
            "RN50": 1024,
            "RN101": 512,
            "RN50x4": 640,
            "RN50x16": 768,
            "RN50x64": 1024,
            "ViT-B/32": 512,
            "ViT-B/16": 512,
            "ViT-L/14": 768,
        }[self.clip_version]
        print("Loading CLIP model...")
        self.clip_model, self.preprocess = clip.load(self.clip_version)  # clip.available_models()
        self.clip_model.to(self.device).eval()

    def init_categories(self, categories: List[str]) -> np.ndarray:
        self.categories = categories
        self.scores_mat = get_lseg_score(
            self.clip_model,
            self.categories,
            self.grid_feat,
            self.clip_feat_dim,
            use_multiple_templates=True,
            add_other=True,
        )  # score for name and other
        scene_data_dir = self.data_dir
        # save_path = scene_data_dir / "vlmap_cam" / "scores_mat.npy"
        # np.save(save_path, self.scores_mat)
        # print(f"{save_path} is saved.")
        return self.scores_mat

    def index_map(self, language_desc: str, with_init_cat: bool = True):
        if with_init_cat and self.scores_mat is not None and self.categories is not None:
            cat_id = find_similar_category_id(language_desc, self.categories)
            scores_mat = self.scores_mat
        else:
            if with_init_cat:
                raise Exception(
                    "Categories are not preloaded. Call init_categories(categories: List[str]) to initialize categories."
                )
            scores_mat = get_lseg_score(
                self.clip_model,
                [language_desc],
                self.grid_feat,
                self.clip_feat_dim,
                use_multiple_templates=True,
                add_other=True,
            )  # score for name and other
            cat_id = 0
        # logging.info(f"self.categories: {self.categories}")
        # logging.info(f"cat_id: {cat_id}")
        # logging.info(f"catscores_mat_id: {scores_mat.shape}")
        max_ids = np.argmax(scores_mat, axis=1)
        mask = max_ids == cat_id
        return mask

    def index_map_v2(self, language_desc: str, with_init_cat: bool = True):
        if with_init_cat and self.scores_mat is not None and self.categories is not None:
            cat_id = find_similar_category_id(language_desc, self.categories)
            scores_mat = self.scores_mat
        else:
            if with_init_cat:
                raise Exception(
                    "Categories are not preloaded. Call init_categories(categories: List[str]) to initialize categories."
                )
            scores_mat = get_lseg_score(
                self.clip_model,
                [language_desc],
                self.grid_feat,
                self.clip_feat_dim,
                use_multiple_templates=True,
                add_other=True,
            )  # score for name and other
            cat_id = 0
        # logging.info(f"self.categories: {self.categories}")
        # logging.info(f"cat_id: {cat_id}")
        # logging.info(f"catscores_mat_id: {scores_mat.shape}")
        max_ids = np.argmax(scores_mat, axis=1)
        mask = max_ids == cat_id
        return mask, max_ids

    def customize_obstacle_map(
        self,
        potential_obstacle_names: List[str],
        obstacle_names: List[str],
        passable_names: List[str],
        vis: bool = False,
    ):
        if self.obstacles_cropped is None and self.obstacles_map is None:
            self.generate_obstacle_map()
        if not hasattr(self, "clip_model"):
            print("init_clip in customize obstacle map")
            self._init_clip()

        self.obstacles_new_cropped = get_dynamic_obstacles_map_3d(
            self.clip_model,
            self.obstacles_cropped,
            self.map_config.potential_obstacle_names,
            self.map_config.obstacle_names,
            self.grid_feat,
            self.grid_pos,
            self.rmin,
            self.cmin,
            self.clip_feat_dim,
            h_min = self.min_height,
            h_max = self.max_height,
            cs = self.cs,
            vis=vis,
        )
        # 对一个二值地图（binary_map）进行膨胀处理，同时可选地应用高斯滤波
        self.obstacles_new_cropped = Map._dilate_map(
            self.obstacles_new_cropped == 0,
            self.map_config.dilate_iter,
            self.map_config.gaussian_sigma,
        )

        self.obstacles_new_cropped = self.obstacles_new_cropped == 0

        envelope_map_reverse = ~self.envelope_cropped
        envelope_cropped_filter = Map._dilate_map(
            envelope_map_reverse,
            self.map_config.dilate_iter,
            self.map_config.gaussian_sigma,
            use_dilation=False,
        )
        self.passable_map = ~(np.logical_or(self.obstacles_new_cropped == 0, envelope_cropped_filter ==0))

        # x_start, y_start = 62, 205
        # x_end, y_end = 71, 217

        # # 确保坐标在数组范围内
        # self.edit_obstacle(x_start, y_start, x_end, y_end)


        if vis:
            cv2.imshow("envelope_cropped_filter", (envelope_cropped_filter * 255).astype(np.uint8))
            cv2.imshow("obstacles_new_cropped", (self.obstacles_new_cropped * 255).astype(np.uint8))
            cv2.imshow("Safe Passable Area", (self.passable_map * 255).astype(np.uint8))
            cv2.waitKey()
        return
        #可通行区域
        combined_potential_classes = list(set(self.map_config.potential_obstacle_names + self.map_config.passable_names))

        # 创建空的障碍物地图（因为我们只关心语义识别结果）
        dummy_obstacles = np.zeros_like(self.obstacles_cropped, dtype=bool)

        # 识别可通行区域
        passable_area_cropped = get_dynamic_obstacles_map_3d(
            self.clip_model,
            dummy_obstacles,
            combined_potential_classes,  # 合并后的潜在类别，提升分割效果
            self.map_config.passable_names,              # 实际关心的可通行类别
            self.grid_feat,
            self.grid_pos,
            self.rmin,
            self.cmin,
            self.clip_feat_dim,
            vis=False,
        )

        passable_area_new_cropped = Map._dilate_map(
            passable_area_cropped == 0,
            5,
            self.map_config.gaussian_sigma,
        )
        # get_dynamic_obstacles_map_3d返回的是识别出的目标区域（在这里是可通行区域）
        passable_map = passable_area_cropped == 0

        # 6. 从可通行区域中减去膨胀的障碍物区域，得到安全可通行区域
        # 确保膨胀后的障碍物不会超出可通行区域
        self.safe_passable_map = np.logical_and(
            passable_map,
            self.obstacles_new_cropped
        )

        # 7. 可视化结果
        if vis:
            # 显示原始障碍物
            cv2.imshow("Original Obstacles", ((self.obstacles_cropped == 0) * 255).astype(np.uint8))
            # 显示可通行区域
            cv2.imshow("Passable Area", (passable_map * 255).astype(np.uint8))
            cv2.imshow("passable_area_new_cropped", (passable_area_new_cropped * 255).astype(np.uint8))
            # 显示膨胀后的障碍物
            cv2.imshow("Dilated Obstacles", (self.obstacles_new_cropped * 255).astype(np.uint8))
            # 显示安全可通行区域
            cv2.imshow("Safe Passable Area", (self.safe_passable_map * 255).astype(np.uint8))
            cv2.waitKey()

    def customize_obstacle_3d_map(
        self,
        obstacles_cropped,
        potential_obstacle_names: List[str],
        obstacle_names: List[str],
        h_min: float = 0.0,
        h_max: float = 1.5,
        vis: bool = False,
    ):
        if not hasattr(self, "clip_model"):
            print("init_clip in customize obstacle map")
            self._init_clip()

        obstacles_new_cropped = get_dynamic_obstacles_map_3d(
            self.clip_model,
            obstacles_cropped,
            potential_obstacle_names,
            obstacle_names,
            self.grid_feat,
            self.grid_pos,
            self.rmin,
            self.cmin,
            self.clip_feat_dim,
            h_min = h_min,
            h_max = h_max,
            cs = self.cs,
            vis=vis,
        )
        # 对一个二值地图（binary_map）进行膨胀处理，同时可选地应用高斯滤波
        obstacles_new_cropped = Map._dilate_map(
            obstacles_new_cropped == 0,
            self.map_config.dilate_iter,
            self.map_config.gaussian_sigma,
        )

        obstacles_new_cropped = obstacles_new_cropped == 0

        envelope_map_reverse = ~self.envelope_cropped
        envelope_cropped_filter = Map._dilate_map(
            envelope_map_reverse,
            self.map_config.dilate_iter,
            self.map_config.gaussian_sigma,
            use_dilation=False,
        )
        passable_map = ~(np.logical_or(obstacles_new_cropped == 0, envelope_cropped_filter ==0))

        # x_start, y_start = 62, 205
        # x_end, y_end = 71, 217

        # # 确保坐标在数组范围内
        # self.edit_obstacle(x_start, y_start, x_end, y_end)

        return passable_map

    def edit_obstacle(self, x_start, y_start, x_end, y_end):
        h, w = self.passable_map.shape
        x_start = max(0, min(x_start, w - 1))
        x_end = max(0, min(x_end + 1, w))  # 结束坐标+1（切片右开区间）
        y_start = max(0, min(y_start, h - 1))
        y_end = max(0, min(y_end + 1, h))   # 结束坐标+1（切片右开区间）

        # 设置指定区域为True
        self.passable_map[y_start:y_end, x_start:x_end] = True


    # def load_categories(self, categories: List[str] = None):
    #     if categories is None:
    #         if self.map_config["categories"] == "mp3d":
    #             categories = mp3dcat.copy()
    #         elif self.map_config["categories"] == "ai2thor":
    #             categories = ai2thor_class_list.copy()

    #     predicts = segment_lseg_map(self.clip_model, categories, self.map_cropped, self.clip_feat_dim)
    #     no_map_mask = self.obstacles_new_cropped > 0  # free space in the map

    #     self.labeled_map_cropped = predicts.reshape((self.xmax - self.xmin + 1, self.ymax - self.ymin + 1))
    #     self.labeled_map_cropped[no_map_mask] = -1
    #     labeled_map = -1 * np.ones((self.map.shape[0], self.map.shape[1]))

    #     labeled_map[self.xmin : self.xmax + 1, self.ymin : self.ymax + 1] = self.labeled_map_cropped

    #     self.categories = categories
    #     self.labeled_map_full = labeled_map

    # def load_region_categories(self, categories: List[str]):
    #     if "other" not in categories:
    #         self.region_categories = ["other"] + categories
    #     predicts = segment_lseg_map(
    #         self.clip_model, self.region_categories, self.map_cropped, self.clip_feat_dim, add_other=False
    #     )
    #     self.labeled_region_map_cropped = predicts.reshape((self.xmax - self.xmin + 1, self.ymax - self.ymin + 1))

    # def get_region_predict_mask(self, name: str) -> np.ndarray:
    #     assert self.region_categories
    #     cat_id = find_similar_category_id(name, self.region_categories)
    #     mask = self.labeled_map_cropped == cat_id
    #     return mask

    # def get_predict_mask(self, name: str) -> np.ndarray:
    #     cat_id = find_similar_category_id(name, self.categories)
    #     return self.labeled_map_cropped == cat_id

    # def get_distribution_map(self, name: str) -> np.ndarray:
    #     assert self.categories
    #     cat_id = find_similar_category_id(name, self.categories)
    #     if self.scores_map is None:
    #         scores_list = get_lseg_score(self.clip_model, self.categories, self.map_cropped, self.clip_feat_dim)
    #         h, w = self.map_cropped.shape[:2]
    #         self.scores_map = scores_list.reshape((h, w, len(self.categories)))
    #     # labeled_map_cropped = self.labeled_map_cropped.copy()
    #     return self.scores_map[:, :, cat_id]

    def get_pos(self, name: str) -> Tuple[List[List[int]], List[List[float]], List[np.ndarray], Any]:
        """
        Get the contours, centers, and bbox list of a certain category
        on a full map
        """
        assert self.categories
        # cat_id = find_similar_category_id(name, self.categories)
        # labeled_map_cropped = self.scores_mat.copy()  # (N, C) N: number of voxels, C: number of categories
        # labeled_map_cropped = np.argmax(labeled_map_cropped, axis=1)  # (N,)
        # pc_mask = labeled_map_cropped == cat_id # (N,)
        # self.grid_pos[pc_mask]
        pc_mask = self.index_map(name, with_init_cat=True)
        mask_2d = pool_3d_label_to_2d(pc_mask, self.grid_pos, self.gs)
        mask_2d = mask_2d[self.rmin : self.rmax + 1, self.cmin : self.cmax + 1]
        # print(f"showing mask for object cat {name}")
        # cv2.imshow(f"mask_{name}", (mask_2d.astype(np.float32) * 255).astype(np.uint8))
        # cv2.waitKey()

        foreground = binary_closing(mask_2d, iterations=3)
        foreground = gaussian_filter(foreground.astype(float), sigma=0.8, truncate=3)
        foreground = foreground > 0.5
        # cv2.imshow(f"mask_{name}_gaussian", (foreground * 255).astype(np.uint8))
        foreground = binary_dilation(foreground)
        # cv2.imshow(f"mask_{name}_processed", (foreground.astype(np.float32) * 255).astype(np.uint8))
        # cv2.waitKey()

        contours, centers, bbox_list, _ = get_segment_islands_pos(foreground, 1)
        # print("centers", centers)

        # whole map position
        for i in range(len(contours)):
            centers[i][0] += self.rmin
            centers[i][1] += self.cmin
            bbox_list[i][0] += self.rmin
            bbox_list[i][1] += self.rmin
            bbox_list[i][2] += self.cmin
            bbox_list[i][3] += self.cmin
            for j in range(len(contours[i])):
                contours[i][j, 0] += self.rmin
                contours[i][j, 1] += self.cmin

        return contours, centers, bbox_list

    def get_pos_and_color(self, name: str, vis: bool = False) -> Tuple[List[List[int]], List[List[float]], List[np.ndarray], List[Dict]]:
        """
        Get the contours, centers, bbox list and color distributions of a certain category
        on a full map
        """
        assert self.categories
        # 获取目标类别的3D点云掩码
        pc_mask = self.index_map(name, with_init_cat=True)
        # pc_mask_index = np.where(pc_mask)[0]
        mask_2d = pool_filter_by_height_3d_label_to_2d(pc_mask, self.grid_pos, self.gs, self.cs, self.min_height, self.max_height)
        mask_2d = mask_2d[self.rmin : self.rmax + 1, self.cmin : self.cmax + 1]
        # mask_2d_index = np.stack(np.where(mask_2d), axis=1)
        if vis:
            cv2.imshow(f"mask_{name}", (mask_2d.astype(np.float32) * 255).astype(np.uint8))
            cv2.waitKey()
        # 创建彩色mask图像（裁剪区域大小）
        color_mask = np.zeros((mask_2d.shape[1], mask_2d.shape[0], 3), dtype=np.uint8)

        foreground = binary_closing(mask_2d, iterations=3)
        foreground = gaussian_filter(foreground.astype(float), sigma=0.8, truncate=3)
        foreground = foreground > 0.5
        foreground = binary_dilation(foreground)
        # foreground_index = np.stack(np.where(foreground), axis=1)
        contours, centers, bbox_list, _ = get_segment_islands_pos(foreground, 1)

        contours_reverse = [None] * len(contours)
        for i in range(len(contours)):
            contours_reverse[i] = contours[i][:, [1, 0]]  # 从 (row,col) 转为 (col,row)

        # 存储每个物体的颜色分布信息
        color_distributions = []

        # 为每个轮廓创建点云索引列表
        contour_indices = [[] for _ in range(len(contours))]
        r_c = []
        local_r_c = []
        # 遍历所有属于目标类别的体素
        for idx in np.where(pc_mask)[0]:
            # 获取体素的全局坐标
            r, c, h = self.grid_pos[idx]
            r_c.append([r,c])
            # 转换为裁剪区域坐标
            local_r = r - self.rmin
            local_c = c - self.cmin
            local_r_c.append([local_r,local_c])
            # 检查是否在裁剪区域内
            if 0 <= local_r < foreground.shape[0] and 0 <= local_c < foreground.shape[1]:
                # 检查点是否在某个轮廓内
                point = (local_c, local_r)  # OpenCV格式 (x,y)

                for contour_idx, contour in enumerate(contours_reverse):
                    # 检查点是否在当前轮廓内
                    if cv2.pointPolygonTest(contour, point, False) >= 0:
                        contour_indices[contour_idx].append(idx)
                        break

        # 处理每个轮廓的点云
        for i, indices in enumerate(contour_indices):
            # 获取当前轮廓的点云索引
            obj_indices = indices

            # 计算颜色分布
            color_dist = {}
            if obj_indices:

                # mask = np.zeros(len(self.grid_pos), dtype=bool)
                # mask[obj_indices] = True
                # from instance_nav.utils.visualize_utils import visualize_masked_map_3d
                # visualize_masked_map_3d(self.grid_pos, mask, self.grid_rgb)

                obj_colors = self.grid_rgb[obj_indices]
                n_points = len(obj_colors)

                # 根据点数选择合适的聚类方法
                if n_points < 3:
                    # 点数太少，直接使用平均颜色
                    avg_color = np.mean(obj_colors, axis=0).astype(int).tolist()
                    color_dist = {"main_colors": [{"color": avg_color, "proportion": 1.0}]}
                    main_color = tuple(int(c) for c in avg_color)
                else:
                    # # 使用K-means聚类识别主要颜色
                    from sklearn.cluster import KMeans
                    try:
                        n_clusters = min(2, n_points)
                        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(obj_colors)
                        cluster_centers = kmeans.cluster_centers_.astype(int)
                        cluster_labels, counts = np.unique(kmeans.labels_, return_counts=True)
                        total = len(obj_colors)

                        # 按占比排序颜色
                        sorted_indices = np.argsort(counts)[::-1]
                        main_colors = []

                        for idx in sorted_indices:
                            color = cluster_centers[idx].tolist()
                            proportion = counts[idx] / total
                            main_colors.append({"color": color, "proportion": proportion})

                        color_dist = {"main_colors": main_colors}
                        main_color = tuple(int(c) for c in main_colors[0]["color"])
                    except Exception as e:
                        print(f"KMeans聚类失败: {e}")
                        avg_color = np.mean(obj_colors, axis=0).astype(int).tolist()
                        color_dist = {"main_colors": [{"color": avg_color, "proportion": 1.0}]}
                        main_color = tuple(int(c) for c in avg_color)
                    #使用DBSCAN聚类识别主要颜色
                    # from sklearn.cluster import DBSCAN
                    # from sklearn.metrics import silhouette_score

                    # obj_colors = np.array(obj_colors)  # 确保是numpy数组
                    # n_points_func = len(obj_colors)  # 用新变量名避免与外部n_points冲突
                    # if n_points_func < 4:
                    #     # 点太少，直接返回平均颜色
                    #     avg_color = np.mean(obj_colors, axis=0).astype(int).tolist()
                    #     main_colors = [{"color": avg_color, "proportion": 1.0}]
                    # else:
                    #     # 归一化颜色到[0,1]以便DBSCAN的eps参数更直观
                    #     obj_colors_norm = obj_colors / 255.0

                    #     # 尝试不同的eps值，选择最优的（簇数在2到max_clusters之间）
                    #     eps_values = np.linspace(0.01, 0.3, 10)  # 从紧簇到松簇
                    #     min_samples = max(3, n_points_func // 20)  # 动态最小样本
                    #     best_eps = None
                    #     best_score = -1
                    #     best_labels = None

                    #     max_clusters = 2  # 与原K-means保持一致，最多2个主色
                    #     max_possible_clusters = min(max_clusters, n_points_func // min_samples)
                    #     for eps in eps_values:
                    #         db = DBSCAN(eps=eps, min_samples=min_samples).fit(obj_colors_norm)
                    #         labels = db.labels_
                    #         core_mask = labels != -1
                    #         unique_labels = set(labels[core_mask])
                    #         n_clusters = len(unique_labels)

                    #         if n_clusters < 2 or n_clusters > max_possible_clusters:
                    #             continue

                    #         n_core_points = np.sum(core_mask)
                    #         if n_core_points < 2:
                    #             continue

                    #         # 只用非噪声点计算silhouette分数
                    #         score = silhouette_score(obj_colors_norm[core_mask], labels[core_mask])
                    #         if score > best_score:
                    #             best_score = score
                    #             best_eps = eps
                    #             best_labels = labels

                    #     # 如果没有找到合适的模型，退化为1簇平均色
                    #     if best_labels is None:
                    #         avg_color = np.mean(obj_colors, axis=0).astype(int).tolist()
                    #         main_colors = [{"color": avg_color, "proportion": 1.0}]
                    #     else:
                    #         # 统计簇分布（忽略噪声）
                    #         core_mask = best_labels != -1
                    #         unique_labels, counts = np.unique(best_labels[core_mask], return_counts=True)
                    #         total = n_points_func  # 比例基于总点数，包括噪声

                    #         # 计算每个簇的中心颜色
                    #         cluster_centers = []
                    #         for label in unique_labels:
                    #             cluster_points = obj_colors[best_labels == label]
                    #             center = np.mean(cluster_points, axis=0).astype(int).tolist()
                    #             cluster_centers.append(center)

                    #         # 排序按比例降序
                    #         sorted_indices = np.argsort(counts)[::-1]
                    #         main_colors = []
                    #         for idx in sorted_indices:
                    #             color = cluster_centers[idx]
                    #             proportion = counts[idx] / total
                    #             main_colors.append({"color": color, "proportion": proportion})

                    # # 封装结果（原函数返回值直接作为main_colors）
                    # color_dist = {"main_colors": main_colors}
                    # main_color = tuple(int(c) for c in main_colors[0]["color"])

            else:
                # 没有颜色数据，使用黑色
                color_dist = {"main_colors": [{"color": [0, 0, 0], "proportion": 1.0}]}
                main_color = (0, 0, 0)

            color_distributions.append(color_dist)

            # 使用主色填充物体区域（裁剪区域内坐标）
            bgr_color = (main_color[2], main_color[1], main_color[0])
            cv2.drawContours(color_mask, [contours[i].astype(np.int32)], -1, bgr_color, thickness=cv2.FILLED)

            # 转换坐标到全局地图（与原始代码保持一致）
            centers[i][0] += self.rmin
            centers[i][1] += self.cmin
            bbox_list[i][0] += self.rmin
            bbox_list[i][1] += self.rmin
            bbox_list[i][2] += self.cmin
            bbox_list[i][3] += self.cmin
            for j in range(len(contours[i])):
                contours[i][j, 0] += self.rmin
                contours[i][j, 1] += self.cmin

        # 可视化彩色mask（裁剪区域内）
        if vis:
            # 创建用于可视化的轮廓（转换回(row,col)格式以匹配mask_2d坐标系）
            visualization_contours = []
            for contour in contours:
                # 将轮廓从OpenCV格式(col,row)转回图像数组格式(row,col)
                vis_contour = contour[:, [1, 0]].copy()  # 交换回(row,col)
                visualization_contours.append(vis_contour.astype(np.int32))

            # 添加轮廓边界（白色）
            contour_mask = np.zeros_like(color_mask)
            for contour in visualization_contours:
                cv2.drawContours(contour_mask, [contour], -1, (255, 255, 255), 1)

            # 叠加轮廓到彩色mask
            combined_mask = cv2.addWeighted(color_mask, 0.8, contour_mask, 0.2, 0)
            combined_mask_display = combined_mask.transpose(1, 0, 2)
            # 显示彩色mask
            logging.debug(f"color_mask_name: {name}")
            cv2.imshow(f"color_mask_{name}", combined_mask_display)
            cv2.waitKey()

        return contours, centers, bbox_list, color_distributions

    def get_pos_and_color_quantization(self, name: str, vis: bool = False) -> Tuple[List[List[int]], List[List[float]], List[np.ndarray], List[Dict]]:
        """
        Get the contours, centers, bbox list and color distributions of a certain category
        on a full map. Uses Color Quantization (Histogram-based method) for ablation study.
        """
        assert self.categories
        # 获取目标类别的3D点云掩码
        pc_mask = self.index_map(name, with_init_cat=True)
        # pc_mask_index = np.where(pc_mask)[0]
        mask_2d = pool_filter_by_height_3d_label_to_2d(pc_mask, self.grid_pos, self.gs, self.cs, self.min_height, self.max_height)
        mask_2d = mask_2d[self.rmin : self.rmax + 1, self.cmin : self.cmax + 1]
        # mask_2d_index = np.stack(np.where(mask_2d), axis=1)
        if vis:
            cv2.imshow(f"mask_{name}", (mask_2d.astype(np.float32) * 255).astype(np.uint8))
            cv2.waitKey()
        # 创建彩色mask图像（裁剪区域大小）
        color_mask = np.zeros((mask_2d.shape[1], mask_2d.shape[0], 3), dtype=np.uint8)

        foreground = binary_closing(mask_2d, iterations=3)
        foreground = gaussian_filter(foreground.astype(float), sigma=0.8, truncate=3)
        foreground = foreground > 0.5
        foreground = binary_dilation(foreground)
        # foreground_index = np.stack(np.where(foreground), axis=1)
        contours, centers, bbox_list, _ = get_segment_islands_pos(foreground, 1)

        contours_reverse = [None] * len(contours)
        for i in range(len(contours)):
            contours_reverse[i] = contours[i][:, [1, 0]]  # 从 (row,col) 转为 (col,row)

        # 存储每个物体的颜色分布信息
        color_distributions = []

        # 为每个轮廓创建点云索引列表
        contour_indices = [[] for _ in range(len(contours))]
        r_c = []
        local_r_c = []
        # 遍历所有属于目标类别的体素
        for idx in np.where(pc_mask)[0]:
            # 获取体素的全局坐标
            r, c, h = self.grid_pos[idx]
            r_c.append([r,c])
            # 转换为裁剪区域坐标
            local_r = r - self.rmin
            local_c = c - self.cmin
            local_r_c.append([local_r,local_c])
            # 检查是否在裁剪区域内
            if 0 <= local_r < foreground.shape[0] and 0 <= local_c < foreground.shape[1]:
                # 检查点是否在某个轮廓内
                point = (local_c, local_r)  # OpenCV格式 (x,y)

                for contour_idx, contour in enumerate(contours_reverse):
                    # 检查点是否在当前轮廓内
                    if cv2.pointPolygonTest(contour, point, False) >= 0:
                        contour_indices[contour_idx].append(idx)
                        break

        # 处理每个轮廓的点云
        for i, indices in enumerate(contour_indices):
            # 获取当前轮廓的点云索引
            obj_indices = indices

            # 计算颜色分布
            color_dist = {}
            if obj_indices:

                # mask = np.zeros(len(self.grid_pos), dtype=bool)
                # mask[obj_indices] = True
                # from instance_nav.utils.visualize_utils import visualize_masked_map_3d
                # visualize_masked_map_3d(self.grid_pos, mask, self.grid_rgb)

                obj_colors = self.grid_rgb[obj_indices]
                n_points = len(obj_colors)

                # 根据点数选择合适的计算方法
                if n_points < 3:
                    # 点数太少，直接使用平均颜色
                    avg_color = np.mean(obj_colors, axis=0).astype(int).tolist()
                    color_dist = {"main_colors": [{"color": avg_color, "proportion": 1.0}]}
                    main_color = tuple(int(c) for c in avg_color)
                else:
                    # ===== 使用颜色量化法识别主要颜色 =====
                    # 代替 KMeans/DBSCAN，使用基于直方图的统计方法
                    try:
                        n_bins = 16  # 将 [0,255] 空间分割为 16x16x16 的网格
                        bin_size = 256 // n_bins

                        # 1. 量化颜色到分箱
                        # 例如: 240 // 16 = 15
                        quantized = (obj_colors // bin_size).astype(int)

                        # 2. 为每个点创建唯一的键以便统计
                        # Key = r * bins^2 + g * bins + b
                        keys = quantized[:, 0] * (n_bins**2) + quantized[:, 1] * n_bins + quantized[:, 2]

                        # 3. 统计每个分箱的出现频率
                        unique_keys, counts = np.unique(keys, return_counts=True)

                        # 4. 获取频率最高的 2 个分箱（与KMeans数量保持一致）
                        top_k = min(2, len(unique_keys))
                        sorted_indices = np.argsort(counts)[::-1][:top_k]

                        main_colors = []

                        for idx in sorted_indices:
                            target_key = unique_keys[idx]

                            # 找到属于当前分箱的所有原始点
                            mask = (keys == target_key)
                            bin_points = obj_colors[mask]

                            # 计算该分箱内原始点颜色的平均值（比单纯取分箱中心更平滑）
                            center_color = np.mean(bin_points, axis=0).astype(int).tolist()
                            proportion = counts[idx] / n_points

                            main_colors.append({"color": center_color, "proportion": proportion})

                        # 防御性代码：如果没有分箱（理论上不会发生），回退到平均
                        if not main_colors:
                            avg_color = np.mean(obj_colors, axis=0).astype(int).tolist()
                            main_colors = [{"color": avg_color, "proportion": 1.0}]

                        color_dist = {"main_colors": main_colors}
                        main_color = tuple(int(c) for c in main_colors[0]["color"])

                    except Exception as e:
                        print(f"颜色量化法失败: {e}")
                        avg_color = np.mean(obj_colors, axis=0).astype(int).tolist()
                        color_dist = {"main_colors": [{"color": avg_color, "proportion": 1.0}]}
                        main_color = tuple(int(c) for c in avg_color)

            else:
                # 没有颜色数据，使用黑色
                color_dist = {"main_colors": [{"color": [0, 0, 0], "proportion": 1.0}]}
                main_color = (0, 0, 0)

            color_distributions.append(color_dist)

            # 使用主色填充物体区域（裁剪区域内坐标）
            bgr_color = (main_color[2], main_color[1], main_color[0])
            cv2.drawContours(color_mask, [contours[i].astype(np.int32)], -1, bgr_color, thickness=cv2.FILLED)

            # 转换坐标到全局地图（与原始代码保持一致）
            centers[i][0] += self.rmin
            centers[i][1] += self.cmin
            bbox_list[i][0] += self.rmin
            bbox_list[i][1] += self.rmin
            bbox_list[i][2] += self.cmin
            bbox_list[i][3] += self.cmin
            for j in range(len(contours[i])):
                contours[i][j, 0] += self.rmin
                contours[i][j, 1] += self.cmin

        # 可视化彩色mask（裁剪区域内）
        if vis:
            # 创建用于可视化的轮廓（转换回(row,col)格式以匹配mask_2d坐标系）
            visualization_contours = []
            for contour in contours:
                # 将轮廓从OpenCV格式(col,row)转回图像数组格式(row,col)
                vis_contour = contour[:, [1, 0]].copy()  # 交换回
                visualization_contours.append(vis_contour.astype(np.int32))

            # 添加轮廓边界（白色）
            contour_mask = np.zeros_like(color_mask)
            for contour in visualization_contours:
                cv2.drawContours(contour_mask, [contour], -1, (255, 255, 255), 1)

            # 叠加轮廓到彩色mask
            combined_mask = cv2.addWeighted(color_mask, 0.8, contour_mask, 0.2, 0)
            combined_mask_display = combined_mask.transpose(1, 0, 2)
            # 显示彩色mask
            logging.debug(f"color_mask_name: {name}")
            cv2.imshow(f"color_mask_{name}", combined_mask_display)
            cv2.waitKey()

        return contours, centers, bbox_list, color_distributions

    def get_pos_color_and_height(self, name: str, floor: int, vis: bool = False) -> Tuple[List[List[int]], List[List[float]], List[np.ndarray], List[Dict]]:
        """
        Get the contours, centers, bbox list and color distributions of a certain category
        on a full map
        """
        assert self.categories
        # 获取目标类别的3D点云掩码
        pc_mask = self.index_map(name, with_init_cat=True)
        pc_mask_indices = np.where(pc_mask)[0]
        all_contours = []
        all_centers = []
        all_bbox_list = []
        all_color_distributions = []
        all_avg_heights = []

        min_3d_heights = self.min_3d_heights
        max_3d_heights = self.max_3d_heights
        if floor is not None:
            min_3d_heights = [min_3d_heights[floor]]
            max_3d_heights = [max_3d_heights[floor]]

        for floor_i in range(len(min_3d_heights)):
            h_min = min_3d_heights[floor_i]
            h_max = max_3d_heights[floor_i]

            # Compute 2D mask for this floor
            mask_2d = pool_filter_by_height_3d_label_to_2d(pc_mask, self.grid_pos, self.gs, self.cs, h_min, h_max)
            mask_2d = mask_2d[self.rmin : self.rmax + 1, self.cmin : self.cmax + 1]

            if vis:
                cv2.imshow(f"mask_{name}_floor_{floor_i}", (mask_2d.astype(np.float32) * 255).astype(np.uint8))
                cv2.waitKey()

            # 创建彩色mask图像（裁剪区域大小）
            color_mask = np.zeros((mask_2d.shape[1], mask_2d.shape[0], 3), dtype=np.uint8)

            foreground = binary_closing(mask_2d, iterations=3)
            foreground = gaussian_filter(foreground.astype(float), sigma=0.8, truncate=3)
            foreground = foreground > 0.5
            foreground = binary_dilation(foreground)
            contours, centers, bbox_list, _ = get_segment_islands_pos(foreground, 1)

            contours_reverse = [None] * len(contours)
            for i in range(len(contours)):
                contours_reverse[i] = contours[i][:, [1, 0]]  # 从 (row,col) 转为 (col,row)

            # 存储每个物体的颜色分布信息
            color_distributions = []
            avg_heights = []

            # 为每个轮廓创建点云索引列表
            contour_indices = [[] for _ in range(len(contours))]

            # 遍历所有属于目标类别的体素，过滤本层高度
            if len(pc_mask_indices) > 0:
                voxel_heights = self.grid_pos[pc_mask_indices, 2] * self.cs
                height_mask = (h_min <= voxel_heights) & (voxel_heights < h_max)
                filtered_indices = pc_mask_indices[height_mask]

                if len(filtered_indices) > 0:
                    # Convert to cropped coordinates
                    local_r = self.grid_pos[filtered_indices, 0] - self.rmin
                    local_c = self.grid_pos[filtered_indices, 1] - self.cmin
                    # Filter points within foreground bounds
                    bounds_mask = (local_r >= 0) & (local_r < foreground.shape[0]) & (local_c >= 0) & (local_c < foreground.shape[1])
                    valid_indices = filtered_indices[bounds_mask]
                    valid_local_r = local_r[bounds_mask]
                    valid_local_c = local_c[bounds_mask]

                    # Assign points to contours
                    for point_idx in range(len(valid_indices)):
                        point = (float(valid_local_c[point_idx]), float(valid_local_r[point_idx]))  # OpenCV format (x,y)
                        for contour_idx, contour in enumerate(contours_reverse):
                            if cv2.pointPolygonTest(contour, point, False) >= 0:
                                contour_indices[contour_idx].append(valid_indices[point_idx])
                                break

            # 处理每个轮廓的点云
            for i, indices in enumerate(contour_indices):
                # 获取当前轮廓的点云索引
                obj_indices = indices

                # 计算颜色分布
                color_dist = {}
                obj_height_avg = 0.0
                if obj_indices:
                    obj_colors = self.grid_rgb[obj_indices]
                    obj_height_avg = int(np.mean(self.grid_pos[obj_indices, 2]).round())
                    main_colors = get_main_colors(obj_colors, max_clusters=4)
                    color_dist = {"main_colors": main_colors}
                    main_color = tuple(int(c) for c in main_colors[0]["color"])
                else:
                    # 没有颜色数据，使用黑色
                    color_dist = {"main_colors": [{"color": [0, 0, 0], "proportion": 1.0}]}
                    main_color = (0, 0, 0)

                color_distributions.append(color_dist)
                avg_heights.append(obj_height_avg)

                # 使用主色填充物体区域（裁剪区域内坐标）
                bgr_color = (main_color[2], main_color[1], main_color[0])
                cv2.drawContours(color_mask, [contours[i].astype(np.int32)], -1, bgr_color, thickness=cv2.FILLED)

                # 转换坐标到全局地图（与原始代码保持一致）
                centers[i][0] += self.rmin
                centers[i][1] += self.cmin
                bbox_list[i][0] += self.rmin
                bbox_list[i][1] += self.rmin
                bbox_list[i][2] += self.cmin
                bbox_list[i][3] += self.cmin
                for j in range(len(contours[i])):
                    contours[i][j, 0] += self.rmin
                    contours[i][j, 1] += self.cmin

            # 可视化彩色mask（裁剪区域内）
            if vis:
                # 创建用于可视化的轮廓（转换回(row,col)格式以匹配mask_2d坐标系）
                visualization_contours = []
                for contour in contours:
                    # 将轮廓从OpenCV格式(col,row)转回图像数组格式(row,col)
                    vis_contour = contour[:, [1, 0]].copy()  # 交换回(row,col)
                    visualization_contours.append(vis_contour.astype(np.int32))

                # 添加轮廓边界（白色）
                contour_mask = np.zeros_like(color_mask)
                for contour in visualization_contours:
                    cv2.drawContours(contour_mask, [contour], -1, (255, 255, 255), 1)

                # 叠加轮廓到彩色mask
                combined_mask = cv2.addWeighted(color_mask, 0.8, contour_mask, 0.2, 0)
                combined_mask_display = combined_mask.transpose(1, 0, 2)
                # 显示彩色mask
                logging.debug(f"color_mask_name: {name}_floor_{floor_i}")
                cv2.imshow(f"color_mask_{name}_floor_{floor_i}", combined_mask_display)
                cv2.waitKey()

            all_contours.extend(contours)
            all_centers.extend(centers)
            all_bbox_list.extend(bbox_list)
            all_color_distributions.extend(color_distributions)
            all_avg_heights.extend(avg_heights)

        return all_contours, all_centers, all_bbox_list, all_color_distributions, all_avg_heights
