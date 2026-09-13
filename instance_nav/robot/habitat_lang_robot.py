# Adapted from VLMaps (https://github.com/vlmaps/vlmaps).
# Copyright (c) 2023 Tom-Huang. MIT license; see THIRD_PARTY_LICENSES/VLMaps-LICENSE.

import os
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from omegaconf import DictConfig

# function to display the topdown map
import habitat_sim
import cv2
import open3d as o3d
import json
from instance_nav.robot.lang_robot import LangRobot
from instance_nav.dataloader.habitat_dataloader import HabitatSceneLoader
from instance_nav.navigator.navigator import Navigator
from instance_nav.controller.discrete_nav_controller import DiscreteNavController

from instance_nav.utils.mapping_utils import (
    grid_id2base_pos_3d,
    grid_id2base_pos_3d_batch,
    base_pos2grid_id_3d,
    cvt_pose_vec2tf,
)
from instance_nav.utils.index_utils import find_similar_category_id
from instance_nav.utils.habitat_utils import make_cfg, tf2agent_state, agent_state2tf, display_sample
from instance_nav.utils.matterport3d_categories import mp3dcat

from typing import List, Tuple, Dict, Any, Union


class HabitatLanguageRobot(LangRobot):
    """
    This class inherits the LangRobot interface for the user.
    It also handles the interface with the Habitat simulator,
    the data, and the control of agent in the environment.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.test_scene_dir = self.config["data_paths"]["habitat_scene_dir"]
        data_dir = Path(self.config["data_paths"]["scene_data_dir"])
        self.scene_data_dirs = [
            data_dir / x for x in sorted(os.listdir(data_dir)) if x != ".DS_Store"
        ]
        # ignore artifact generated in MacOS
        self.map_type = self.config["params"]["map_type"]
        self.camera_height = self.config["params"]["camera_height"]
        self.gs = self.config["params"]["gs"]
        self.cs = self.config["params"]["cs"]
        self.forward_dist = self.config["params"]["forward_dist"]
        self.turn_angle = self.config["params"]["turn_angle"]

        self.sim: habitat_sim.Simulator
        self.sim = None
        self.last_scene_name = ""
        # self.agent_model_dir = self.config["data_paths"]["agent_mesh_dir"]

        self.vis = False

        self.nav = Navigator()
        self.controller = DiscreteNavController(self.config["params"]["controller_config"])
        self.stairs_paths = []
        self.stairs_map_coords = []
        self.stair_floor_mappings = []
        self.unique_heights = []

    def load_stairs(self):
        """
        Load stairs.json for the current scene from scene_data_dir.
        """
        # Reset scene-specific state so that values from a previously loaded
        # scene cannot leak into the next one.
        self.stairs_paths = []
        self.stairs_map_coords = []
        self.stair_floor_mappings = []
        self.unique_heights = []

        if not hasattr(self, 'scene_id') or not hasattr(self, 'scene_data_dirs'):
            raise RuntimeError("Scene not initialized; call setup_scene() before load_stairs().")
        scene_data_dir = self.scene_data_dirs[self.scene_id]
        stairs_file = scene_data_dir / "stairs.json"
        if stairs_file.exists():
            with open(stairs_file, "r") as f:
                self.stairs_paths = json.load(f)
            print(f"Loaded stairs.json from {stairs_file}")
            stairs = self.stairs_paths
            stairs_map_coords = []
            stair_floor_mappings = []
            for path in stairs:
                path_coords = []
                for pose in path:
                    tf = cvt_pose_vec2tf(np.array(pose["position"] + pose["rotation"]))  # Convert to 4x4 transform
                    tf[:3, 3] += np.array([0, self.config.map_config.pose_info.camera_height, 0])
                    row, col, height, _ = self.scene_loader.conver_tf_from_habitat_tf(tf)
                    height -= int(self.map.pcd_min[2] / self.cs)
                    path_coords.append({"row": row, "col": col, "height": height})  # Store y as height
                stairs_map_coords.append(path_coords)

            height_tolerance = 20
            floor_heights = []
            for path in stairs_map_coords:
                start_height = path[0]["height"]
                end_height = path[-1]["height"]
                floor_heights.extend([start_height, end_height])

            # Remove near-duplicate heights within tolerance
            unique_heights = []
            for h in sorted(floor_heights):
                if not unique_heights or all(abs(h - uh) > height_tolerance for uh in unique_heights):
                    unique_heights.append(h)
            unique_heights.sort()  # Sort heights to assign floor indices (lowest = floor 0)

            stair_floor_mappings = []
            for idx, path in enumerate(stairs_map_coords):
                start_height = path[0]["height"]
                end_height = path[-1]["height"]
                # Find the closest unique height for start and end
                start_floor = min(range(len(unique_heights)), key=lambda i: abs(unique_heights[i] - start_height))
                end_floor = min(range(len(unique_heights)), key=lambda i: abs(unique_heights[i] - end_height))
                stair_floor_mappings.append({
                    "path_idx": idx,
                    "start_floor": start_floor,
                    "end_floor": end_floor,
                    "is_ascending": end_height > start_height
                })
            self.stairs_map_coords = stairs_map_coords
            self.stair_floor_mappings = stair_floor_mappings
            self.unique_heights = unique_heights
        else:
            raise FileNotFoundError(
                f"Multi-floor evaluation requires stairs.json, but it was not "
                f"found for scene_id={self.scene_id} at {stairs_file}."
            )

    def generate_3d_obstacle_map(self):
        if not self.unique_heights:
            raise RuntimeError(
                "No floor heights were loaded; verify that stairs.json contains "
                "at least one non-empty stair path."
            )
        self.map.generate_3d_obstacle_map(self.unique_heights,
                                          self.config.map_config.potential_obstacle_names,
                                          self.config.map_config.obstacle_names,
                                          self.config.map_config.min_floor_height,
                                          self.config.map_config.max_floor_height,
                                          vis=self.config.nav.vis)

    def setup_scene_v2(self, scene_id: int):
        """
        Setup the simulator, load scene data and prepare
        the LangRobot interface for navigation
        """
        self.scene_id = scene_id
        scene_data_dir = self.scene_data_dirs[scene_id]
        print(scene_data_dir)
        self.scene_name = scene_data_dir.name.split("_")[0]

        self.setup_map(scene_data_dir)
        # np.array<bool>
        cropped_obst_map = self.map.get_obstacle_cropped()
        if self.config.map_config.potential_obstacle_names and self.config.map_config.obstacle_names:
            # 使用 prebuilt_map.yaml 中的 obstacle_names 生成障碍物地图。
            # Map._dilate_map() 对一个二值地图(obstacles_new_cropped, binary_map）进行膨胀处理，同时可选地应用高斯滤波
            self.map.customize_obstacle_map(
                self.config.map_config.potential_obstacle_names,
                self.config.map_config.obstacle_names,
                self.config.map_config.passable_names,
                vis=self.config.nav.vis,
            )
            cropped_obst_map = self.map.get_customized_obstacle_cropped()

        self.nav.build_visgraph(
            cropped_obst_map,
            self.scene_loader.rmin,
            self.scene_loader.cmin,
            vis=self.config["nav"]["vis"],
            use_internal_contour = self.config["nav"]["use_internal_contour"],
            detect_internal_contours = self.config["nav"]["detect_internal_contours"]
        )
        self.nav.passable_map = self.map.passable_map
        if self.config["nav"]["vis2"]:
            cv2.imshow("Safe Passable Area", (self.map.passable_map * 255).astype(np.uint8))
            cv2.waitKey()

    def setup_scene(self, scene_id: int):
        """
        Setup the simulator, load scene data and prepare
        the LangRobot interface for navigation
        """
        self.scene_id = scene_id
        scene_data_dir = self.scene_data_dirs[scene_id]
        print(scene_data_dir)
        self.scene_name = scene_data_dir.name.split("_")[0]

        self._setup_sim(self.scene_name)

        self.setup_map(scene_data_dir)
        # np.array<bool>
        cropped_obst_map = self.map.get_obstacle_cropped()
        if self.config.map_config.potential_obstacle_names and self.config.map_config.obstacle_names:
            # 使用 prebuilt_map.yaml 中的 obstacle_names 生成障碍物地图。
            # Map._dilate_map() 对一个二值地图(obstacles_new_cropped, binary_map）进行膨胀处理，同时可选地应用高斯滤波
            self.map.customize_obstacle_map(
                self.config.map_config.potential_obstacle_names,
                self.config.map_config.obstacle_names,
                self.config.map_config.passable_names,
                vis=self.config.nav.vis,
            )
            cropped_obst_map = self.map.get_customized_obstacle_cropped()

        self.nav.build_visgraph(
            cropped_obst_map,
            self.scene_loader.rmin,
            self.scene_loader.cmin,
            vis=self.config["nav"]["vis"],
            use_internal_contour = self.config["nav"]["use_internal_contour"],
            detect_internal_contours = self.config["nav"]["detect_internal_contours"]
        )
        self.nav.passable_map = self.map.passable_map
        if self.config["nav"]["vis2"]:
            cv2.imshow("Safe Passable Area", (self.map.passable_map * 255).astype(np.uint8))
            cv2.waitKey()
        # self._setup_localizer(scene_data_dir)

    def setup_map(self, scene_data_dir: str):
        self.load_scene_map(scene_data_dir, self.config["map_config"])

        # TODO: check if needed
        if "3d" in self.config.map_config.map_type:
            self.map.init_categories(mp3dcat.copy())
            self.global_pc = grid_id2base_pos_3d_batch(self.map.grid_pos, self.cs, self.gs)

        self.scene_loader = HabitatSceneLoader(scene_data_dir, self.config.map_config, map=self.map)

    def _setup_sim(self, scene_name: str):
        """
        Setup Habitat simulator, load habitat scene and relevant mesh data
        """
        # 如果sim实例已经存在，则先关闭它
        if self.sim is not None:
            self.sim.close()
        # 构建测试场景的完整路径
        self.test_scene = os.path.join(self.test_scene_dir, scene_name, scene_name + ".glb")
        # 设置sim的配置
        self.sim_setting = {
            "scene": self.test_scene,
            **self.config["params"]["sim_setting"],
        }
        # 根据sim_setting生成配置对象
        cfg = make_cfg(self.sim_setting)
        cfg.sim_cfg.frustum_culling = True
        cfg.sim_cfg.enable_gfx_replay_save = True
        # 创建sim实例
        # 如果sim实例不存在，或者当前场景名称与上次不同，则创建新的sim实例
        if self.sim is None or scene_name != self.last_scene_name:
            self.sim = habitat_sim.Simulator(cfg)
            # 初始化agent
            agent = self.sim.initialize_agent(self.sim_setting["default_agent"])
        else:
            # 如果sim实例已存在且场景名称未变，则重新配置sim
            self.sim.reconfigure(cfg)
        # 更新上次场景名称
        self.last_scene_name = scene_name
        # TODO: add in document to enable this features
        # load agent mesh for visualization
        # if self.agent_model_dir:
        #     rigid_obj_mgr = self.sim.get_rigid_object_manager()
        #     obj_attr_mgr = self.sim.get_object_template_manager()
        #     locobot_template_id = obj_attr_mgr.load_configs(self.agent_model_dir)[0]
        #     locobot = rigid_obj_mgr.add_object_by_template_id(locobot_template_id, self.sim.agents[0].scene_node)

    def set_agent_state(self, tf: np.ndarray):
        agent_state = tf2agent_state(tf)
        self.sim.get_agent(0).set_state(agent_state)
        self._set_nav_curr_pose()

    def get_agent_tf(self) -> np.ndarray:
        agent_state = self.sim.get_agent(0).get_state()
        return agent_state2tf(agent_state)

    def load_gt_region_map(self, region_gt: List[Dict[str, np.ndarray]]):
        obst_cropped = self.scene_loader.get_obstacles_cropped()
        self.region_categories = sorted(list(region_gt.keys()))
        self.gt_region_map = np.zeros(
            (len(self.region_categories), obst_cropped.shape[0], obst_cropped.shape[1]), dtype=np.uint8
        )

        for cat_i, cat in enumerate(self.region_categories):
            for box_i, box in enumerate(region_gt[cat]):
                center = np.array(box["region_center"])
                size = np.array(box["region_size"])
                top_left = center - size / 2
                bottom_right = center + size / 2
                top_left_2d = self.scene_loader.convert_habitat_pos_list_to_cropped_map_pos_list([top_left])[0]
                bottom_right_2d = self.scene_loader.convert_habitat_pos_list_to_cropped_map_pos_list(
                    [bottom_right]
                )[0]

                self.gt_region_map[cat_i] = cv2.rectangle(
                    self.gt_region_map[cat_i],
                    (int(top_left_2d[1]), int(top_left_2d[0])),
                    (int(bottom_right_2d[1]), int(bottom_right_2d[0])),
                    1,
                    -1,
                )

    def get_distribution_map(
        self, name: str, scores: np.ndarray, pos_list_cropped: List[List[float]], decay_rate: float = 0.1
    ):
        if scores.shape[0] > 1:
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        obst_map_cropped = self.map.get_customized_obstacle_cropped()
        dist_map = np.zeros_like(obst_map_cropped, dtype=np.float32)
        for pos_i, pos in enumerate(pos_list_cropped):
            # dist_map[int(pos[0]), int(pos[1])] = scores[pos_i]
            tmp_dist_map = np.zeros_like(dist_map, dtype=np.float32)
            pos = np.round(pos[0]), np.round(pos[1])
            tmp_dist_map[int(pos[0]), int(pos[1])] = scores[pos_i]

            con = scores[pos_i]
            dists = distance_transform_edt(tmp_dist_map == 0)
            reduct = con * dists * decay_rate
            tmp = np.ones_like(tmp_dist_map) * con - reduct
            tmp_dist_map = np.clip(tmp, 0, 1)
            dist_map += tmp_dist_map
        dist_map = (dist_map - np.min(dist_map)) / (np.max(dist_map) - np.min(dist_map))
        if self.config["nav"]["vis"]:
            self._vis_dist_map(dist_map, name=name)
        return dist_map

    def get_distribution_map_3d(
        self, name: str, scores: np.ndarray, pos_list_3d: List[List[float]], decay_rate: float = 0.1
    ):
        """
        pos_list_3d: list of 3d positions in 3d map coordinate
        """
        if scores.shape[0] > 1:
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        sim_mat = np.zeros((self.global_pc.shape[0], len(scores)))
        for pro_i, (con, pos) in enumerate(zip(scores, pos_list_3d)):
            print("confidence: ", con)

            dists = np.linalg.norm(self.global_pc[:, [0, 2]] - pos[[0, 2]], axis=1) / self.scene_loader.cs
            sim = np.clip(con - decay_rate * dists, 0, 1)
            sim_mat[:, pro_i] = sim

        sim = np.max(sim_mat, axis=1)
        if self.config["nav"]["vis"]:
            self._vis_dist_map_3d(sim, name=name)

        return sim.flatten()

    def get_vl_distribution_map(self, name: str, decay_rate: float = 0.1) -> np.ndarray:
        predict_mask = self.map.get_predict_mask(name)
        predict_mask = predict_mask.astype(np.float32)
        predict_mask = (gaussian_filter(predict_mask, sigma=1) > 0.5).astype(np.float32)
        dists = distance_transform_edt(predict_mask == 0)
        tmp = np.ones_like(dists) - (dists * decay_rate)
        dist_map = np.where(tmp < 0, np.zeros_like(tmp), tmp)
        dist_map = (dist_map - np.min(dist_map)) / (np.max(dist_map) - np.min(dist_map))
        if self.config["nav"]["vis"]:
            self._vis_dist_map(predict_mask, name=name + "_predict_mask")
            self._vis_dist_map(dist_map, name=name + f"_{decay_rate}")
        return dist_map

    def get_vl_distribution_map_3d(self, name: str, decay_rate: float = 0.1) -> np.ndarray:
        predict = np.argmax(self.map.scores_mat, axis=1)
        i = find_similar_category_id(name, self.map.categories)
        sim = predict == i

        target_pc = self.global_pc[sim, :]
        other_ids = np.where(sim == 0)[0]
        other_pc = self.global_pc[other_ids, :]
        target_sim = np.ones((target_pc.shape[0], 1))
        other_sim = np.zeros((other_pc.shape[0], 1))
        for other_p_i, p in enumerate(other_pc):
            dist = np.linalg.norm(target_pc - p, axis=1) / self.cs
            min_dist_i = np.argmin(dist)
            min_dist = dist[min_dist_i]
            other_sim[other_p_i] = np.clip(1 - min_dist * decay_rate, 0, 1)

        new_pc_global = self.global_pc.copy()
        new_sim = np.ones((new_pc_global.shape[0], 1), dtype=np.float32)
        for s_i, s in enumerate(other_sim):
            new_sim[other_ids[s_i]] = s

        if self.config["nav"]["vis"]:
            self._vis_dist_map_3d(new_sim, name=name)
        return new_sim.flatten()

    def get_region_distribution_map(self, name: str, decay_rate: float = 0.1) -> np.ndarray:
        if self.area_map_type == "clip_sparse":
            return self.get_clip_sparse_region_distribution_map(name, decay_rate)
        elif self.area_map_type == "concept_fusion":
            return self.get_concept_fusion_region_distribution_map(name, decay_rate)
        elif self.area_map_type == "lseg":
            return self.get_lseg_region_map(name, decay_rate)
        elif self.area_map_type == "gt":
            return self.get_gt_region_map(name, decay_rate)

    def get_gt_region_map(self, name: str, decay_rate: float = 0.1) -> np.ndarray:
        assert self.area_map_type == "gt"
        id = find_similar_category_id(name, self.region_categories)
        predict_mask = self.gt_region_map[id]

        obst_map_cropped = self.map.get_customized_obstacle_cropped()
        dist_map = np.zeros_like(obst_map_cropped, dtype=np.float32)
        dists = distance_transform_edt(predict_mask == 0)
        tmp = np.ones_like(dists) - (dists * decay_rate)
        dist_map = np.clip(tmp, 0, 1)
        dist_map = (dist_map - np.min(dist_map)) / (np.max(dist_map) - np.min(dist_map))
        if self.config["nav"]["vis"]:
            self._vis_dist_map(dist_map, name=name)
        return dist_map

    def get_lseg_region_map(self, name: str, decay_rate: float = 0.1) -> np.ndarray:
        assert self.map_type == "lseg"
        assert self.area_map_type == "lseg"
        obst_map_cropped = self.map.get_customized_obstacle_cropped()
        dist_map = np.zeros_like(obst_map_cropped, dtype=np.float32)
        predict_mask = self.map.get_region_predict_mask(name)
        dists = distance_transform_edt(predict_mask == 0)
        tmp = np.ones_like(dists) - (dists * decay_rate)
        dist_map = np.clip(tmp, 0, 1)
        dist_map = (dist_map - np.min(dist_map)) / (np.max(dist_map) - np.min(dist_map))
        if self.config["nav"]["vis"]:
            self._vis_dist_map(dist_map, name=name)
        return dist_map

    def get_concept_fusion_region_distribution_map(self, name: str, decay_rate: float = 0.1) -> np.ndarray:
        assert self.area_map is not None, "Area map is not initialized."
        obst_map_cropped = self.map.get_customized_obstacle_cropped()
        dist_map = np.zeros_like(obst_map_cropped, dtype=np.float32)
        predict_mask = self.area_map.get_predict_mask(name)
        # print("predict_mask: ", predict_mask.shape)
        # mask_vis = cv2.cvtColor((predict_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # cv2.imshow("mask_vis", mask_vis)
        # cv2.waitKey()
        dists = distance_transform_edt(predict_mask == 0)
        tmp = np.ones_like(dists) - (dists * decay_rate)
        dist_map = np.where(tmp < 0, np.zeros_like(tmp), tmp)
        dist_map = (dist_map - np.min(dist_map)) / (np.max(dist_map) - np.min(dist_map))
        if self.config["nav"]["vis"]:
            self._vis_dist_map(dist_map, name=name)
        return dist_map

    def get_clip_sparse_region_distribution_map(self, name: str, decay_rate: float = 0.1) -> np.ndarray:
        assert self.area_map is not None, "Area map is not initialized."
        obst_map_cropped = self.map.get_customized_obstacle_cropped()
        dist_map = np.zeros_like(obst_map_cropped, dtype=np.float32)
        scores = self.area_map.get_scores(name)
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        robot_pose_list = self.area_map.get_robot_pose_list()
        ids = np.argsort(-scores.flatten())
        # max_id = np.argmax(scores)
        # obst_map = self.scene_loader.get_obstacles_cropped_no_floor()
        # obst_map = np.tile(obst_map[:, :, None] * 255, [1, 1, 3]).astype(np.uint8)

        for i, tf_hab in enumerate(robot_pose_list):
            tmp_dist_map = np.zeros_like(dist_map, dtype=np.float32)
            row, col, deg = self.scene_loader.convert_habitat_tf_to_cropped_map_pose(tf_hab)
            if row < 0 or row >= dist_map.shape[0] or col < 0 or col >= dist_map.shape[1]:
                continue
            # print(i, tf_hab[:3, 3].flatten(), row, col)
            # obst_map[int(row), int(col)] = (0, 255, 0)
            # cv2.circle(obst_map, (int(col), int(row)), 3, (0, 255, 0), -1)
            # cv2.imshow("obst_map", obst_map)
            # cv2.waitKey(1)
            s = scores[i]
            tmp_dist_map[row, col] = s
            dists = distance_transform_edt(tmp_dist_map == 0)
            tmp = np.ones_like(dists) * s - (dists * decay_rate)
            tmp_dist_map = np.clip(tmp, 0, 1)
            dist_map = np.where(dist_map > tmp_dist_map, dist_map, tmp_dist_map)

        dist_map = (dist_map - np.min(dist_map)) / (np.max(dist_map) - np.min(dist_map))
        if self.config["nav"]["vis"]:
            self._vis_dist_map(dist_map, name=name)
        return dist_map

    def get_map(self, obj: str = None, sound: str = None):
        """
        Return the distribution map of a certain object or sound category with decay 0.01
        """
        assert obj is not None or sound is not None, "Object and sound names are both None."
        if obj is not None:
            return self.get_vl_distribution_map(obj, decay_rate=0.01)
        elif sound is not None:
            return self.get_sound_distribution_map(sound, decay_rate=0.01)

    def get_major_map(self, obj: str = None, sound: str = None):
        """
        Return the distribution map of a certain object or sound category with decay 0.1
        """
        assert obj is not None or sound is not None, "Object and sound names are both None."
        if obj is not None:
            return self.get_vl_distribution_map(obj, decay_rate=0.1)
        elif sound is not None:
            return self.get_sound_distribution_map(sound, decay_rate=0.1)

    def get_map_3d(self, obj: str = None, sound: str = None, img: np.ndarray = None, intr_mat: np.ndarray = None):
        """
        Return the distribution map of a certain object or sound category with decay 0.01
        """
        assert obj is not None or sound is not None or img is not None, "Object, sound names, and image are all None."
        if obj is not None:
            return self.get_vl_distribution_map_3d(obj, decay_rate=0.03)
        elif sound is not None:
            return self.get_sound_distribution_map_3d(sound, decay_rate=0.05)
        elif img is not None:
            return self.get_image_distribution_map_3d(img, query_intr_mat=intr_mat, decay_rate=0.05)

    def get_major_map_3d(self, obj: str = None, sound: str = None, img: np.ndarray = None, intr_mat: np.ndarray = None):
        """
        Return the distribution map of a certain object or sound category with decay 0.1
        """
        assert obj is not None or sound is not None or img is not None, "Object, sound names, and image are all None."
        if obj is not None:
            return self.get_vl_distribution_map_3d(obj, decay_rate=0.1)
        elif sound is not None:
            return self.get_sound_distribution_map_3d(sound, decay_rate=0.05)
        elif img is not None and intr_mat is not None:
            return self.get_image_distribution_map_3d(img, query_intr_mat=intr_mat, decay_rate=0.01)

    def _vis_dist_map(self, dist_map: np.ndarray, name: str = ""):
        obst_map = self.scene_loader.get_obstacles_cropped_no_floor()
        obst_map = np.tile(obst_map[:, :, None] * 255, [1, 1, 3]).astype(np.uint8)
        target_heatmap = show_cam_on_image(obst_map.astype(float) / 255.0, dist_map)
        cv2.imshow(f"heatmap_{name}", target_heatmap)

    def _vis_dist_map_3d(self, heatmap: np.ndarray, transparency: float = 0.3, name: str = ""):
        print(f"heatmap of {name}")
        sim_new = (heatmap * 255).astype(np.uint8)
        rgb_pc = cv2.applyColorMap(sim_new, cv2.COLORMAP_JET)
        rgb_pc = rgb_pc.reshape(-1, 3)[:, ::-1].astype(np.float32) / 255.0
        rgb_pc = rgb_pc * transparency + self.map.grid_rgb / 255.0 * (1 - transparency)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.global_pc)
        pcd.colors = o3d.utility.Vector3dVector(rgb_pc)
        o3d.visualization.draw_geometries([pcd])

    def get_max_pos(self, map: np.ndarray) -> Tuple[float, float]:
        id = np.argmax(map)
        row, col = np.unravel_index(id, map.shape)

        if self.config["nav"]["vis"]:
            self._vis_dist_map(map, name="fuse")
        return row + self.scene_loader.rmin, col + self.scene_loader.cmin

    def get_max_pos_3d(self, heat: np.ndarray) -> Tuple[float, float, float]:
        id = np.argmax(heat)
        grid_map_pos_3d = self.map.grid_pos[id]
        return grid_map_pos_3d

    def move_to(self, pos: Tuple[float, float]) -> List[str]:
        """Move the robot to the position on the full map
            based on accurate localization in the environment

        Args:
            pos (Tuple[float, float]): (row, col) on full map

        Returns:
            List[str]: list of actions
        """
        actual_actions_list = []
        success = False
        # while not success:
        self._set_nav_curr_pose()
        curr_pose_on_full_map = self.get_agent_pose_on_map()  # (row, col, angle_deg) on full map
        # print(f"self.config[\"nav\"][\"vis\"] : {self.config['nav']['vis']}")
        paths = self.nav.plan_to(
            curr_pose_on_full_map[:2], pos, vis=self.config["nav"]["plann_vis"]
        )  # take (row, col) in full map
        # print(paths)
        actions_list, poses_list = self.controller.convert_paths_to_actions(curr_pose_on_full_map, paths[1:])
        success, real_actions_list = self.execute_actions(actions_list, poses_list, vis=self.config["nav"]["plann_vis"])
        actual_actions_list.extend(real_actions_list)

        actual_actions_list.append("stop")

        if not hasattr(self, "recorded_actions_list"):
            self.recorded_actions_list = []
        self.recorded_actions_list.extend(actual_actions_list)

        return actual_actions_list

    def move_to_v2(self, pos: Tuple[float, float]) -> List[str]:
        """Move the robot to the position on the full map
            based on accurate localization in the environment

        Args:
            pos (Tuple[float, float]): (row, col) on full map

        Returns:
            List[str]: list of actions
        """
        actual_actions_list = []
        success = False
        # while not success:
        self._set_nav_curr_pose()
        curr_pose_on_full_map = self.get_agent_pose_on_map()  # (row, col, angle_deg) on full map
        # print(f"self.config[\"nav\"][\"vis\"] : {self.config['nav']['vis']}")
        paths = self.nav.plan_to_v2(
            self.map.passable_map, curr_pose_on_full_map[:2], pos, vis=self.config["nav"]["plann_vis"]
        )  # take (row, col) in full map
        # print(paths)
        actions_list, poses_list = self.controller.convert_paths_to_actions(curr_pose_on_full_map, paths[1:])
        success, real_actions_list = self.execute_actions(actions_list, poses_list, vis=self.config["nav"]["plann_vis"])
        actual_actions_list.extend(real_actions_list)

        actual_actions_list.append("stop")

        if not hasattr(self, "recorded_actions_list"):
            self.recorded_actions_list = []
        self.recorded_actions_list.extend(actual_actions_list)

        return actual_actions_list

    def move_to_3d(self, pos: Tuple[float, float], avg_height: float) -> List[str]:
        """
        Move the robot to the position on the full map, handling cross-floor navigation for up to three floors.
        Uses stairs.json to navigate between floors, with straight-line paths on the source floor and plan_to_v2 for
        intermediate and target floors.

        Args:
            pos (Tuple[float, float]): (row, col) on full map
            avg_height (float): Average height of the target object in Habitat coordinates

        Returns:
            List[str]: List of actions for the entire multi-floor navigation
        """
        actual_actions_list = []
        self._set_nav_curr_pose()
        curr_pose_on_full_map = self.get_agent_pose_on_map()  # (row, col, angle_deg)
        # current_3d_pos = self.scene_loader.full_map_3d_pose  # (x, y, z) in Habitat coordinates
        current_tran_3d_pose = self.scene_loader.tran_map_3d_pose

        # Determine source and target floors
        src_floor = min(range(len(self.unique_heights)), key=lambda i: abs(self.unique_heights[i] - current_tran_3d_pose[2]))
        if avg_height == None:
            tgt_floor = src_floor
        else:
            tgt_floor = min(range(len(self.unique_heights)), key=lambda i: abs(self.unique_heights[i] - avg_height))

        if src_floor == tgt_floor:
            path = self.nav.plan_to_v2(self.map.passable_3d_maps[src_floor], curr_pose_on_full_map[:2], pos, vis=self.config["nav"]["plann_vis"], floor=src_floor)
            actions_list, poses_list = self.controller.convert_paths_to_actions(curr_pose_on_full_map, path)
            success, real_actions_list = self.execute_actions(actions_list, poses_list, vis=self.config["nav"]["plann_vis"])
            actual_actions_list.extend(real_actions_list)
            actual_actions_list.append("stop")
            if not hasattr(self, "recorded_actions_list"):
                self.recorded_actions_list = []
            self.recorded_actions_list.extend(actual_actions_list)
            return actual_actions_list

        # Determine navigation direction (up or down)
        ascending = tgt_floor > src_floor
        floor_diff = abs(tgt_floor - src_floor)
        if floor_diff > 2:
            print("Navigation across more than two floors not supported")
            return self.move_to_v2(pos)  # Fallback to single-floor navigation

        # Find nearest stair start/end points
        def find_nearest_stair_points(floor_from, floor_to, src_pos, tgt_pos):
            min_dist = float('inf')
            best_start = None
            best_end = None
            best_path_idx = None
            for mapping in self.stair_floor_mappings:
                # Check if the stair connects the desired floors (in either direction)
                connects_floors = (
                    (mapping["start_floor"] == floor_from and mapping["end_floor"] == floor_to) or
                    (mapping["start_floor"] == floor_to and mapping["end_floor"] == floor_from)
                )
                if connects_floors:
                    path = self.stairs_map_coords[mapping["path_idx"]]
                    # Determine start and end based on navigation direction
                    if (mapping["start_floor"] == floor_from and mapping["end_floor"] == floor_to):
                        path_start = path[0]
                        path_end = path[-1]
                    else:  # Reverse the path direction
                        path_start = path[-1]
                        path_end = path[0]
                    start_dist = np.hypot(src_pos[0] - path_start["row"], src_pos[1] - path_start["col"])
                    end_dist = np.hypot(tgt_pos[0] - path_end["row"], tgt_pos[1] - path_end["col"])
                    total_dist = start_dist + end_dist
                    if total_dist < min_dist:
                        min_dist = total_dist
                        best_start = path_start
                        best_end = path_end
                        best_path_idx = mapping["path_idx"]
            return best_start, best_end, best_path_idx

        # Plan paths for each segment
        paths = []
        passable_3d_maps = self.map.passable_3d_maps
        if floor_diff == 1:  # Direct navigation from src_floor to tgt_floor
            stair_start, stair_end, path_idx = find_nearest_stair_points(src_floor, tgt_floor, curr_pose_on_full_map[:2], pos)
            if stair_start is None or stair_end is None:
                print("No suitable stair path found, falling back to single-floor navigation")
                return self.move_to_v2(pos)

            # Source floor: straight-line path to stair start
            src_to_stair = self.nav.plan_to_v2(passable_3d_maps[src_floor], curr_pose_on_full_map[:2], (stair_start["row"], stair_start["col"]), vis=self.config["nav"]["plann_vis"], floor=src_floor)
            paths.append(src_to_stair)

            # Stair path (reverse if descending and the path was recorded ascending, or vice versa)
            stair_path = [(p["row"], p["col"]) for p in  self.stairs_map_coords[path_idx]]
            is_path_ascending = self.stair_floor_mappings[path_idx]["is_ascending"]
            if (ascending and not is_path_ascending) or (not ascending and is_path_ascending):
                stair_path = stair_path[::-1]
            paths.append(stair_path)

            # Target floor: plan from stair end to target
            tgt_path = self.nav.plan_to_v2(passable_3d_maps[tgt_floor], (stair_end["row"], stair_end["col"]), pos, vis=self.config["nav"]["plann_vis"], floor=tgt_floor)
            paths.append(tgt_path[1:])  # Skip the start point as it's already in stair_path

        else:  # floor_diff == 2 (e.g., floor 0 to 2 or 2 to 0)
            # First leg: floor 0 to 1 or 2 to 1
            mid_floor = 1
            stair1_start, stair1_end, path1_idx = find_nearest_stair_points(src_floor, mid_floor, curr_pose_on_full_map[:2], (0, 0))  # Midpoint irrelevant
            if stair1_start is None or stair1_end is None:
                print("No suitable stair path found for first leg, falling back to single-floor navigation")
                return self.move_to_v2(pos)

            # Source floor: straight-line path to stair start
            src_to_stair1 = self.nav.plan_to_v2(passable_3d_maps[src_floor], curr_pose_on_full_map[:2], (stair1_start["row"], stair1_start["col"]), vis=self.config["nav"]["plann_vis"], floor=src_floor)
            paths.append(src_to_stair1)

            # First stair path
            stair1_path = [(p["row"], p["col"]) for p in  self.stairs_map_coords[path1_idx]]
            is_path1_ascending = self.stair_floor_mappings[path1_idx]["is_ascending"]
            if (ascending and not is_path1_ascending) or (not ascending and is_path1_ascending):
                stair1_path = stair1_path[::-1]
            paths.append(stair1_path)

            # Middle floor: plan from stair1 end to stair2 start
            stair2_start, stair2_end, path2_idx = find_nearest_stair_points(mid_floor, tgt_floor, (stair1_end["row"], stair1_end["col"]), pos)
            if stair2_start is None or stair2_end is None:
                print("No suitable stair path found for second leg, falling back to single-floor navigation")
                return self.move_to_v2(pos)
            mid_path = self.nav.plan_to_v2(passable_3d_maps[mid_floor], (stair1_end["row"], stair1_end["col"]), (stair2_start["row"], stair2_start["col"]), vis=self.config["nav"]["plann_vis"], floor=mid_floor)
            paths.append(mid_path[1:])

            # Second stair path
            stair2_path = [(p["row"], p["col"]) for p in  self.stairs_map_coords[path2_idx]]
            is_path2_ascending = self.stair_floor_mappings[path2_idx]["is_ascending"]
            if (ascending and not is_path2_ascending) or (not ascending and is_path2_ascending):
                stair2_path = stair2_path[::-1]
            paths.append(stair2_path)

            # Target floor: plan from stair2 end to target
            tgt_path = self.nav.plan_to_v2(passable_3d_maps[tgt_floor], (stair2_end["row"], stair2_end["col"]), pos, vis=self.config["nav"]["plann_vis"], floor=tgt_floor)
            paths.append(tgt_path[1:])

        # Combine paths and generate actions
        combined_paths = []
        for path in paths:
            if path:  # Skip empty paths
                combined_paths.extend(path[1:] if combined_paths else path)  # Include first point only for the first path

        actions_list, poses_list = self.controller.convert_paths_to_actions(curr_pose_on_full_map, combined_paths)
        success, real_actions_list = self.execute_actions(actions_list, poses_list, vis=self.config["nav"]["plann_vis"])
        actual_actions_list.extend(real_actions_list)
        actual_actions_list.append("stop")

        if not hasattr(self, "recorded_actions_list"):
            self.recorded_actions_list = []
        self.recorded_actions_list.extend(actual_actions_list)

        return actual_actions_list

    def turn(self, angle_deg: float):
        """
        Turn right a relative angle in degrees
        """
        if angle_deg < 0:
            actions_list = ["turn_left"] * int(np.abs(angle_deg / self.turn_angle))
        else:
            actions_list = ["turn_right"] * int(angle_deg / self.turn_angle)

        success, real_actions_list = self.execute_actions(actions_list, vis=self.config.nav.vis)

        self.recorded_actions_list.extend(real_actions_list)
        return real_actions_list

    def execute_actions(
        self,
        actions_list: List[str],
        poses_list: List[List[float]] = None,
        vis: bool = False,
    ) -> Tuple[bool, List[str]]:
        """
        Execute actions and check
        """
        if poses_list is not None:
            assert len(actions_list) == len(poses_list)
        if vis:
            map = self.map.get_customized_obstacle_cropped()
            map = (map[:, :, None] * 255).astype(np.uint8)
            map = np.tile(map, [1, 1, 3])
            self.display_goals_on_map(map, 3, (0, 255, 0))
            if hasattr(self, "recorded_robot_pos") and len(self.recorded_robot_pos) > 0:
                map = self.display_full_map_pos_list_on_map(map, self.recorded_robot_pos)
            else:
                self.recorded_robot_pos = []

        real_actions_list = []
        for action_i, action in enumerate(actions_list):
            self._execute_action(action)

            real_actions_list.append(action)
            if vis:
                self.display_obs(waitkey=False)
                self.display_curr_pos_on_map(map)
            if poses_list is None:
                continue
            row, col, angle = self._get_full_map_pose()
            if vis:
                self.recorded_robot_pos.append((row, col))
            # x, z = grid_id2base_pos_3d(self.gs, self.cs, col, row)
            # pred_x, pred_z, pred_angle = poses_list[action_i]
            # success = self._check_if_pose_match_prediction(x, z, pred_x, pred_z)
            # if not success:
            #     return success, real_actions_list
        return True, real_actions_list

    def pass_goal_bboxes(self, goal_bboxes: Dict[str, Any]):
        self.goal_bboxes = goal_bboxes

    def pass_goal_tf(self, goal_tfs: List[np.ndarray]):
        self.goal_tfs = goal_tfs

    def pass_goal_tf_list(self, goal_tfs: List[List[np.ndarray]]):
        self.all_goal_tfs = goal_tfs
        self.goal_id = 0

    def _execute_action(self, action: str):
        self.sim.step(action)

    def _check_if_pose_match_prediction(self, real_x: float, real_z: float, pred_x: float, pred_z: float):
        dist_thres = self.forward_dist
        dx = pred_x - real_x
        dz = pred_z - real_z
        dist = np.sqrt(dx * dx + dz * dz)
        return dist < dist_thres

    def _set_nav_curr_pose(self):
        """
        Set self.curr_pos_on_map and self.curr_ang_deg_on_map
        based on the simulator agent ground truth pose
        """
        agent_state = self.sim.get_agent(0).get_state()
        hab_tf = agent_state2tf(agent_state)
        self.scene_loader.from_habitat_tf(hab_tf)
        row, col, angle_deg = self.scene_loader.to_full_map_pose()
        self.curr_pos_on_map = (row, col)
        self.curr_ang_deg_on_map = angle_deg
        # print("set curr pose: ", row, col, angle_deg)
    def _set_nav_curr_pose_v2(self, pose):
        """
        Set self.curr_pos_on_map and self.curr_ang_deg_on_map
        based on the simulator agent ground truth pose
        """
        self.curr_pos_on_map = (pose[0], pose[1])
        self.curr_ang_deg_on_map = 0
        # print("set curr pose: ", row, col, angle_deg)

    def to_base_pos(self, row_tran, col_tran):
        """
        从网格坐标 (row_tran, col_tran) 反向转换回 base 坐标系下的 x, y
        """
        x = (self.gs / 2 - row_tran) * self.cs
        y = (self.gs / 2 - col_tran) * self.cs

        return x, y

    def _get_full_map_pose(self) -> Tuple[float, float, float]:
        agent_state = self.sim.get_agent(0).get_state()
        hab_tf = agent_state2tf(agent_state)
        self.scene_loader.from_habitat_tf(hab_tf)
        row, col, angle_deg = self.scene_loader.to_full_map_pose()
        return row, col, angle_deg

    def display_obs(self, waitkey: bool = False):
        obs = self.sim.get_sensor_observations(0)
        display_sample(self.sim_setting, obs["color_sensor"], waitkey=waitkey)

    def display_curr_pos_on_map(self, map: np.ndarray):
        row, col, angle = self._get_full_map_pose()
        self.scene_loader.from_full_map_pose(row, col, angle)
        row, col, angle = self.scene_loader.to_cropped_map_pose()
        map = cv2.circle(map, (int(col), int(row)), 3, (255, 0, 0), -1)
        cv2.imshow("real path", map)
        cv2.waitKey(1)

    def display_full_map_pos_list_on_map(self, map: np.ndarray, pos_list: List[List[float]]) -> np.ndarray:
        for pos_i, pos in enumerate(pos_list):
            self.scene_loader.from_full_map_pose(pos[0], pos[1], 0)
            row, col, _ = self.scene_loader.to_cropped_map_pose()
            map = cv2.circle(map, (int(col), int(row)), 3, (255, 0, 0), -1)
        return map

    def display_goals_on_map(
        self,
        map: np.ndarray,
        radius_pix: int = 3,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        if not hasattr(self, "goal_tfs") and not hasattr(self, "goal_bboxes"):
            return map
        if hasattr(self, "goal_tfs"):
            map = self.display_goal_tfs_on_map(map, radius_pix, color)
        elif hasattr(self, "goal_bboxes"):
            map = self.display_goal_bboxes_on_map(map, color)
        else:
            print("no goal tfs or bboxes passed to the robot.")
        return map

    def display_goal_bboxes_on_map(
        self,
        map: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        centers = self.goal_bboxes["centers"]
        sizes = self.goal_bboxes["sizes"]
        cs = self.scene_loader.cs
        centers_cropped = self.scene_loader.convert_habitat_pos_list_to_cropped_map_pos_list(centers)
        for center_i, center in enumerate(centers_cropped):
            size = [float(x) / cs for x in sizes[center_i]]  # in habitat robot coord
            size = size[[2, 0]]
            min_corner, max_corner = get_bbox(center, size)
            rmin, rmax = int(min_corner[0]), int(max_corner[0])
            cmin, cmax = int(min_corner[1]), int(max_corner[1])
            cv2.rectangle(map, (cmin, rmin), (cmax, rmax), color, 2)

        return map

    def display_goal_tfs_on_map(
        self,
        map: np.ndarray,
        radius_pix: int = 3,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        if self.goal_tfs is None:
            if self.all_goal_tfs is None:
                print("no goal tfs passed to the robot")
                return map
            self.goal_tfs = self.all_goal_tfs[self.goal_id]
            self.goal_id += 1

        for tf_i, tf in enumerate(self.goal_tfs):
            self.scene_loader.from_habitat_tf(tf)
            (row, col, angle_deg) = self.scene_loader.to_cropped_map_pose()
            cv2.circle(map, (col, row), radius_pix, color, 2)
        self.goal_tfs = None
        return map
