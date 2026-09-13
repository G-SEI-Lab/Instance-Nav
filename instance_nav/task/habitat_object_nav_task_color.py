# Adapted from VLMaps (https://github.com/vlmaps/vlmaps).
# Copyright (c) 2023 Tom-Huang. MIT license; see THIRD_PARTY_LICENSES/VLMaps-LICENSE.

from pathlib import Path
import json
from typing import Dict, List, Tuple, Union
import logging
from omegaconf import DictConfig
import numpy as np
import habitat_sim
import cv2

from instance_nav.task.habitat_task import HabitatTask
from instance_nav.utils.habitat_utils import agent_state2tf, get_position_floor_objects
from instance_nav.utils.navigation_utils import get_dist_to_bbox_2d
from instance_nav.utils.habitat_utils import display_sample
from instance_nav.robot.habitat_lang_robot import HabitatLanguageRobot
from instance_nav.map.map import Map
from instance_nav.utils.mapping_utils import base_pos2grid_id_3d_3

logging.basicConfig(
    level=logging.INFO,
    format='[%(filename)s:%(lineno)d] %(message)s'
)

class HabitatObjectNavigationTaskColor(HabitatTask):
    def load_task(self):
        assert hasattr(self, "scene_loader"), "Please call setup_scene() first"

        task_path = Path(self.scene_loader.data_dir) / "color_object_nav_dataset.json"
        with open(task_path, "r") as f:
            self.task_dict = json.load(f)

    def setup_task(self, task_id: int):
        json_task_id = self.task_dict[task_id]["task_id"]
        assert json_task_id == task_id, "Task ID mismatch"
        self.task_id = task_id
        self.init_hab_tf = np.array(self.task_dict[task_id]["tf_habitat"], dtype=np.float32).reshape((4, 4))
        self.map_grid_size = self.task_dict[task_id]["map_grid_size"]
        self.map_cell_size = self.task_dict[task_id]["map_cell_size"]
        self.scene = self.task_dict[task_id]["scene"]
        self.instruction = self.task_dict[task_id]["instruction"]
        self.goal_classes = [x["name"] for x in self.task_dict[task_id]["objects_info"]]
        self.objects_info = self.task_dict[task_id]["objects_info"]
        # metric
        self.n_subgoals_in_task = len(self.goal_classes)
        self.curr_subgoal_id = 0
        self.finished_subgoals = []
        self.distance_to_subgoals = []
        self.success = False
        self.actions = []

    def setup_task_v2(self, task_id: int):
        json_task_id = self.task_dict[task_id]["task_id"]
        assert json_task_id == task_id, "Task ID mismatch"
        self.task_id = task_id
        self.init_hab_tf = np.array(self.task_dict[task_id]["tf_habitat"], dtype=np.float32).reshape((4, 4))
        self.map_grid_size = self.task_dict[task_id]["map_grid_size"]
        self.map_cell_size = self.task_dict[task_id]["map_cell_size"]
        self.scene = self.task_dict[task_id]["scene"]
        self.goal_classes = [x["name"] for x in self.task_dict[task_id]["objects_info"]]
        self.objects_info = self.task_dict[task_id]["objects_info"]
        # metric
        self.n_subgoals_in_task = len(self.goal_classes)
        self.curr_subgoal_id = 0
        self.finished_subgoals = []
        self.distance_to_subgoals = []
        self.success = False
        self.actions = []

    def get_all_objects(self, sim: habitat_sim.Simulator):
        scene = sim.semantic_scene
        agent = sim.get_agent(0)
        self.same_floor_objects_list = get_position_floor_objects(
            scene, self.init_hab_tf[:3, 3], self.scene_loader.camera_height + 0.5
        )

    def get_class_objects(self, class_name):
        try:
            return [x for x in self.same_floor_objects_list if x.category.name() == class_name]
        except NameError:
            logging.error("Call get_all_objects() before calling get_class_objects()")
            raise

    def get_class_objects_by_id(self, object_id):
        try:
            return [x for x in self.same_floor_objects_list if x.id == object_id]
        except NameError:
            logging.error("Call get_all_objects() before calling get_class_objects()")
            raise

    def find_closest_object_from_class(self, class_name: str, pos_hab: np.array):
        """
        pos_hab: 3d position in habitat world frame
        """
        logging.info(f"class name: {class_name}")
        class_objects = self.get_class_objects(class_name)
        dists_list = []
        for object in class_objects:
            obj_pos = object.aabb.center
            obj_size = object.aabb.sizes
            dist = get_dist_to_bbox_2d(obj_pos[[0, 2]], obj_size[[0, 2]], pos_hab[[0, 2]])
            dists_list.append(dist)

        ranks = np.argsort(np.array(dists_list))
        closest_obj = class_objects[ranks[0]]
        closest_dist = dists_list[ranks[0]]
        return closest_obj, closest_dist

    def find_object_by_id_from_class(self, object_id: str, pos_hab: np.array):
        """
        pos_hab: 3d position in habitat world frame
        """
        class_objects = self.get_class_objects_by_id(object_id)
        logging.info(f"object_id : {object_id}")
        dists_list = []
        for object in class_objects:
            obj_pos = object.aabb.center
            obj_size = object.aabb.sizes
            dist = get_dist_to_bbox_2d(obj_pos[[0, 2]], obj_size[[0, 2]], pos_hab[[0, 2]])
            dists_list.append(dist)

        ranks = np.argsort(np.array(dists_list))
        closest_obj = class_objects[ranks[0]]
        closest_dist = dists_list[ranks[0]]
        return closest_obj, closest_dist

    def visualize_closest_object(self, robot: HabitatLanguageRobot, target_object: dict, vis: bool = False):
        obj_id = target_object["object_id"]
        obj_name = target_object["name"]
        obj_pos = np.array(target_object["position"], dtype=np.float32)  # 3D world coordinates

        logging.info(f"Visualizing object: {obj_name} (ID: {obj_id}) at position: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")

        # Convert 3D world coordinates to 2D grid coordinates
        row, col = self.world_to_grid(robot, obj_pos)
        logging.info(f"Grid (row, col): ({row}, {col})")

        # Convert to cropped image coordinates
        map = robot.map
        row_local = row - map.rmin
        col_local = col - map.cmin
        logging.info(f"Local (row, col): ({row_local}, {col_local})")

        # Visualize
        if vis:
            obs_map = map.get_customized_obstacle_cropped()
            if obs_map is None or obs_map.size == 0:
                logging.error("Error: obs_map is empty or invalid!")
                return target_object, None

            # Create three-channel BGR image
            obs_map_vis = (obs_map * 255).astype(np.uint8)
            obs_map_vis = np.stack([obs_map_vis] * 3, axis=-1)

            # Draw object center point (red)
            cv2.circle(
                obs_map_vis,
                (col_local, row_local),
                radius=5,
                color=(0, 0, 255),
                thickness=-1
            )

            # Display the image
            cv2.imshow(f"Object Visualization: {obj_name} (ID: {obj_id})", obs_map_vis)
            # cv2.waitKey()  # Uncomment if waiting for keypress is desired

        return target_object, None

    def is_task_finished(self):
        # TODO: think about other finish conditions
        # currently, we only check if the agent has called stop aciton for all subgoals
        return self.curr_subgoal_id == self.n_subgoals_in_task

    def test_step(self, sim: habitat_sim.Simulator, robot: HabitatLanguageRobot, action: str, agent_position: np.array = None, vis: bool = False):
        self.actions.append(action)
        if action == "stop":
            if agent_position is None:
                agent = sim.get_agent(0)
                agent_state = agent.get_state()
                agent_position = agent_state.position
            next_subgoal_name = self.goal_classes[self.curr_subgoal_id]
            self.get_all_objects(sim)
            # closest_object, closest_dist = self.find_closest_object_from_class(next_subgoal_name, agent_position)
            closest_object, closest_dist = self.find_and_visualize_closest_object(robot, next_subgoal_name, agent_position, True)
            self.distance_to_subgoals.append(closest_dist)
            if closest_dist < self.config.nav.valid_range:
                self.finished_subgoals.append(self.curr_subgoal_id)
                logging.info(f"###({self.curr_subgoal_id + 1}/{4}) {next_subgoal_name} reached! Distance: {closest_dist}m.###")
            else:
                logging.info(f"###({self.curr_subgoal_id + 1}/{4}) {next_subgoal_name} unreached! Distance: {closest_dist}m.###")

            self.curr_subgoal_id += 1

            # 在stop动作后专门显示并等待按键
            if vis:
                obs = sim.get_sensor_observations(0)
                # 修改这里：设置为True，等待按键
                display_sample({}, obs["color_sensor"], waitkey=True)
        else:
            sim.step(action)
            if vis:
                obs = sim.get_sensor_observations(0)
                # 修改这里：设置为False，不等待按键
                display_sample({}, obs["color_sensor"], waitkey=False)

        if self.is_task_finished():
            self.n_tot_tasks += 1
            self.n_tot_subgoals += self.n_subgoals_in_task
            self.n_success_subgoals += len(self.finished_subgoals)
            if len(self.finished_subgoals) == self.n_subgoals_in_task:
                self.success = True
                self.n_success_tasks += 1
            self.subgoal_success_rate = float(len(self.finished_subgoals)) / self.n_subgoals_in_task

    def test_step_v2(self, sim: habitat_sim.Simulator, robot: HabitatLanguageRobot, action: str, agent_position: np.array = None, vis: bool = False):
        """
        Execute a single step in the navigation task and evaluate success based on object position and radius.

        Args:
            sim (Simulator): Habitat simulator instance.
            robot (HabitatLanguageRobot): Robot instance for navigation.
            action (str): Action to execute (e.g., 'move_forward', 'stop').
            agent_position (np.array, optional): Current position of the agent. If None, retrieved from simulator.
            vis (bool): Whether to visualize the step.
        """
        self.actions.append(action)

        if action == "stop":
            if agent_position is None:
                agent = sim.get_agent(0)
                agent_state = agent.get_state()
                agent_position = np.array(agent_state.position, dtype=np.float32)

            next_subgoal_name = self.goal_classes[self.curr_subgoal_id]
            self.get_all_objects(sim)
            # Find the object in objects_info matching the current subgoal name
            target_object = None
            for obj in self.objects_info:
                if obj["name"] == next_subgoal_name:
                    target_object = obj
                    break

            if target_object is None:
                logging.error(f"No object found in objects_info for subgoal {next_subgoal_name}. Skipping.")
                self.distance_to_subgoals.append(float('inf'))
                self.curr_subgoal_id += 1
                if vis:
                    obs = sim.get_sensor_observations(0)
                    display_sample({}, obs["color_sensor"], waitkey=True)
                return
            closest_object, closest_dist = self.search_calculate_distance_object_by_id(robot, target_object["object_id"], agent_position, vis)
            # Extract object position and radius
            # adjusted_distance, height_diff = self.calculate_dist_base_position(agent_position, target_object)
            logging.info(f"Closest object: {closest_object.category.name()} id: {closest_object.id}")
            self.distance_to_subgoals.append(closest_dist)

            # Evaluate success: distance < 1m and on the same floor
            # if adjusted_distance < 1.0 and is_same_floor:
            if closest_dist < 1.0:
                self.finished_subgoals.append(self.curr_subgoal_id)
                logging.info(f"###({self.curr_subgoal_id + 1}/{self.n_subgoals_in_task}) {next_subgoal_name} reached! Distance: {closest_dist:.2f}m###")
            else:
                logging.info(f"###({self.curr_subgoal_id + 1}/{self.n_subgoals_in_task}) {next_subgoal_name} unreached! Distance: {closest_dist:.2f}m###")
            if vis:
                self.visualize_closest_object(robot, target_object, vis)
            self.curr_subgoal_id += 1

            # Display visualization and wait for keypress on stop action
            if vis:
                obs = sim.get_sensor_observations(0)
                display_sample({}, obs["color_sensor"], waitkey=True)
        else:
            sim.step(action)
            if vis:
                obs = sim.get_sensor_observations(0)
                display_sample({}, obs["color_sensor"], waitkey=False)

        # Update task metrics
        if self.is_task_finished():
            self.n_tot_tasks += 1
            self.n_tot_subgoals += self.n_subgoals_in_task
            self.n_success_subgoals += len(self.finished_subgoals)
            if len(self.finished_subgoals) == self.n_subgoals_in_task:
                self.success = True
                self.n_success_tasks += 1
            self.subgoal_success_rate = float(len(self.finished_subgoals)) / self.n_subgoals_in_task

    def calculate_dist_base_position(self, agent_position, target_object):
        object_position = np.array(target_object["position"], dtype=np.float32)
        object_radius = float(target_object["radius"])

            # Calculate 2D Euclidean distance (ignoring height)
        distance_2d = np.sqrt(
                (agent_position[0] - object_position[0])**2 +
                (agent_position[2] - object_position[2])**2
            )

            # Adjust distance by subtracting object radius
        adjusted_distance = max(0.0, distance_2d - object_radius)

            # Check height difference to ensure same floor (within 1m)
        height_diff = abs(agent_position[1] - object_position[1])
        is_same_floor = height_diff <= 1.0
        return adjusted_distance,height_diff

    def save_single_task_metric(
        self,
        save_path: Union[Path, str],
        forward_dist: float = 0.05,
        turn_angle: float = 1,
    ):
        results_dict = {}
        results_dict["task_id"] = self.task_id
        results_dict["scene"] = self.scene
        results_dict["num_subgoals"] = self.n_subgoals_in_task
        # results_dict["num_subgoal_success"] = self.n_success_subgoals
        results_dict["subgoal_success_rate"] = self.subgoal_success_rate
        results_dict["finished_subgoal_ids"] = self.finished_subgoals
        results_dict["goal_classes"] = self.goal_classes
        results_dict["instruction"] = self.instruction
        results_dict["forward_dict"] = forward_dist
        results_dict["turn_angle"] = turn_angle
        results_dict["init_tf_hab"] = self.init_hab_tf.tolist()
        results_dict["actions"] = self.actions
        with open(save_path, "w") as f:
            json.dump(results_dict, f, indent=4)

    def save_single_task_metric_v2(
        self,
        save_path: Union[Path, str],
        forward_dist: float = 0.05,
        turn_angle: float = 1,
    ):
        results_dict = {}
        results_dict["task_id"] = self.task_id
        results_dict["scene"] = self.scene
        results_dict["num_subgoals"] = self.n_subgoals_in_task
        # results_dict["num_subgoal_success"] = self.n_success_subgoals
        results_dict["subgoal_success_rate"] = self.subgoal_success_rate
        results_dict["finished_subgoal_ids"] = self.finished_subgoals
        results_dict["goal_classes"] = self.goal_classes
        results_dict["forward_dict"] = forward_dist
        results_dict["turn_angle"] = turn_angle
        results_dict["init_tf_hab"] = self.init_hab_tf.tolist()
        results_dict["actions"] = self.actions
        with open(save_path, "w") as f:
            json.dump(results_dict, f, indent=4)

    def find_and_visualize_closest_object(self, robot: HabitatLanguageRobot, class_name: str, pos_hab: np.ndarray, vis: bool = False):
        closest_obj, closest_dist = self.find_closest_object_from_class(class_name, pos_hab)
        obj_pos = closest_obj.aabb.center  # 3D 世界坐标
        map = robot.map
        # 转换为 2D 网格坐标
        row, col = self.world_to_grid(robot, obj_pos)
        logging.info(f"Object 3D position: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}], Grid (row, col): ({row}, {col})")

        # 转换为裁剪图像坐标
        row_local = row - map.rmin
        col_local = col - map.cmin
        logging.info(f"Local (row, col): ({row_local}, {col_local})")

        # 可视化
        if vis:
            obs_map = map.get_customized_obstacle_cropped()
            if obs_map is None or obs_map.size == 0:
                logging.info("Error: obs_map is empty or invalid!")
                return
            # 创建三通道 BGR 图像
            obs_map_vis = (obs_map * 255).astype(np.uint8)
            obs_map_vis = np.stack([obs_map_vis] * 3, axis=-1)

            # 绘制物体中心点（红色）
            cv2.circle(
                obs_map_vis,
                (col_local, row_local),
                radius=5,
                color=(0, 0, 255),
                thickness=-1
            )

            # 保存图像以供调试
            # cv2.imwrite("debug_output.png", obs_map_vis)

            # 显示最终图像
            cv2.imshow("Closest Object Visualization", obs_map_vis)
            # cv2.waitKey()

        return closest_obj, closest_dist

    def search_calculate_distance_object_by_id(self, robot: HabitatLanguageRobot, object_id: str, pos_hab: np.ndarray, vis: bool = False):
        closest_obj, closest_dist = self.find_object_by_id_from_class(object_id, pos_hab)
        obj_pos = closest_obj.aabb.center  # 3D 世界坐标
        map = robot.map
        # 转换为 2D 网格坐标
        row, col = self.world_to_grid(robot, obj_pos)
        logging.info(f"Object 3D position: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}], Grid (row, col): ({row}, {col})")

        # 转换为裁剪图像坐标
        row_local = row - map.rmin
        col_local = col - map.cmin
        logging.info(f"Local (row, col): ({row_local}, {col_local})")

        # 可视化
        if vis:
            obs_map = map.get_customized_obstacle_cropped()
            if obs_map is None or obs_map.size == 0:
                logging.info("Error: obs_map is empty or invalid!")
                return
            # 创建三通道 BGR 图像
            obs_map_vis = (obs_map * 255).astype(np.uint8)
            obs_map_vis = np.stack([obs_map_vis] * 3, axis=-1)

            # 绘制物体中心点（红色）
            cv2.circle(
                obs_map_vis,
                (col_local, row_local),
                radius=5,
                color=(0, 0, 255),
                thickness=-1
            )

            # 保存图像以供调试
            # cv2.imwrite("debug_output.png", obs_map_vis)

            # 显示最终图像
            cv2.imshow("Closest Object Visualization", obs_map_vis)
            # cv2.waitKey()

        return closest_obj, closest_dist

    def world_to_grid(self, robot: HabitatLanguageRobot, pos_3d: np.ndarray) -> Tuple[int, int]:
        """
        Convert Habitat world position to cropped grid coordinates (row, col)
        using the same transformation as `_get_full_map_pose`.
        """
        scene_loader = robot.scene_loader
        map = robot.map
        # Step 1: 构造 Habitat 世界坐标对应的变换矩阵
        tf_hab = self._pos3d_to_tf(pos_3d)

        # Step 2: 应用变换对齐坐标系
        tf = scene_loader.inv_init_base_tf @ scene_loader.base_transform @ tf_hab @ np.linalg.inv(scene_loader.base_transform)

        # Step 3: 提取全局地图坐标 (x, y)
        x, y, _ = tf[:3, 3]

        # Step 4: 转换为网格坐标 (row, col)
        row, col = base_pos2grid_id_3d_3(map.gs, map.cs, x, y, 0)

        return row, col
    def _pos3d_to_tf(self, pos_3d: np.ndarray) -> np.ndarray:
        """Convert 3D position to 4x4 transformation matrix (identity rotation)"""
        tf = np.eye(4)
        tf[:3, 3] = pos_3d
        return tf
