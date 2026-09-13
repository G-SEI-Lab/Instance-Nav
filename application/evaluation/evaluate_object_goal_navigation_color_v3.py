# Adapted from VLMaps (https://github.com/vlmaps/vlmaps).
# Copyright (c) 2023 Tom-Huang. MIT license; see THIRD_PARTY_LICENSES/VLMaps-LICENSE.

import os
from pathlib import Path
from omegaconf import DictConfig
import hydra
import logging
from instance_nav.task.habitat_object_nav_task_across import HabitatObjectNavigationTaskAcross
from instance_nav.robot.habitat_lang_robot import HabitatLanguageRobot
from instance_nav.utils.llm_utils import parse_color_object_goal_instruction_v3
from instance_nav.utils.matterport3d_categories import (mp3dcat_2)

@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="object_goal_navigation_cfg",
)
def main(config: DictConfig) -> None:
    logging.basicConfig(
        level= logging.DEBUG,
        format='[%(filename)s:%(lineno)d] %(message)s'
        )
    # 设置环境变量，关闭日志输出
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"

    # 获取数据目录
    data_dir = Path(config.data_paths.scene_data_dir)

    # 设置地图路径，设置地图和相机参数
    # 设置机器人和操控参数
    robot = HabitatLanguageRobot(config)

    # 创建导航任务实例，没有其它操作
    object_nav_task = HabitatObjectNavigationTaskAcross(config)

    # 重置导航任务的度量指标
    object_nav_task.reset_metrics()

    # Ablation switch: keep the parsed floor information, but optionally exclude
    # it from instance matching. With color weight 1.0, floor=None means that
    # candidates on every floor are ranked using color only. The selected
    # instance height is then passed to the unchanged 3D navigation pipeline.
    use_floor_for_matching = config.nav.get("use_floor_for_matching", True)

    # 获取场景ID列表
    scene_ids = []
    if isinstance(config.scene_id, int):
        scene_ids.append(config.scene_id)
    else:
        scene_ids = config.scene_id

    logging.info(f"Effective scene_ids: {list(scene_ids)}")

    # 遍历场景ID列表
    for scene_i, scene_id in enumerate(scene_ids):
        scene_data_dir = robot.get_scene_data_dirs()[scene_id]
        logging.info(f"Resolved scene_id={scene_id} to {scene_data_dir}")
        required_files = ["color_object_3d_nav_dataset.json", "stairs.json"]
        missing_files = [
            filename
            for filename in required_files
            if not (scene_data_dir / filename).exists()
        ]
        if missing_files:
            raise FileNotFoundError(
                f"scene_id={scene_id} ({scene_data_dir.name}) is not a multi-floor "
                f"evaluation scene; missing: {', '.join(missing_files)}"
            )

        # 设置场景
        robot.setup_scene(scene_id)

        robot.load_stairs()

        robot.generate_3d_obstacle_map()

        # 初始化类别
        robot.map.init_categories(mp3dcat_2.copy())

        # 设置导航任务的场景
        object_nav_task.setup_scene(robot.scene_loader)

        # 加载导航任务
        object_nav_task.load_task()

        # 遍历任务ID列表
        for task_id in range(len(object_nav_task.task_dict)):
            """
            初始化任务的各种属性，包括任务ID、空间变换矩阵、地图尺寸、场景、指令、目标对象列表
            以及一些度量属性（如子目标数量、当前子目标ID、已完成的子目标列表、到子目标的距离、任务成功状态和动作列表）
            """
            object_nav_task.setup_task_v2(task_id)


            # 解析目标指令中的物体类别,调用GPT API解析指令，返回一个列表，每个元素都是一个物体
            object_categories, colors_rgb, absolute_floors = parse_color_object_goal_instruction_v3(object_nav_task.objects_info, robot.map.categories)

            # 打印目标指令
            logging.info(f"objects: {object_categories}")

            # 清空已记录的动作
            robot.empty_recorded_actions()
            """
            设置代理状态,读取object_navigation_tasks.json，
            由数据集提供，包含所有任务的初始状态，
            这里设置的是初始位姿
            """
            robot.set_agent_state(object_nav_task.init_hab_tf)

            # 遍历物体类别列表
            for cat_i, (cat, color, floor, obj_info) in enumerate(zip(object_categories, colors_rgb, absolute_floors, object_nav_task.objects_info)):
                # 打印导航到的类别
                color_values_gt = obj_info["color_value"]
                matching_floor = floor if use_floor_for_matching else None
                logging.info(
                    f"Navigating to category {cat} with color {color}, "
                    f"instruction floor {floor}, matching floor {matching_floor}, "
                    f"gt color values {color_values_gt}"
                )

                # 执行移动到物体的动作
                actions_list = robot.move_to_color_3d_object(
                    cat,
                    color,
                    color_values_gt,
                    matching_floor,
                    weight=1.0,
                    vis=config.nav.vis,
                )

            # 获取已记录的动作列表
            recorded_actions_list = robot.get_recorded_actions()

            # 重置代理状态
            robot.set_agent_state(object_nav_task.init_hab_tf)
            logging.info("###repeat task###")
            # 遍历已记录的动作列表
            for action in recorded_actions_list:
                # 执行测试步骤
                object_nav_task.test_step_v2(robot.sim, robot, action, vis=config.nav.vis)

            # 获取保存目录
            result_suffix = (
                "_color_3d_obj_nav_results"
                if use_floor_for_matching
                else "_color_3d_obj_nav_results_no_floor_matching"
            )
            save_dir = robot.scene_loader.data_dir / (
                config.map_config.map_type + result_suffix
            )

            # 创建保存目录
            os.makedirs(save_dir, exist_ok=True)

            # 获取保存路径
            save_path = save_dir / f"{task_id:02}.json"

            # 保存单个任务的度量指标
            object_nav_task.save_single_task_metric_v2(save_path)


if __name__ == "__main__":
    main()
