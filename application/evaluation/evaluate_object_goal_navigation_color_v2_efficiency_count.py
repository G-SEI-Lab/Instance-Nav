# Adapted from VLMaps (https://github.com/vlmaps/vlmaps).
# Copyright (c) 2023 Tom-Huang. MIT license; see THIRD_PARTY_LICENSES/VLMaps-LICENSE.

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig
import hydra
import logging

from instance_nav.task.habitat_object_nav_task_color import HabitatObjectNavigationTaskColor
from instance_nav.robot.habitat_lang_robot import HabitatLanguageRobot
from instance_nav.utils.llm_utils import parse_color_object_goal_instruction_v2_1
from instance_nav.utils.matterport3d_categories import mp3dcat_2


def _safe_mean(values: List[float]) -> Optional[float]:
    if len(values) == 0:
        return None
    return float(sum(values) / len(values))


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, set):
        return list(x)
    if hasattr(x, "tolist"):
        return x.tolist()
    if hasattr(x, "item"):
        return x.item()
    return str(x)


def _get_finished_subgoal_ids(task: Any) -> set:
    """
    只读取已有状态，不改变 task 流程。
    """
    for attr in [
        "finished_subgoal_ids",
        "finished_subgoals",
        "finished_sub_goal_ids",
        "success_subgoal_ids",
    ]:
        if hasattr(task, attr):
            value = getattr(task, attr)
            if value is None:
                continue
            if isinstance(value, set):
                return set(value)
            if isinstance(value, (list, tuple)):
                return set(value)

    for container_attr in ["metrics", "metric", "_metrics"]:
        if hasattr(task, container_attr):
            container = getattr(task, container_attr)
            if isinstance(container, dict):
                for key in [
                    "finished_subgoal_ids",
                    "finished_subgoals",
                    "finished_sub_goal_ids",
                    "success_subgoal_ids",
                ]:
                    if key in container and container[key] is not None:
                        return set(container[key])

    return set()


def _has_new_success(before_ids: set, after_ids: set, subgoal_idx: int) -> bool:
    """
    判断 test_step_v2 之后当前 sub-goal 是否新增成功。
    不干预原始 metric，只用于 profiling。
    """
    if subgoal_idx in after_ids and subgoal_idx not in before_ids:
        return True

    if len(after_ids) > len(before_ids):
        return True

    return False


def _append_profiling_to_metric_json(save_path: Path, profiling: Dict[str, Any]) -> None:
    """
    原始 metric 先由 save_single_task_metric_v2 保存；
    这里仅追加 profiling 字段，不影响原始保存逻辑。
    """
    with open(save_path, "r") as f:
        metric_data = json.load(f)

    metric_data["profiling"] = profiling

    metric_data["avg_plan_decision_time_ms_per_subgoal"] = profiling["summary"][
        "avg_plan_decision_time_ms_per_subgoal"
    ]
    metric_data["avg_step_latency_ms_per_step"] = profiling["summary"][
        "avg_step_latency_ms_per_step"
    ]
    metric_data["avg_successful_subgoal_time_s"] = profiling["summary"][
        "avg_successful_subgoal_time_s"
    ]

    with open(save_path, "w") as f:
        json.dump(metric_data, f, indent=2, default=_to_jsonable)


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="object_goal_navigation_cfg",
)
def main(config: DictConfig) -> None:
    logging.basicConfig(
        level=logging.DEBUG, format="[%(filename)s:%(lineno)d] %(message)s"
    )

    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"

    data_dir = Path(config.data_paths.scene_data_dir)

    robot = HabitatLanguageRobot(config)
    object_nav_task = HabitatObjectNavigationTaskColor(config)
    object_nav_task.reset_metrics()

    scene_ids = []
    if isinstance(config.scene_id, int):
        scene_ids.append(config.scene_id)
    else:
        scene_ids = config.scene_id

    for scene_i, scene_id in enumerate(scene_ids):
        robot.setup_scene(scene_id)
        robot.map.init_categories(mp3dcat_2.copy())

        object_nav_task.setup_scene(robot.scene_loader)
        object_nav_task.load_task()

        for task_id in range(len(object_nav_task.task_dict)):
            object_nav_task.setup_task_v2(task_id)

            # ============================================================
            # Parse timing
            #
            # parse_color_object_goal_instruction_v2_1 只执行一次，
            # 但 Successful Sub-goal Time 中每个成功 goal 都加一次 parse_time_s。
            # ============================================================
            parse_start = time.perf_counter()

            object_categories, colors_rgb = parse_color_object_goal_instruction_v2_1(
                object_nav_task.objects_info, robot.map.categories
            )

            parse_end = time.perf_counter()
            parse_time_s = parse_end - parse_start
            parse_time_ms = parse_time_s * 1000.0

            logging.info(f"objects: {object_categories}")
            logging.info(
                f"[Profiling] task={task_id:02}, "
                f"parse_time={parse_time_ms:.3f} ms"
            )

            robot.empty_recorded_actions()
            robot.set_agent_state(object_nav_task.init_hab_tf)

            # ============================================================
            # Profiling containers
            # ============================================================

            # all sub-goals，仅 diagnostic
            all_plan_decision_time_ms_per_subgoal: List[float] = []

            # 主表 Avg. Plan / Decision Time：
            # 只统计成功 sub-goal 的 plan / decision time
            successful_plan_decision_time_ms_per_subgoal: List[float] = []

            # test_step_v2 replay-only latency，仅 diagnostic
            replay_step_latency_ms_per_step: List[float] = []

            # 主表 Avg. Step Latency：
            # effective step latency
            effective_step_latency_ms_per_step: List[float] = []

            # 主表 Avg. Successful Sub-goal Time：
            # parse_time_s + successful plan time + replay time until success
            successful_subgoal_time_s: List[float] = []

            # diagnostic
            successful_subgoal_replay_time_s: List[float] = []
            successful_subgoal_parse_time_s: List[float] = []

            subgoal_records: List[Dict[str, Any]] = []

            # ============================================================
            # 原始 planning loop + 外围计时
            #
            # move_to_color_object:
            # target retrieval + RRT/path planning + action sequence generation
            # ============================================================
            for cat_i, (cat, color, obj_info) in enumerate(
                zip(object_categories, colors_rgb, object_nav_task.objects_info)
            ):
                color_values_gt = obj_info["color_value"]
                logging.info(
                    f"Navigating to category {cat} with color {color}, "
                    f"gt color values {color_values_gt}"
                )

                action_start_idx = len(robot.get_recorded_actions())

                plan_start = time.perf_counter()

                # actions_list = robot.move_to_color_object(
                #     cat, color, color_values_gt, vis=config.nav.vis
                # )

                actions_list = robot.move_to_object(
                    cat, vis=config.nav.vis
                )

                plan_end = time.perf_counter()

                action_end_idx = len(robot.get_recorded_actions())
                plan_decision_time_ms = (plan_end - plan_start) * 1000.0
                plan_decision_time_s = plan_decision_time_ms / 1000.0

                all_plan_decision_time_ms_per_subgoal.append(
                    plan_decision_time_ms
                )

                subgoal_records.append(
                    {
                        "subgoal_id": cat_i,
                        "category": cat,
                        "color": color,
                        "color_values_gt": color_values_gt,

                        "action_start_idx": action_start_idx,
                        "action_end_idx": action_end_idx,
                        "num_actions": action_end_idx - action_start_idx,

                        "parse_time_s_added": parse_time_s,
                        "parse_time_ms_added": parse_time_ms,

                        "plan_decision_time_ms": plan_decision_time_ms,
                        "plan_decision_time_s": plan_decision_time_s,

                        "success": False,

                        # replay-only
                        "subgoal_replay_time_s": None,

                        # final successful time:
                        # parse + plan + replay until successful test_step_v2
                        "subgoal_time_s": None,
                        "subgoal_total_time_s": None,

                        "avg_replay_step_latency_ms": None,
                    }
                )

                logging.info(
                    f"[Profiling] task={task_id:02}, subgoal={cat_i}, "
                    f"plan_decision_time={plan_decision_time_ms:.3f} ms, "
                    f"recorded_actions={action_end_idx - action_start_idx}"
                )

            # 获取已记录的动作列表：保持原始流程
            recorded_actions_list = robot.get_recorded_actions()

            # 重置代理状态：保持原始流程
            robot.set_agent_state(object_nav_task.init_hab_tf)
            logging.info("###repeat task###")

            # ============================================================
            # 原始 replay loop + 外围计时
            #
            # 成功判定放在 test_step_v2(...) 之后。
            #
            # Successful Sub-goal Time:
            #   parse_time_s
            # + current sub-goal move_to_color_object time
            # + current sub-goal replay start 到 test_step_v2 成功后的时间
            # ============================================================

            current_subgoal_idx = 0
            current_subgoal_start_t = time.perf_counter()
            current_subgoal_before_finished_ids = _get_finished_subgoal_ids(
                object_nav_task
            )
            current_subgoal_step_latencies: List[float] = []

            # 跳过可能没有 action 的 sub-goal
            while (
                current_subgoal_idx < len(subgoal_records)
                and subgoal_records[current_subgoal_idx]["num_actions"] <= 0
            ):
                subgoal_records[current_subgoal_idx]["success"] = False
                subgoal_records[current_subgoal_idx]["subgoal_replay_time_s"] = None
                subgoal_records[current_subgoal_idx]["subgoal_time_s"] = None
                subgoal_records[current_subgoal_idx]["subgoal_total_time_s"] = None
                subgoal_records[current_subgoal_idx]["avg_replay_step_latency_ms"] = None
                current_subgoal_idx += 1

            for action_idx, action in enumerate(recorded_actions_list):
                if current_subgoal_idx < len(subgoal_records):
                    current_record = subgoal_records[current_subgoal_idx]

                    if action_idx == current_record["action_start_idx"]:
                        current_subgoal_start_t = time.perf_counter()
                        current_subgoal_before_finished_ids = (
                            _get_finished_subgoal_ids(object_nav_task)
                        )
                        current_subgoal_step_latencies = []

                step_start = time.perf_counter()

                # 原始 test_step_v2，不改流程
                object_nav_task.test_step_v2(
                    robot.sim, robot, action, vis=config.nav.visNav
                )

                step_end = time.perf_counter()
                replay_step_latency_ms = (step_end - step_start) * 1000.0
                replay_step_latency_ms_per_step.append(replay_step_latency_ms)

                # 注意：成功判断放在 test_step_v2 后面
                if current_subgoal_idx < len(subgoal_records):
                    current_subgoal_step_latencies.append(replay_step_latency_ms)

                    current_record = subgoal_records[current_subgoal_idx]
                    after_finished_ids = _get_finished_subgoal_ids(object_nav_task)

                    if (
                        not current_record["success"]
                        and _has_new_success(
                            current_subgoal_before_finished_ids,
                            after_finished_ids,
                            current_subgoal_idx,
                        )
                    ):
                        # replay time until successful test_step_v2
                        subgoal_replay_time_s = (
                            time.perf_counter() - current_subgoal_start_t
                        )

                        plan_decision_time_s = current_record[
                            "plan_decision_time_s"
                        ]

                        # 核心改动：
                        # Successful Sub-goal Time =
                        # parse + plan / decision + replay until successful test_step_v2
                        subgoal_total_time_s = (
                            parse_time_s
                            + plan_decision_time_s
                            + subgoal_replay_time_s
                        )

                        current_record["success"] = True
                        current_record["subgoal_replay_time_s"] = (
                            subgoal_replay_time_s
                        )
                        current_record["subgoal_time_s"] = subgoal_total_time_s
                        current_record["subgoal_total_time_s"] = (
                            subgoal_total_time_s
                        )

                        successful_subgoal_time_s.append(subgoal_total_time_s)
                        successful_subgoal_replay_time_s.append(
                            subgoal_replay_time_s
                        )
                        successful_subgoal_parse_time_s.append(parse_time_s)

                        # 主表 Avg. Plan / Decision Time：
                        # 只统计成功 sub-goal 的 move_to_color_object 时间
                        successful_plan_decision_time_ms_per_subgoal.append(
                            current_record["plan_decision_time_ms"]
                        )

                    # 到达当前 sub-goal 的 action 边界
                    if action_idx + 1 >= current_record["action_end_idx"]:
                        if current_record["subgoal_replay_time_s"] is None:
                            current_record["subgoal_replay_time_s"] = (
                                time.perf_counter() - current_subgoal_start_t
                            )

                        if not current_record["success"]:
                            current_record["subgoal_time_s"] = None
                            current_record["subgoal_total_time_s"] = None

                        current_record["avg_replay_step_latency_ms"] = _safe_mean(
                            current_subgoal_step_latencies
                        )

                        current_subgoal_idx += 1

                        while (
                            current_subgoal_idx < len(subgoal_records)
                            and subgoal_records[current_subgoal_idx]["num_actions"] <= 0
                        ):
                            subgoal_records[current_subgoal_idx]["success"] = False
                            subgoal_records[current_subgoal_idx][
                                "subgoal_replay_time_s"
                            ] = None
                            subgoal_records[current_subgoal_idx]["subgoal_time_s"] = None
                            subgoal_records[current_subgoal_idx][
                                "subgoal_total_time_s"
                            ] = None
                            subgoal_records[current_subgoal_idx][
                                "avg_replay_step_latency_ms"
                            ] = None
                            current_subgoal_idx += 1

            # ============================================================
            # Effective Step Latency
            #
            # 主表 Avg. Step Latency:
            #   (successful decision total + replay total) / total steps
            #
            # 这里不把 parse 摊入 step latency；
            # parse 已经计入 Successful Sub-goal Time。
            # 如果后面想报 e2e step latency，可以读 e2e_effective_step_latency_* 字段。
            # ============================================================
            total_steps = len(replay_step_latency_ms_per_step)

            successful_plan_decision_total_s = (
                sum(successful_plan_decision_time_ms_per_subgoal) / 1000.0
            )

            replay_step_latency_total_s = (
                sum(replay_step_latency_ms_per_step) / 1000.0
            )

            if total_steps > 0:
                plan_overhead_ms_per_step = (
                    successful_plan_decision_total_s / total_steps * 1000.0
                )

                effective_step_latency_ms_per_step = [
                    replay_ms + plan_overhead_ms_per_step
                    for replay_ms in replay_step_latency_ms_per_step
                ]

                avg_effective_step_latency_ms = _safe_mean(
                    effective_step_latency_ms_per_step
                )
            else:
                plan_overhead_ms_per_step = None
                effective_step_latency_ms_per_step = []
                avg_effective_step_latency_ms = None

            effective_step_latency_total_s = (
                successful_plan_decision_total_s + replay_step_latency_total_s
            )

            # diagnostic: 如果你后面想把 parse 也摊进 step latency，可用这个
            successful_parse_total_s = (
                sum(successful_subgoal_parse_time_s)
                if len(successful_subgoal_parse_time_s) > 0
                else 0.0
            )
            e2e_effective_step_latency_total_s = (
                successful_parse_total_s
                + successful_plan_decision_total_s
                + replay_step_latency_total_s
            )
            e2e_effective_step_latency_ms_per_step = (
                e2e_effective_step_latency_total_s / total_steps * 1000.0
                if total_steps > 0
                else None
            )

            save_dir = robot.scene_loader.data_dir / (
                config.map_config.map_type + "_obj_nav_results_eff"
            )

            os.makedirs(save_dir, exist_ok=True)

            save_path = save_dir / f"{task_id:02}.json"
            logging.info(f"###save path: {save_path}###")

            # 原始保存逻辑不变
            object_nav_task.save_single_task_metric_v2(save_path)

            profiling = {
                "method_name": "FG-Nav",
                "method_type": "offline_map_based",
                "scene": str(scene_id),
                "task_id": int(task_id),

                "profiling_scope": {
                    "parse_time": (
                        "parse_color_object_goal_instruction_v2_1 is measured once "
                        "per task and added to each successful sub-goal time."
                    ),
                    "plan_decision_time": (
                        "target retrieval + RRT/path planning + "
                        "action sequence generation; main average only counts "
                        "successful sub-goals."
                    ),
                    "step_latency": (
                        "effective step latency for FG-Nav, computed as "
                        "(successful plan/decision total time + replay step total time) "
                        "/ total execution steps."
                    ),
                    "replay_step_latency": (
                        "diagnostic only; measured around object_nav_task.test_step_v2."
                    ),
                    "successful_subgoal_time": (
                        "parse time + successful sub-goal plan/decision time + "
                        "replay time until successful test_step_v2."
                    ),
                    "offline_mapping": "excluded",
                    "vis_during_planning": bool(config.nav.vis),
                    "vis_during_execution": bool(config.nav.visNav),
                },

                "units": {
                    "avg_plan_decision_time_ms_per_subgoal": (
                        "ms / successful sub-goal"
                    ),
                    "avg_step_latency_ms_per_step": "ms / step",
                    "avg_replay_step_latency_ms_per_step": "ms / step",
                    "avg_successful_subgoal_time_s": (
                        "s / successful sub-goal"
                    ),
                    "parse_time_s": "s / task, added to each successful sub-goal",
                },

                "summary": {
                    # 主表 1:
                    # 只统计成功 sub-goal 的 decision time
                    "avg_plan_decision_time_ms_per_subgoal": _safe_mean(
                        successful_plan_decision_time_ms_per_subgoal
                    ),
                    "avg_plan_decision_time_ms_per_successful_subgoal": _safe_mean(
                        successful_plan_decision_time_ms_per_subgoal
                    ),

                    # diagnostic:
                    "avg_plan_decision_time_ms_per_all_subgoals": _safe_mean(
                        all_plan_decision_time_ms_per_subgoal
                    ),

                    # 主表 2:
                    "avg_step_latency_ms_per_step": avg_effective_step_latency_ms,

                    # diagnostic only:
                    "avg_replay_step_latency_ms_per_step": _safe_mean(
                        replay_step_latency_ms_per_step
                    ),

                    # 主表 3:
                    "avg_successful_subgoal_time_s": _safe_mean(
                        successful_subgoal_time_s
                    ),

                    # diagnostics:
                    "avg_replay_time_s_per_successful_subgoal": _safe_mean(
                        successful_subgoal_replay_time_s
                    ),
                    "avg_parse_time_s_added_per_successful_subgoal": _safe_mean(
                        successful_subgoal_parse_time_s
                    ),

                    "parse_time_s": parse_time_s,
                    "parse_time_ms": parse_time_ms,

                    "num_all_subgoals": len(all_plan_decision_time_ms_per_subgoal),
                    "num_successful_subgoals_profiled": len(
                        successful_subgoal_time_s
                    ),
                    "num_execution_steps": total_steps,
                    "num_total_actions": len(recorded_actions_list),

                    "successful_plan_decision_total_s": (
                        successful_plan_decision_total_s
                    ),
                    "replay_step_latency_total_s": replay_step_latency_total_s,
                    "effective_step_latency_total_s": (
                        effective_step_latency_total_s
                    ),
                    "plan_overhead_ms_per_step": plan_overhead_ms_per_step,

                    # diagnostic: parse + successful decision + replay, then / step
                    "successful_parse_total_s": successful_parse_total_s,
                    "e2e_effective_step_latency_total_s": (
                        e2e_effective_step_latency_total_s
                    ),
                    "e2e_effective_step_latency_ms_per_step": (
                        e2e_effective_step_latency_ms_per_step
                    ),
                },

                # 主字段：给统计脚本复用
                "plan_decision_time_ms_per_subgoal": (
                    successful_plan_decision_time_ms_per_subgoal
                ),
                "successful_plan_decision_time_ms_per_subgoal": (
                    successful_plan_decision_time_ms_per_subgoal
                ),

                # diagnostic: 所有 sub-goal
                "plan_decision_time_ms_per_all_subgoals": (
                    all_plan_decision_time_ms_per_subgoal
                ),

                "decision_total_s": successful_plan_decision_total_s,

                # 主字段：effective step latency
                "step_latency_ms_per_step": effective_step_latency_ms_per_step,
                "step_latency_total_s": effective_step_latency_total_s,
                "avg_step_latency_ms_per_step": avg_effective_step_latency_ms,
                "effective_step_latency_ms_per_step": (
                    effective_step_latency_ms_per_step
                ),

                # diagnostic only: replay-only latency
                "replay_step_latency_ms_per_step": (
                    replay_step_latency_ms_per_step
                ),
                "replay_step_latency_total_s": replay_step_latency_total_s,
                "avg_replay_step_latency_ms_per_step": _safe_mean(
                    replay_step_latency_ms_per_step
                ),

                # 主字段：parse + decision + replay until success
                "successful_subgoal_time_s": successful_subgoal_time_s,
                "successful_subgoal_times_s": successful_subgoal_time_s,

                # diagnostics
                "successful_subgoal_replay_time_s": (
                    successful_subgoal_replay_time_s
                ),
                "successful_subgoal_parse_time_s": (
                    successful_subgoal_parse_time_s
                ),
                "parse_time_s": parse_time_s,
                "parse_time_ms": parse_time_ms,

                "subgoal_records": subgoal_records,
            }

            _append_profiling_to_metric_json(save_path, profiling)

            logging.info(
                f"[Profiling Summary] task={task_id:02}, "
                f"parse_time={parse_time_ms:.3f} ms, "
                f"avg_plan_decision_time_successful="
                f"{profiling['summary']['avg_plan_decision_time_ms_per_subgoal']} "
                f"ms/successful sub-goal, "
                f"avg_plan_decision_time_all="
                f"{profiling['summary']['avg_plan_decision_time_ms_per_all_subgoals']} "
                f"ms/all sub-goal, "
                f"avg_effective_step_latency="
                f"{profiling['summary']['avg_step_latency_ms_per_step']} "
                f"ms/step, "
                f"avg_replay_step_latency="
                f"{profiling['summary']['avg_replay_step_latency_ms_per_step']} "
                f"ms/step, "
                f"avg_successful_subgoal_time="
                f"{profiling['summary']['avg_successful_subgoal_time_s']} "
                f"s/successful sub-goal"
            )


if __name__ == "__main__":
    main()
