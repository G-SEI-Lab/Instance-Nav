# Adapted from VLMaps (https://github.com/vlmaps/vlmaps).
# Copyright (c) 2023 Tom-Huang. MIT license; see THIRD_PARTY_LICENSES/VLMaps-LICENSE.

import numpy as np
import cv2
from scipy.spatial.distance import cdist
from scipy.ndimage import label
import pyvisgraph as vg
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, List, Dict
import logging
import time

def get_segment_islands_pos(segment_map, label_id, detect_internal_contours=False):
    mask = segment_map == label_id
    mask = mask.astype(np.uint8)
    detect_type = cv2.RETR_EXTERNAL
    if detect_internal_contours:
        detect_type = cv2.RETR_TREE

    contours, hierarchy = cv2.findContours(mask, detect_type, cv2.CHAIN_APPROX_SIMPLE)
    # convert contours back to numpy index order
    contours_list = []
    for contour in contours:
        tmp = contour.reshape((-1, 2))
        tmp_1 = np.stack([tmp[:, 1], tmp[:, 0]], axis=1)
        contours_list.append(tmp_1)

    centers_list = []
    bbox_list = []
    for c in contours_list:
        xmin = np.min(c[:, 0])
        xmax = np.max(c[:, 0])
        ymin = np.min(c[:, 1])
        ymax = np.max(c[:, 1])
        bbox_list.append([xmin, xmax, ymin, ymax])

        centers_list.append([(xmin + xmax) / 2, (ymin + ymax) / 2])

    return contours_list, centers_list, bbox_list, hierarchy


def find_closest_points_between_two_contours(obs_map, contour_a, contour_b):
    a = np.zeros_like(obs_map, dtype=np.uint8)
    b = np.zeros_like(obs_map, dtype=np.uint8)
    cv2.drawContours(a, [contour_a[:, [1, 0]]], 0, 255, 1)
    cv2.drawContours(b, [contour_b[:, [1, 0]]], 0, 255, 1)
    rows_a, cols_a = np.where(a == 255)
    rows_b, cols_b = np.where(b == 255)
    pts_a = np.concatenate([rows_a.reshape((-1, 1)), cols_a.reshape((-1, 1))], axis=1)
    pts_b = np.concatenate([rows_b.reshape((-1, 1)), cols_b.reshape((-1, 1))], axis=1)
    dists = cdist(pts_a, pts_b)
    id = np.argmin(dists)
    ida, idb = np.unravel_index(id, dists.shape)
    return [rows_a[ida], cols_a[ida]], [rows_b[idb], cols_b[idb]]


def point_in_contours(obs_map, contours_list, point):
    """
    obs_map: np.ndarray, 1 free, 0 occupied
    contours_list: a list of cv2 contours [[(col1, row1), (col2, row2), ...], ...]
    point: (row, col)
    """
    row, col = int(point[0]), int(point[1])
    ids = []
    print("contours num: ", len(contours_list))
    for con_i, contour in enumerate(contours_list):
        contour_cv2 = contour[:, [1, 0]]
        con_mask = np.zeros_like(obs_map, dtype=np.uint8)
        cv2.drawContours(con_mask, [contour_cv2], 0, 255, -1)
        # con_mask_copy = con_mask.copy()
        # cv2.circle(con_mask_copy, (col, row), 10, 0, 3)
        # cv2.imshow("contour_mask", con_mask_copy)
        # cv2.waitKey()
        if con_mask[row, col] == 255:
            ids.append(con_i)

    return ids


def build_visgraph_with_obs_map(obs_map, use_internal_contour=False, internal_point=None, vis=False, detect_internal_contours: bool=False):
    # 将障碍物地图转换为可视化图像
    obs_map_vis = (obs_map[:, :, None] * 255).astype(np.uint8)
    obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])

    # 如果开启可视化，则显示障碍物地图（可选）
    if vis:
        cv2.imshow("obs", obs_map_vis)

    # 获取障碍物地图中的轮廓、中心点、边界框和层次结构
    contours_list, centers_list, bbox_list, hierarchy = get_segment_islands_pos(
        obs_map, 0, detect_internal_contours=use_internal_contour
    )

    if use_internal_contour:
        ids = point_in_contours(obs_map, contours_list, internal_point)
        assert len(ids) == 2, f"The internal point is not in 2 contours, but {len(ids)}"
        point_a, point_b = find_closest_points_between_two_contours(
            obs_map, contours_list[ids[0]], contours_list[ids[1]]
        )
        obs_map = cv2.line((obs_map * 255).astype(np.uint8), (point_a[1], point_a[0]), (point_b[1], point_b[0]), 255, 5)
        obs_map = obs_map == 255
        contours_list, centers_list, bbox_list, hierarchy = get_segment_islands_pos(
            obs_map, 0, detect_internal_contours=False
        )

    poly_list = []

    # 遍历所有轮廓
    for contour in contours_list:
        # 如果开启可视化，则绘制轮廓（不再调用 cv2.imshow 和 cv2.waitKey）
        if vis:
            contour_cv2 = contour[:, [1, 0]]
            cv2.drawContours(obs_map_vis, [contour_cv2], 0, (0, 255, 0), 3)

    # 在所有轮廓绘制完成后，统一显示
    # if vis:
    #     cv2.imshow("obs", obs_map_vis)
    #     cv2.waitKey()  # 只等待一次，显示所有轮廓

    # 提取轮廓点并构建 VisGraph
    for contour in contours_list:
        contour_pos = []
        for [row, col] in contour:
            contour_pos.append(vg.Point(row, col))
        poly_list.append(contour_pos)

    g = vg.VisGraph()
    g.build(poly_list, workers=4)
    return g



def get_nearby_position(goal: Tuple[float, float], G: vg.VisGraph) -> Tuple[float, float]:
    for dr, dc in zip([-1, 1, -1, 1], [-1, -1, 1, 1]):
        goalvg_new = vg.Point(goal[0] + dr, goal[1] + dc)
        poly_id_new = G.point_in_polygon(goalvg_new)
        if poly_id_new == -1:
            return (goal[0] + dr, goal[1] + dc)


def plan_to_pos_v2(start, goal, obstacles, G: vg.VisGraph = None, vis=False):
    """
    plan a path on a cropped obstacles map represented by a graph.
    Start and goal are tuples of (row, col) in the map.
    """

    # print("start: ", start)
    # print("goal: ", goal)
    if vis:
        obs_map_vis = (obstacles[:, :, None] * 255).astype(np.uint8)
        obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])
        obs_map_vis = cv2.circle(obs_map_vis, (int(start[1]), int(start[0])), 3, (255, 0, 0), -1)
        obs_map_vis = cv2.circle(obs_map_vis, (int(goal[1]), int(goal[0])), 3, (0, 0, 255), -1)
        cv2.imshow("planned path", obs_map_vis)
        cv2.waitKey()

    path = []
    startvg = vg.Point(start[0], start[1])
    if obstacles[int(start[0]), int(start[1])] == 0:
        print("start in obstacles")
        rows, cols = np.where(obstacles == 1)
        dist_sq = (rows - start[0]) ** 2 + (cols - start[1]) ** 2
        id = np.argmin(dist_sq)
        new_start = [rows[id], cols[id]]
        path.append(new_start)
        startvg = vg.Point(new_start[0], new_start[1])

    goalvg = vg.Point(goal[0], goal[1])
    poly_id = G.point_in_polygon(goalvg)
    if obstacles[int(goal[0]), int(goal[1])] == 0:
        print("goal in obstacles")
        try:
            goalvg = G.closest_point(goalvg, poly_id, length=1)
        except:
            goal_new = get_nearby_position(goal, G)
            goalvg = vg.Point(goal_new[0], goal_new[1])

        print("goalvg: ", goalvg)
    path_vg = G.shortest_path(startvg, goalvg)

    for point in path_vg:
        subgoal = [point.x, point.y]
        path.append(subgoal)
    # print(path)

    # check the final goal is not in obstacles
    # if obstacles[int(goal[0]), int(goal[1])] == 0:
    #     path = path[:-1]

    if vis:
        obs_map_vis = (obstacles[:, :, None] * 255).astype(np.uint8)
        obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])

        for i, point in enumerate(path):
            subgoal = (int(point[1]), int(point[0]))
            # print(i, subgoal)
            obs_map_vis = cv2.circle(obs_map_vis, subgoal, 5, (255, 0, 0), -1)
            if i > 0:
                cv2.line(obs_map_vis, last_subgoal, subgoal, (255, 0, 0), 2)
            last_subgoal = subgoal
        obs_map_vis = cv2.circle(obs_map_vis, (int(start[1]), int(start[0])), 5, (0, 255, 0), -1)
        obs_map_vis = cv2.circle(obs_map_vis, (int(goal[1]), int(goal[0])), 5, (0, 0, 255), -1)

        seg = Image.fromarray(obs_map_vis)
        cv2.imshow("planned path", obs_map_vis)
        cv2.waitKey()

    return path

def adjust_if_in_obstacle(point, obstacles, min_region_size=1000, max_distance=100):
    """
    如果点在障碍物里，将其调整到一个合理的自由点，优先选择与原始点较近的点。
    - point: numpy array [row, col]
    - obstacles: 2D numpy array, 0=障碍物, 1=自由空间
    - min_region_size: 连通区域最小像素数
    - max_distance: 最大允许距离（像素），超过则随机选择
    """
    point = np.array(point)
    r, c = int(point[0]), int(point[1])

    if obstacles[r, c] > 0:
        return point  # 已经在自由空间，无需调整

    # 标记自由空间连通区域
    free_space = obstacles > 0
    labeled, num_features = label(free_space)

    tried_labels = set()
    height, width = obstacles.shape

    # 找到地图中所有自由点
    free_rows, free_cols = np.where(free_space)
    free_points = np.column_stack((free_rows, free_cols))

    # 计算所有自由点与原始点的欧几里得距离
    distances = np.sqrt(np.sum((free_points - point) ** 2, axis=1))

    while True:
        # 按距离排序，选择最近的点
        sorted_indices = np.argsort(distances)
        for idx in sorted_indices:
            new_r, new_c = free_points[idx]
            region_label = labeled[new_r, new_c]

            # 避免重复尝试同一个小区域
            if region_label in tried_labels:
                continue

            region_size = np.sum(labeled == region_label)

            if region_size >= min_region_size:
                # 检查距离是否在 max_distance 内
                if distances[idx] <= max_distance:
                    return np.array([new_r, new_c])
                else:
                    # 如果所有近点都不合适，随机返回一个符合条件的点
                    return np.array([new_r, new_c])

            tried_labels.add(region_label)

        # 如果循环结束仍未找到，随机选择一个符合条件的点
        idx = np.random.randint(0, len(free_rows))
        new_r, new_c = free_rows[idx], free_cols[idx]
        region_label = labeled[new_r, new_c]
        if region_label not in tried_labels and np.sum(labeled == region_label) >= min_region_size:
            return np.array([new_r, new_c])

def plan_to_pos_v3(start, goal, obstacles, vis=False):
    """
    Plan a path using RRT* (Rapidly-exploring Random Tree Star).
    Start and goal are (row, col) in the map.
    obstacles: 2D numpy array (0 = obstacle, 1 = free space).
    """

    # 转成 numpy 数组
    start = np.array(start, dtype=np.float64)
    goal = np.array(goal, dtype=np.float64)

    start = adjust_if_in_obstacle(start, obstacles)
    goal = adjust_if_in_obstacle(goal, obstacles)

    # 碰撞检测函数
    def is_collision_free(p1, p2, obs):
        p1 = np.array(p1, dtype=np.float64)
        p2 = np.array(p2, dtype=np.float64)
        direction = p2 - p1
        distance = np.linalg.norm(direction)
        if distance < 1e-6:
            return True
        direction /= distance
        num_samples = int(distance * 2) + 2
        for i in range(num_samples):
            point = p1 + (i / (num_samples - 1)) * (p2 - p1)
            r = int(np.round(point[0]))
            c = int(np.round(point[1]))
            if r < 0 or r >= obs.shape[0] or c < 0 or c >= obs.shape[1] or obs[r, c] == 0:
                return False
        return True

    # 参数
    max_iter = 8000
    step_size = 30.0
    goal_sample_prob = 0.2
    goal_threshold = 20.0
    rewire_radius = 50.0
    height, width = obstacles.shape
    vis_interval = 200

    free_points = np.argwhere(obstacles == 1)

    # 初始化树
    tree = [start.copy()]
    parents = [-1]
    costs = [0.0]

    # 可视化底图
    if vis:
        obs_map_vis = (obstacles[:, :, None] * 255).astype(np.uint8)
        obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])
        obs_map_vis = cv2.circle(obs_map_vis, (int(start[1]), int(start[0])), 3, (255, 0, 0), -1)
        obs_map_vis = cv2.circle(obs_map_vis, (int(goal[1]), int(goal[0])), 3, (0, 0, 255), -1)
    start_time = time.perf_counter()
    for i in range(max_iter):
        # 采样
        if np.random.rand() < goal_sample_prob:
            sample = goal
        else:
            idx = np.random.randint(len(free_points))
            sample = free_points[idx]

        sample = np.array(sample, dtype=np.float64)
        r, c = int(sample[0]), int(sample[1])
        if r < 0 or r >= height or c < 0 or c >= width or obstacles[r, c] == 0:
            continue

        # 最近点
        tree_array = np.stack(tree)
        dists = cdist(sample.reshape(1, 2), tree_array)
        nearest_idx = np.argmin(dists)
        nearest = tree[nearest_idx]

        # 扩展
        dir_vec = sample - nearest
        dist = np.linalg.norm(dir_vec)
        if dist > step_size:
            new_pos = nearest + (dir_vec / dist) * step_size
        else:
            new_pos = sample

        if not is_collision_free(nearest, new_pos, obstacles):
            continue

        # Step 1: 找邻居
        dists_new = cdist(new_pos.reshape(1, 2), tree_array)[0]
        neighbor_idx = np.where(dists_new < rewire_radius)[0]

        # Step 2: 选择父节点（最小代价）
        min_cost = costs[nearest_idx] + np.linalg.norm(new_pos - nearest)
        best_parent = nearest_idx
        for ni in neighbor_idx:
            if is_collision_free(tree[ni], new_pos, obstacles):
                new_cost = costs[ni] + np.linalg.norm(new_pos - tree[ni])
                if new_cost < min_cost:
                    min_cost = new_cost
                    best_parent = ni

        # Step 3: 插入新节点
        tree.append(new_pos)
        parents.append(best_parent)
        costs.append(min_cost)
        new_idx = len(tree) - 1

        # Step 4: 重连 (Rewire)
        for ni in neighbor_idx:
            if ni == best_parent:
                continue
            if is_collision_free(new_pos, tree[ni], obstacles):
                new_cost = costs[new_idx] + np.linalg.norm(tree[ni] - new_pos)
                if new_cost < costs[ni]:
                    parents[ni] = new_idx
                    costs[ni] = new_cost

        # Step 5: 检查是否到达目标
        if np.linalg.norm(new_pos - goal) < goal_threshold:
            if is_collision_free(new_pos, goal, obstacles):
                tree.append(goal)
                parents.append(new_idx)  # ✅ 修复死循环：goal 的父节点是 new_pos
                costs.append(costs[new_idx] + np.linalg.norm(goal - new_pos))
                break

        # 可视化生长
        if vis and (i + 1) % vis_interval == 0:
            logging.info(f"Visualizing tree {i + 1}...")
            temp_map = obs_map_vis.copy()
            for idx, node in enumerate(tree):
                node_pos = (int(node[1]), int(node[0]))
                temp_map = cv2.circle(temp_map, node_pos, 1, (0, 255, 255), -1)
                if parents[idx] != -1:
                    parent_pos = (int(tree[parents[idx]][1]), int(tree[parents[idx]][0]))
                    temp_map = cv2.line(temp_map, parent_pos, node_pos, (0, 255, 255), 1)
            cv2.imshow("RRT* Tree Growth", temp_map)
            cv2.waitKey(1)

    # 路径回溯
    path = []
    if len(tree) > 1 and np.linalg.norm(tree[-1] - goal) < goal_threshold + 1e-6:
        current = len(tree) - 1
        while current != -1:
            path.append(tree[current].tolist())
            current = parents[current]
        path.reverse()
    else:
        print("No path found after max iterations")
        return []

    path = optimize_path_with_clearance(path, obstacles, safe_margin=3)
    end_time = time.perf_counter()
    processing_time = end_time - start_time
    logging.info(f"'############ RRT star run time:':{processing_time:.3f} seconds############")
    # 可视化最终路径
    if vis:
        for i, point in enumerate(path):
            subgoal = (int(point[1]), int(point[0]))
            obs_map_vis = cv2.circle(obs_map_vis, subgoal, 3, (255, 0, 0), -1)
            if i > 0:
                last_subgoal = (int(path[i - 1][1]), int(path[i - 1][0]))
                cv2.line(obs_map_vis, last_subgoal, subgoal, (255, 0, 0), 2)
        obs_map_vis = cv2.circle(obs_map_vis, (int(start[1]), int(start[0])), 5, (0, 255, 0), -1)
        obs_map_vis = cv2.circle(obs_map_vis, (int(goal[1]), int(goal[0])), 5, (0, 0, 255), -1)
        cv2.imshow("Final RRT* Path", obs_map_vis)
        cv2.waitKey()

    return path

def optimize_path_with_clearance(path, obstacles, safe_margin=5):
    """
    调整路径，使其点尽量远离障碍物 safe_margin 像素
    """
    path = np.array(path)

    # 计算距离场 (自由空间到障碍物的最短距离)
    obs_uint8 = (obstacles == 0).astype(np.uint8)  # 障碍物=1
    dist_map = cv2.distanceTransform(1 - obs_uint8, cv2.DIST_L2, 3)

    # 优化后的路径
    new_path = []
    for point in path:
        r, c = int(point[0]), int(point[1])
        if r < 0 or r >= dist_map.shape[0] or c < 0 or c >= dist_map.shape[1]:
            new_path.append(point)
            continue

        # 当前点到障碍物的距离
        d = dist_map[r, c]
        if d >= safe_margin:
            new_path.append(point)
            continue

        # 梯度方向（远离障碍物的方向）
        gx = dist_map[min(r+1, dist_map.shape[0]-1), c] - dist_map[max(r-1,0), c]
        gy = dist_map[r, min(c+1, dist_map.shape[1]-1)] - dist_map[r, max(c-1,0)]
        grad = np.array([gx, gy], dtype=float)

        if np.linalg.norm(grad) > 1e-6:
            grad /= np.linalg.norm(grad)
            shift = (safe_margin - d) * grad
            new_r = int(np.clip(r + shift[0], 0, dist_map.shape[0]-1))
            new_c = int(np.clip(c + shift[1], 0, dist_map.shape[1]-1))

            # 确保新点不在障碍物里
            if obstacles[new_r, new_c] == 1:
                new_path.append([new_r, new_c])
            else:
                new_path.append(point)  # 移动失败，保留原点
        else:
            new_path.append(point)

    return np.array(new_path).tolist()


def get_bbox(center, size):
    """
    Return min corner and max corner coordinate
    """
    min_corner = center - size / 2
    max_corner = center + size / 2
    return min_corner, max_corner


def get_dist_to_bbox_2d(center, size, pos):
    min_corner_2d, max_corner_2d = get_bbox(center, size)

    dx = pos[0] - center[0]
    dy = pos[1] - center[1]

    if pos[0] < min_corner_2d[0] or pos[0] > max_corner_2d[0]:
        if pos[1] < min_corner_2d[1] or pos[1] > max_corner_2d[1]:
            """
            star region
            *  |  |  *
            ___|__|___
               |  |
            ___|__|___
               |  |
            *  |  |  *
            """

            dx_c = np.abs(dx) - size[0] / 2
            dy_c = np.abs(dy) - size[1] / 2
            dist = np.sqrt(dx_c * dx_c + dy_c * dy_c)
            return dist
        else:
            """
            star region
               |  |
            ___|__|___
            *  |  |  *
            ___|__|___
               |  |
               |  |
            """
            dx_b = np.abs(dx) - size[0] / 2
            return dx_b
    else:
        if pos[1] < min_corner_2d[1] or pos[1] > max_corner_2d[1]:
            """
            star region
               |* |
            ___|__|___
               |  |
            ___|__|___
               |* |
               |  |
            """
            dy_b = np.abs(dy) - size[1] / 2
            return dy_b

        """
        star region
           |  |
        ___|__|___
           |* |
        ___|__|___
           |  |
           |  |
        """
        return 0


def plan_to_pos_v3_bidirectional(start, goal, obstacles, vis=False, floor: int = 0):
    """
    Plan a path using Bidirectional RRT* (Rapidly-exploring Random Tree Star).
    Start and goal are (row, col) in the map.
    obstacles: 2D numpy array (0 = obstacle, 1 = free space).
    """
    # 转成 numpy 数组
    start = np.array(start, dtype=np.float64)
    goal = np.array(goal, dtype=np.float64)

    start = adjust_if_in_obstacle(start, obstacles)
    goal = adjust_if_in_obstacle(goal, obstacles)

    # 碰撞检测函数
    def is_collision_free(p1, p2, obs):
        p1 = np.array(p1, dtype=np.float64)
        p2 = np.array(p2, dtype=np.float64)
        direction = p2 - p1
        distance = np.linalg.norm(direction)
        if distance < 1e-6:
            return True
        direction /= distance
        num_samples = int(distance * 2) + 2
        for i in range(num_samples):
            point = p1 + (i / (num_samples - 1)) * (p2 - p1)
            r = int(np.round(point[0]))
            c = int(np.round(point[1]))
            if r < 0 or r >= obs.shape[0] or c < 0 or c >= obs.shape[1] or obs[r, c] == 0:
                return False
        return True

    # 参数
    max_iter = 4000
    step_size = 10.0
    goal_sample_prob = 0.2
    goal_threshold = 100.0
    rewire_radius = 50.0
    height, width = obstacles.shape
    vis_interval = 100

    free_points = np.argwhere(obstacles == 1)

    # 初始化两棵树
    tree_start = [start.copy()]
    tree_goal = [goal.copy()]
    parents_start = [-1]
    parents_goal = [-1]
    costs_start = [0.0]
    costs_goal = [0.0]

    # 可视化底图
    if vis:
        obs_map_vis = (obstacles[:, :, None] * 255).astype(np.uint8)
        obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])
        obs_map_vis = cv2.circle(obs_map_vis, (int(start[1]), int(start[0])), 3, (255, 0, 0), -1)
        obs_map_vis = cv2.circle(obs_map_vis, (int(goal[1]), int(goal[0])), 3, (0, 0, 255), -1)

    start_time = time.perf_counter()

    best_path = None
    best_cost = np.inf
    swap = False

    def get_path(tree, parents, end_idx):
        path = []
        cur = end_idx
        while cur != -1:
            path.append(tree[cur].tolist())
            cur = parents[cur]
        path.reverse()
        return path

    for i in range(max_iter):
        # 采样
        if np.random.rand() < goal_sample_prob:
            sample = tree_goal[0] if not swap else tree_start[0]
        else:
            idx = np.random.randint(len(free_points))
            sample = free_points[idx].astype(np.float64)

        r, c = int(sample[0]), int(sample[1])
        if r < 0 or r >= height or c < 0 or c >= width or obstacles[r, c] == 0:
            continue

        # Determine trees based on swap
        if swap:
            tree1, parents1, costs1 = tree_goal, parents_goal, costs_goal
            tree2, parents2, costs2 = tree_start, parents_start, costs_start
        else:
            tree1, parents1, costs1 = tree_start, parents_start, costs_start
            tree2, parents2, costs2 = tree_goal, parents_goal, costs_goal

        # Extend tree1 towards sample
        tree_array1 = np.stack(tree1)
        dists1 = cdist(sample.reshape(1, 2), tree_array1)[0]
        nearest_idx1 = np.argmin(dists1)
        nearest1 = tree1[nearest_idx1]

        dir_vec1 = sample - nearest1
        dist1 = np.linalg.norm(dir_vec1)
        if dist1 < 1e-6:
            swap = not swap
            continue
        new_pos1 = nearest1 + (dir_vec1 / dist1) * min(step_size, dist1)

        if not is_collision_free(nearest1, new_pos1, obstacles):
            swap = not swap
            continue

        # Choose best parent
        dists_to_new1 = cdist(new_pos1.reshape(1, 2), tree_array1)[0]
        neighbor_idx1 = np.where(dists_to_new1 < rewire_radius)[0]
        min_cost1 = np.inf
        best_parent1 = None
        for ni in neighbor_idx1:
            if is_collision_free(tree1[ni], new_pos1, obstacles):
                cost = costs1[ni] + np.linalg.norm(tree1[ni] - new_pos1)
                if cost < min_cost1:
                    min_cost1 = cost
                    best_parent1 = ni
        if best_parent1 is None:
            swap = not swap
            continue

        # Add to tree1
        tree1.append(new_pos1)
        parents1.append(best_parent1)
        costs1.append(min_cost1)

        # Rewire tree1
        for ni in neighbor_idx1:
            if ni == best_parent1:
                continue
            if is_collision_free(new_pos1, tree1[ni], obstacles):
                new_cost = min_cost1 + np.linalg.norm(tree1[ni] - new_pos1)
                if new_cost < costs1[ni]:
                    parents1[ni] = len(tree1) - 1
                    costs1[ni] = new_cost

        # Extend tree2 towards new_pos1
        tree_array2 = np.stack(tree2)
        dists2 = cdist(new_pos1.reshape(1, 2), tree_array2)[0]
        nearest_idx2 = np.argmin(dists2)
        nearest2 = tree2[nearest_idx2]

        dir_vec2 = new_pos1 - nearest2
        dist2 = np.linalg.norm(dir_vec2)
        if dist2 < 1e-6:
            swap = not swap
            continue
        new_pos2 = nearest2 + (dir_vec2 / dist2) * min(step_size, dist2)

        if not is_collision_free(nearest2, new_pos2, obstacles):
            swap = not swap
            continue

        # Choose best parent
        dists_to_new2 = cdist(new_pos2.reshape(1, 2), tree_array2)[0]
        neighbor_idx2 = np.where(dists_to_new2 < rewire_radius)[0]
        min_cost2 = np.inf
        best_parent2 = None
        for ni in neighbor_idx2:
            if is_collision_free(tree2[ni], new_pos2, obstacles):
                cost = costs2[ni] + np.linalg.norm(tree2[ni] - new_pos2)
                if cost < min_cost2:
                    min_cost2 = cost
                    best_parent2 = ni
        if best_parent2 is None:
            swap = not swap
            continue

        # Add to tree2
        tree2.append(new_pos2)
        parents2.append(best_parent2)
        costs2.append(min_cost2)

        # Rewire tree2
        for ni in neighbor_idx2:
            if ni == best_parent2:
                continue
            if is_collision_free(new_pos2, tree2[ni], obstacles):
                new_cost = min_cost2 + np.linalg.norm(tree2[ni] - new_pos2)
                if new_cost < costs2[ni]:
                    parents2[ni] = len(tree2) - 1
                    costs2[ni] = new_cost

        # Check if connected
        dist_connect = np.linalg.norm(new_pos1 - new_pos2)
        if dist_connect < goal_threshold and is_collision_free(new_pos1, new_pos2, obstacles):
            if not swap:
                connect_start = len(tree_start) - 1  # new_pos1
                connect_goal = len(tree_goal) - 1  # new_pos2
            else:
                connect_start = len(tree_start) - 1  # new_pos2
                connect_goal = len(tree_goal) - 1  # new_pos1

            total_cost = costs_start[connect_start] + costs_goal[connect_goal] + dist_connect

            if total_cost < best_cost:
                # Build path
                path_start = get_path(tree_start, parents_start, connect_start)
                path_goal = get_path(tree_goal, parents_goal, connect_goal)
                path_goal_rev = path_goal[::-1]
                if np.allclose(path_start[-1], path_goal[-1], atol=1e-6):
                    temp_path = path_start + path_goal_rev[1:]
                else:
                    temp_path = path_start + path_goal_rev

                best_path = temp_path
                best_cost = total_cost
                break

        # 可视化生长
        if vis:
            logging.info(f"Visualizing trees {i + 1}...")
            temp_map = obs_map_vis.copy()
            # Draw start tree (green)
            for idx, node in enumerate(tree_start):
                node_pos = (int(node[1]), int(node[0]))
                temp_map = cv2.circle(temp_map, node_pos, 1, (0, 255, 0), -1)
                if parents_start[idx] != -1:
                    parent_pos = (int(tree_start[parents_start[idx]][1]), int(tree_start[parents_start[idx]][0]))
                    temp_map = cv2.line(temp_map, parent_pos, node_pos, (0, 255, 0), 1)
            # Draw goal tree (blue)
            for idx, node in enumerate(tree_goal):
                node_pos = (int(node[1]), int(node[0]))
                temp_map = cv2.circle(temp_map, node_pos, 1, (255, 0, 0), -1)
                if parents_goal[idx] != -1:
                    parent_pos = (int(tree_goal[parents_goal[idx]][1]), int(tree_goal[parents_goal[idx]][0]))
                    temp_map = cv2.line(temp_map, parent_pos, node_pos, (255, 0, 0), 1)
            cv2.imshow(f"Bidirectional RRT* Tree Growth : floor {floor}", temp_map)
            cv2.waitKey(1)

        swap = not swap

    # After loop
    if best_path is None:
        print("No path found after max iterations")
        return []

    path = optimize_path_with_clearance(best_path, obstacles, safe_margin=3)
    end_time = time.perf_counter()
    processing_time = end_time - start_time
    logging.info(f"'############ Bidirectional RRT* run time:':{processing_time:.3f} seconds############")

    # 可视化最终路径
    if vis:
        final_map = obs_map_vis.copy()

        # Draw start tree (green)
        for idx, node in enumerate(tree_start):
            node_pos = (int(node[1]), int(node[0]))
            final_map = cv2.circle(final_map, node_pos, 1, (0, 255, 0), -1)
            if parents_start[idx] != -1:
                parent_pos = (int(tree_start[parents_start[idx]][1]), int(tree_start[parents_start[idx]][0]))
                final_map = cv2.line(final_map, parent_pos, node_pos, (0, 255, 0), 1)

        # Draw goal tree (blue)
        for idx, node in enumerate(tree_goal):
            node_pos = (int(node[1]), int(node[0]))
            final_map = cv2.circle(final_map, node_pos, 1, (255, 0, 0), -1)
            if parents_goal[idx] != -1:
                parent_pos = (int(tree_goal[parents_goal[idx]][1]), int(tree_goal[parents_goal[idx]][0]))
                final_map = cv2.line(final_map, parent_pos, node_pos, (255, 0, 0), 1)

        # Draw final path (red)
        for i, point in enumerate(path):
            subgoal = (int(point[1]), int(point[0]))
            final_map = cv2.circle(final_map, subgoal, 3, (0, 0, 255), -1)
            if i > 0:
                last_subgoal = (int(path[i - 1][1]), int(path[i - 1][0]))
                final_map = cv2.line(final_map, last_subgoal, subgoal, (0, 0, 255), 2)

        cv2.imshow(f"Bidirectional RRT* Final Trees & Path : floor {floor}", final_map)
        cv2.waitKey(0)

    return path
