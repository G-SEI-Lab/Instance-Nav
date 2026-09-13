# Adapted from VLMaps (https://github.com/vlmaps/vlmaps).
# Copyright (c) 2023 Tom-Huang. MIT license; see THIRD_PARTY_LICENSES/VLMaps-LICENSE.

import os
import cv2
import numpy as np
import openai
import httpx
from instance_nav.utils.clip_utils import get_text_feats, multiple_templates

def find_similar_category_id_deprecate(class_name, classes_list):
    """
    Return the id of the most similar name to class_name in classes_list
    """
    if class_name in classes_list:
        return classes_list.index(class_name)
    import openai

    openai_key = os.environ["OPENAI_KEY"]
    openai.api_key = openai_key
    classes_list_str = ",".join(classes_list)
    question = f"""
    Q: What is television most relevant to among tv_monitor,plant,chair. A:tv_monitor\n
    Q: What is drawer most relevant to among tv_monitor,chest_of_drawers,chair. A:chest_of_drawers\n
    Q: What is {class_name} most relevant to among {classes_list_str}. A:"""
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=question,
        max_tokens=64,
        temperature=0.0,
        stop="\n",
    )
    result = response["choices"][0]["text"].strip()
    print(f"Similar category of {class_name} is {result}")
    return classes_list.index(result)

def find_similar_category_id(class_name, classes_list):
    if class_name in classes_list:
        return classes_list.index(class_name)
    import openai

    openai_key = os.environ["DASHSCOPE_API_KEY"]
    openai.api_key = openai_key
    classes_list_str = ",".join(classes_list)
    client = openai.OpenAI(api_key=openai_key, base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', http_client=httpx.Client(trust_env=False))
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {
                "role": "user",
                "content": "You are a helpful assistant that can answer questions about objects and their categories.You can only answer one most relevant word."
            },
            {
                "role": "assistant",
                "content": "OK."
            },
            {
                "role": "user",
                "content": "What is television most relevant to among tv_monitor,plant,chair",
            },
            {
                "role": "assistant",
                "content": "tv_monitor"
            },
            {
                "role": "user",
                "content": "What is drawer most relevant to among tv_monitor,chest_of_drawers,chair"
            },
            {
                "role": "assistant",
                "content": "chest_of_drawer"
            },
            {
                "role": "user",
                "content": f"What is {class_name} most relevant to among {classes_list_str}"
            }
        ],
        max_tokens=300,
    )

    text = response.choices[0].message.content
    print(text)
    return classes_list.index(text)


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


def get_lseg_score(
    clip_model,
    landmarks: list,
    lseg_map: np.array,
    clip_feat_dim: int,
    use_multiple_templates: bool = False,
    avg_mode: int = 0,
    add_other=True,
):
    """
    Inputs:
        landmarks: a list of strings that describe the landmarks
        lseg_map: a numpy array with shape (h, w, clip_dim)
        avg_mode: this is for multiple template. 0 for averaging features, 1 for averaging scores
    """
    landmarks_other = landmarks
    if add_other and landmarks_other[-1] != "other":
        landmarks_other = landmarks + ["other"]

    if use_multiple_templates:
        mul_tmp = multiple_templates.copy()
        multi_temp_landmarks_other = [x.format(lm) for lm in landmarks_other for x in mul_tmp]
        text_feats = get_text_feats(multi_temp_landmarks_other, clip_model, clip_feat_dim)

        # average the features
        if avg_mode == 0:
            text_feats = text_feats.reshape((-1, len(mul_tmp), text_feats.shape[-1]))
            text_feats = np.mean(text_feats, axis=1)

        map_feats = lseg_map.reshape((-1, lseg_map.shape[-1]))

        scores_list = map_feats @ text_feats.T

        # average the features
        if avg_mode == 1:
            scores_list = scores_list.reshape((-1, len(landmarks_other), len(mul_tmp)))
            scores_list = np.mean(scores_list, axis=2)
    else:
        text_feats = get_text_feats(landmarks_other, clip_model, clip_feat_dim)

        map_feats = lseg_map.reshape((-1, lseg_map.shape[-1]))

        scores_list = map_feats @ text_feats.T

    return scores_list


def segment_lseg_map(
    clip_model,
    landmarks: list,
    lseg_map: np.array,
    clip_feat_dim: int,
    use_multiple_templates: bool = False,
    avg_mode: int = 0,
    add_other=True,
):
    """
    Inputs:
        landmarks: a list of strings that describe the landmarks
        lseg_map: a numpy array with shape (h, w, clip_dim)
        avg_mode: this is for multiple template. 0 for averaging features, 1 for averaging scores
    """
    scores_list = get_lseg_score(
        clip_model, landmarks, lseg_map, clip_feat_dim, use_multiple_templates, avg_mode, add_other
    )

    empty_mask = np.sum(lseg_map, axis=2) < 1e-6
    predicts = np.argmax(scores_list, axis=1)
    predicts = predicts.reshape((lseg_map.shape[0], lseg_map.shape[1]))
    predicts[empty_mask] = -1

    return predicts


def get_dynamic_obstacles_map_3d(
    clip_model,
    obstacles_cropped,
    potential_obstacle_classes,
    obstacle_classes,
    grid_feat,
    grid_pos,
    rmin,
    cmin,
    clip_feat_dim,
    use_multiple_templates=True,
    avg_mode=0,
    h_min=0.0,
    h_max=1.5,
    cs=0.05,
    vis=False,
):
    # 获取所有障碍物（被占据区域）的掩码
    all_obstacles_mask = obstacles_cropped == 0

    # 使用CLIP模型获取潜在障碍物类别的分割得分
    scores_mat = get_lseg_score(
        clip_model,
        potential_obstacle_classes,
        grid_feat,
        clip_feat_dim,
        use_multiple_templates=use_multiple_templates,
        avg_mode=avg_mode,
    )

    # 根据得分矩阵预测每个像素的类别
    predict = np.argmax(scores_mat, axis=1)

    # 初始化一个空列表，用于存储目标障碍物类别的索引
    obs_inds = []

    # 遍历所有目标障碍物类别，找到它们在潜在障碍物类别中的索引
    for obs_name in obstacle_classes:
        for i, po_obs_name in enumerate(potential_obstacle_classes):
            if obs_name == po_obs_name:
                obs_inds.append(i)

    # 打印找到的障碍物类别索引
    print("obs_inds: ", obs_inds)

    # 初始化一个与predict形状相同的全零布尔数组
    pts_mask = np.zeros_like(predict, dtype=bool)

    # 遍历找到的障碍物类别索引，设置pts_mask中对应类别的位置为True
    for id in obs_inds:
        tmp = predict == id
        pts_mask = np.logical_or(pts_mask, tmp)

    # 初始化一个与obstacles_cropped形状相同的全零布尔数组
    # new_obstacles = obstacles_segment_map != 1
    new_obstacles = np.zeros_like(obstacles_cropped, dtype=bool)

    # 获取pts_mask为True的位置对应的grid_pos中的点
    obs_pts = grid_pos[pts_mask]
    obs_heights = obs_pts[:, 2] * cs  # 将栅格化高度转换为实际高度
    height_mask = (obs_heights >= h_min) & (obs_heights <= h_max)
    # 生成第一个掩码，确保点的坐标在有效范围内
    mask1 = np.logical_and(obs_pts[:, 0] - rmin >= 0, obs_pts[:, 1] - cmin >= 0)

    # 生成第二个掩码，确保点的坐标不超过new_obstacles的形状
    mask2 = np.logical_and(obs_pts[:, 0] - rmin < new_obstacles.shape[0], obs_pts[:, 1] - cmin < new_obstacles.shape[1])

    # 将两个掩码进行逻辑与操作，得到最终的掩码
    mask = np.logical_and(np.logical_and(mask1, mask2), height_mask)

    # 在new_obstacles中标记为动态障碍物
    new_obstacles[obs_pts[mask, 0] - rmin, obs_pts[mask, 1] - cmin] = 1

    # 将所有pts_mask为True的位置都标记为动态障碍物（可能存在重复标记）
    valid_obs_pts = obs_pts[mask]
    new_obstacles[valid_obs_pts[:, 0] - rmin, valid_obs_pts[:, 1] - cmin] = 1

    # 确保动态障碍物位于原始障碍物区域内
    new_obstacles = np.logical_and(new_obstacles, all_obstacles_mask)

    # 反转new_obstacles，因为原代码中0表示障碍物，1表示非障碍物，而我们需要得到的是动态障碍物的掩码
    new_obstacles = np.logical_not(new_obstacles)

    # 如果设置了可视化标志，则显示新的障碍物地图
    if vis:
        cv2.imshow("new obstacles_cropped", (new_obstacles * 255).astype(np.uint8))
        cv2.waitKey()

    # 返回新的动态障碍物地图
    return new_obstacles

def get_main_colors(obj_colors, max_clusters=4):
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    """
    根据点的颜色分布，自动决定簇数并提取主色，使用DBSCAN方法
    """
    obj_colors = np.array(obj_colors)  # 确保是numpy数组
    n_points = len(obj_colors)
    if n_points < 4:
        # 点太少，直接返回平均颜色
        avg_color = np.mean(obj_colors, axis=0).astype(int).tolist()
        return [{"color": avg_color, "proportion": 1.0}]

    # 归一化颜色到[0,1]以便DBSCAN的eps参数更直观
    obj_colors_norm = obj_colors / 255.0

    # 尝试不同的eps值，选择最优的（簇数在2到max_clusters之间）
    eps_values = np.linspace(0.01, 0.3, 10)  # 从紧簇到松簇
    min_samples = max(3, n_points // 20)  # 动态最小样本，少なくとも3
    best_eps = None
    best_score = -1
    best_labels = None

    max_possible_clusters = min(max_clusters, n_points // min_samples)
    for eps in eps_values:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(obj_colors_norm)
        labels = db.labels_
        core_mask = labels != -1
        unique_labels = set(labels[core_mask])
        n_clusters = len(unique_labels)

        if n_clusters < 2 or n_clusters > max_possible_clusters:
            continue

        n_core_points = np.sum(core_mask)
        if n_core_points < 2:
            continue

        # 只用非噪声点计算silhouette分数
        score = silhouette_score(obj_colors_norm[core_mask], labels[core_mask])
        if score > best_score:
            best_score = score
            best_eps = eps
            best_labels = labels

    # 如果没有找到合适的模型，退化为1簇平均色
    if best_labels is None:
        avg_color = np.mean(obj_colors, axis=0).astype(int).tolist()
        return [{"color": avg_color, "proportion": 1.0}]

    # 统计簇分布（忽略噪声）
    core_mask = best_labels != -1
    unique_labels, counts = np.unique(best_labels[core_mask], return_counts=True)
    total = n_points  # 比例基于总点数，包括噪声（噪声不计入任何簇）

    # 计算每个簇的中心颜色
    cluster_centers = []
    for label in unique_labels:
        cluster_points = obj_colors[best_labels == label]
        center = np.mean(cluster_points, axis=0).astype(int).tolist()
        cluster_centers.append(center)

    # 排序按比例降序
    sorted_indices = np.argsort(counts)[::-1]
    main_colors = []
    for idx in sorted_indices:
        color = cluster_centers[idx]
        proportion = counts[idx] / total
        main_colors.append({"color": color, "proportion": proportion})

    return main_colors
