# Adapted from VLMaps (https://github.com/vlmaps/vlmaps).
# Copyright (c) 2023 Tom-Huang. MIT license; see THIRD_PARTY_LICENSES/VLMaps-LICENSE.

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Set, Tuple

def cvt_pose_vec2tf(pos_quat_vec: np.ndarray) -> np.ndarray:
    """
    pos_quat_vec: (px, py, pz, qx, qy, qz, qw)
    """
    pose_tf = np.eye(4)
    pose_tf[:3, 3] = pos_quat_vec[:3].flatten()
    rot = R.from_quat(pos_quat_vec[3:].flatten())
    pose_tf[:3, :3] = rot.as_matrix()
    # rot_change = R.from_euler('xyz', [-90, 0, -90], degrees=True)
    # rot_change = R.from_euler('xyz', [0, 90, 90], degrees=True)
    # rot = R.from_quat(pos_quat_vec[3:].flatten())
    # rot = rot * rot_change
    # pose_tf[:3, :3] = rot.as_matrix()
    return pose_tf


def base_pos2grid_id_3d(gs, cs, x_base, y_base, z_base):
    row = int(gs / 2 - int(x_base / cs))
    col = int(gs / 2 - int(y_base / cs))
    h = int(z_base / cs)
    return [row, col, h]


def base_pos2grid_id_3d_3(gs, cs, x_base, y_base, z_base):
    row = int(gs / 2 - int(x_base / cs))
    col = int(gs / 2 - int(y_base / cs))
    return [row, col]


def grid_id2base_pos_3d(row, col, height, cs, gs):
    base_x = (gs / 2 - row) * cs
    base_y = (gs / 2 - col) * cs
    base_z = height * cs
    return [base_x, base_y, base_z]


def grid_id2base_pos_3d_batch(pos_grid_np, cs, gs):
    """
    pos_grid_np: [N, 3] np.int32
    """
    base_x = (gs / 2 - pos_grid_np[:, 0]) * cs
    base_y = (gs / 2 - pos_grid_np[:, 1]) * cs
    base_z = pos_grid_np[:, 2] * cs
    return [base_x, base_y, base_z]


def base_rot_mat2theta(rot_mat: np.ndarray) -> float:
    """Convert base rotation matrix to rotation angle (rad) assuming x is forward, y is left, z is up

    Args:
        rot_mat (np.ndarray): (3,3) rotation matrix

    Returns:
        float: rotation angle
    """
    theta = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    return theta


def load_3d_map(map_path: str) -> Tuple[Set[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load 3D voxel map with features

    Args:
        map_path (str): path to save the map as an H5DF file.
    Return:
        mapped_iter_list (Set[int]): stores already processed frame's number.
        grid_feat (np.ndarray): (N, feat_dim) features of each 3D point.
        grid_pos (np.ndarray): (N, 3) each row is the (row, col, height) of an occupied cell.
        weight (np.ndarray): (N,) accumulated weight of the cell's features.
        occupied_ids (np.ndarray): (gs, gs, vh) either -1 or 1. 1 indicates
            occupation.
        grid_rgb (np.ndarray, optional): (N, 3) each row stores the rgb value
            of the cell.
        ---
        N is the total number of occupied cells in the 3D voxel map.
        gs is the grid size (number of cells on each side).
        vh is the number of cells in height direction.
    """
    with h5py.File(map_path, "r") as f:
        mapped_iter_list = f["mapped_iter_list"][:].tolist()
        grid_feat = f["grid_feat"][:]
        grid_pos = f["grid_pos"][:]
        weight = f["weight"][:]
        occupied_ids = f["occupied_ids"][:]
        grid_rgb = None
        pcd_min = None
        pcd_max = None
        if "grid_rgb" in f:
            grid_rgb = f["grid_rgb"][:]
        if "pcd_min" in f:
            pcd_min = f["pcd_min"][:]
        if "pcd_max" in f:
            pcd_max = f["pcd_max"][:]
        return mapped_iter_list, grid_feat, grid_pos, weight, occupied_ids, grid_rgb, pcd_min, pcd_max
