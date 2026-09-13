# Adapted from VLMaps (https://github.com/vlmaps/vlmaps).
# Copyright (c) 2023 Tom-Huang. MIT license; see THIRD_PARTY_LICENSES/VLMaps-LICENSE.

import numpy as np

def pool_3d_label_to_2d(
    mask_3d: np.ndarray, grid_pos: np.ndarray, gs: int
) -> np.ndarray:
    mask_2d = np.zeros((gs, gs), dtype=bool)
    for i, pos in enumerate(grid_pos):
        row, col, h = pos
        mask_2d[row, col] = mask_3d[i] or mask_2d[row, col]

    return mask_2d


def pool_filter_by_height_3d_label_to_2d(
    mask_3d: np.ndarray,
    grid_pos: np.ndarray,
    gs: int,
    cs: float,
    h_min: float = 0.0,
    h_max=1.5,
) -> np.ndarray:
    mask_2d = np.zeros((gs, gs), dtype=bool)
    h_min_grid = int(h_min / cs)
    h_max_grid = int(h_max / cs)
    for i, pos in enumerate(grid_pos):
        row, col, h = pos
        if h < h_min_grid or h > h_max_grid:
            continue
        mask_2d[row, col] = mask_3d[i] or mask_2d[row, col]

    return mask_2d
