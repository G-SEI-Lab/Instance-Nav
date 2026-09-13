# Instance-Nav

## Navigation code

This repository now includes the single-floor and multi-floor Instance-Nav
navigation/evaluation code. It **loads prebuilt semantic maps**; map creation
scripts, map builders, model weights, raw RGB-D captures, and experiment logs
are not part of this release.

The evaluation entry points are:

- `application/evaluation/evaluate_object_goal_navigation_color_v2.py` —
  single-floor fine-grained object navigation;
- `application/evaluation/evaluate_object_goal_navigation_color_v3.py` —
  multi-floor navigation with staircase routing;
- `application/evaluation/evaluate_object_goal_navigation_color_v2_efficiency_count.py` —
  single-floor navigation with timing measurements.

The `instance_nav/` package contains task evaluation, robot control, route
planning, prebuilt-map reading, and utilities needed by these entry points.
The prebuilt-map file format still uses the legacy name `vlmap/vlmaps.h5df`.
This is an input artifact name, not a map builder included in this release.

### Environment and data layout

Use an environment with compatible Habitat-Sim/Habitat-Lab, PyTorch and OpenAI
CLIP installations, then install the Python dependencies and this package:

```bash
pip install -r requirements-navigation.txt
pip install -e .
export HABITAT_SCENE_DIR=/path/to/mp3d/scenes
export INSTANCE_NAV_MAP_DIR=/path/to/prebuilt/scene/maps
```

Each scene under `INSTANCE_NAV_MAP_DIR` must contain a prebuilt
`vlmap/vlmaps.h5df`. For single-floor evaluation it must also contain
`color_object_nav_dataset.json` from `dataset/single/<scene>/`; multi-floor
evaluation uses `color_object_3d_nav_dataset.json` from
`dataset/multi-floor/<scene>/` and a `stairs.json` from the scene assets.
The directory names and sorted scene order determine `scene_id` in
`config/object_goal_navigation_cfg.yaml`; set the desired indices before
running. The simulator scene directory and prebuilt map directory are supplied
through the environment variables above, so no workstation path is embedded.

```bash
python application/evaluation/evaluate_object_goal_navigation_color_v2.py
python application/evaluation/evaluate_object_goal_navigation_color_v3.py
```

These are research evaluation entry points, not a standalone demo. They require
the prebuilt scene maps, scene geometry, dataset JSON files, and model services
used by the original experiment. The navigation code has been statically
validated; a full Habitat run requires those external assets and services.

Parts of this code are adapted from VLMaps under the MIT license. See
`THIRD_PARTY_LICENSES/` and the attribution headers in the derived files.

## Dataset
To comprehensively evaluate the perception of fine-grained attributes and the ability to sustain long-horizon tasks, we introduce the Instance-Nav dataset, a fine-grained, description-oriented, multi-floor long-horizon navigation benchmark.
![](image/fig_1.png)
The benchmark construction involves a rigorous scene selection and task generation process. We filtered MP3D to retain 9 distinct scenes stratified by scale and object density, ranging from expansive, object-rich environments to small, sparse settings. To ensure comprehensive assessment, task allocation is calibrated to scene complexity: object-dense scenes are assigned more than ten navigation sequences (approx. 100 target objects) to stress-test attribute disambiguation, while sparse scenes contain fewer sequences for robustness evaluation. This stratified protocol rigorously examines the robustness of attribute modeling across varying environmental complexities.
