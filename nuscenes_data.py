"""
nuscenes_data.py

A small reader for the nuScenes mini dataset. I wrote my own instead of
installing the official nuscenes-devkit because I only need a few things:

  * the list of LiDAR scans of one scene, in time order
  * the points of a scan
  * where the car was at that moment (ego pose)
  * the front camera picture closest in time (only for display)

How nuScenes stores things (short version):
  - every file (LiDAR scan or camera picture) has a "sample_data" record
  - "sample_data" points to a "calibrated_sensor" record (where the sensor
    sits on the car) and an "ego_pose" record (where the car is in the world)
  - rotations are quaternions written as [w, x, y, z]

Coordinate frames used in this project:
  sensor frame : as recorded by the LiDAR
  ego frame    : our car. x = forward, y = left, z = up, origin on the ground
                 under the rear axle. All detection happens in this frame.
  global frame : fixed world map frame. Used for tracking, so that our own
                 motion does not look like the other cars moving.
"""

import json
import os

import numpy as np


def quaternion_to_matrix(q):
    """3x3 rotation matrix from a quaternion [w, x, y, z]."""
    w, x, y, z = q
    # Standard formula for a unit quaternion. Written out by hand so the
    # project doesn't need an extra library just for this one function.
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def make_transform(translation, rotation):
    """4x4 matrix that first rotates, then moves (a "pose")."""
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_to_matrix(rotation)
    matrix[:3, 3] = translation
    return matrix


def apply_transform(matrix, points_xyz):
    """Applies a 4x4 transform to an (N, 3) array of points."""
    # Each point is a row, so the rotation is multiplied from the right
    # (transposed) and then the translation is added to every row.
    return points_xyz @ matrix[:3, :3].T + matrix[:3, 3]


class NuScenesMini:
    def __init__(self, data_root):
        self.data_root = data_root
        table_folder = os.path.join(data_root, "v1.0-mini")
        if not os.path.isdir(table_folder):
            raise FileNotFoundError(
                f"Could not find {table_folder}. Run get_nuscenes_mini.py first "
                f"(see README.md).")

        def load(name):
            with open(os.path.join(table_folder, name + ".json")) as file:
                return json.load(file)

        # Tables that are looked up by token become dictionaries
        # (token -> record), so a lookup is instant instead of a search.
        self.scenes = load("scene")
        self.samples = {r["token"]: r for r in load("sample")}
        self.sample_data = load("sample_data")
        self.ego_poses = {r["token"]: r for r in load("ego_pose")}
        self.calibrations = {r["token"]: r for r in load("calibrated_sensor")}
        sensors = {r["token"]: r["channel"] for r in load("sensor")}

        # Remember the sensor name (e.g. "LIDAR_TOP") on every record.
        for record in self.sample_data:
            calibration = self.calibrations[record["calibrated_sensor_token"]]
            record["channel"] = sensors[calibration["sensor_token"]]

    def scene_names(self):
        return [scene["name"] for scene in self.scenes]

    def scene(self, name):
        for scene in self.scenes:
            if scene["name"] == name:
                return scene
        raise ValueError(f"Unknown scene {name}. Available: {self.scene_names()}")

    def frames(self, scene_name, channel="LIDAR_TOP"):
        """
        All recordings of one sensor in a scene, sorted by time.
        For LIDAR_TOP that is about 20 scans per second: the 2 Hz key frames
        ("samples", the annotated ones) plus the "sweeps" in between.
        """
        scene_token = self.scene(scene_name)["token"]
        records = [
            r for r in self.sample_data
            if r["channel"] == channel
            and self.samples[r["sample_token"]]["scene_token"] == scene_token
        ]
        return sorted(records, key=lambda r: r["timestamp"])

    def load_points(self, frame):
        """
        Points of one LiDAR scan in the EGO frame, as an (N, 4) array:
        x, y, z, intensity. The file stores 5 floats per point
        (x, y, z, intensity, laser ring number).
        """
        path = os.path.join(self.data_root, frame["filename"])
        raw = np.fromfile(path, dtype=np.float32).reshape(-1, 5)

        # The points are measured from the LiDAR on the roof. The calibration
        # says where the LiDAR sits on the car, so this moves every point
        # into the ego frame (origin on the ground under the rear axle).
        calibration = self.calibrations[frame["calibrated_sensor_token"]]
        sensor_to_ego = make_transform(calibration["translation"], calibration["rotation"])

        xyz = apply_transform(sensor_to_ego, raw[:, :3].astype(np.float64))
        return np.column_stack([xyz, raw[:, 3]])

    def ego_to_global(self, frame):
        """4x4 pose of our car in the world when this recording was made."""
        pose = self.ego_poses[frame["ego_pose_token"]]
        return make_transform(pose["translation"], pose["rotation"])

    def image_path(self, frame):
        return os.path.join(self.data_root, frame["filename"])

    def camera_matrices(self, camera_frame):
        """Intrinsic matrix and ego->camera transform of a camera recording."""
        calibration = self.calibrations[camera_frame["calibrated_sensor_token"]]
        camera_to_ego = make_transform(calibration["translation"], calibration["rotation"])
        return np.array(calibration["camera_intrinsic"]), np.linalg.inv(camera_to_ego)


def closest_in_time(frames, timestamp):
    """The recording from `frames` whose timestamp is closest to `timestamp`."""
    times = np.array([f["timestamp"] for f in frames])
    return frames[int(np.argmin(np.abs(times - timestamp)))]
