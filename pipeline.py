"""
pipeline.py

Runs all steps for one LiDAR scan:

  raw scan -> crop + remove own car -> voxel downsample
           -> local ground height -> drop ground and points above 2.5 m
           -> DBSCAN clusters -> box + vehicle rules -> tracker

and keeps the in-between results, so the drawing code can show every step.
"""

import numpy as np

import settings
from clustering import find_vehicle_candidates
from point_processing import crop_area, split_ground, voxel_downsample
from tracker import VehicleTracker


class VehicleDetector:
    def __init__(self, dataset):
        self.dataset = dataset
        self.tracker = VehicleTracker()
        self.previous_pose = None   # (time, pose) of the previous scan
        self.start_time = None

    def process(self, frame):
        raw = self.dataset.load_points(frame)
        ego_to_global = self.dataset.ego_to_global(frame)
        time = frame["timestamp"] / 1e6  # microseconds -> seconds

        # Step 1: area around us, without our own car, downsampled.
        cropped = crop_area(raw)
        points = voxel_downsample(cropped, settings.VOXEL_SIZE)

        # Step 2: ground removal.
        is_ground, height, ground_map = split_ground(points)
        # Only the vehicle body layer is clustered (see settings.py).
        keep = ~is_ground & (height < settings.BODY_LAYER_TOP)
        objects = points[keep]
        object_height = height[keep]

        # Steps 3 + 4: clusters and vehicle candidates.
        labels, clusters = find_vehicle_candidates(objects, object_height)
        candidates = [c for c in clusters if not c["reject_reason"]]

        # Step 5: tracking.
        tracks = self.tracker.update(candidates, ego_to_global, time)

        # Our own speed from the ego poses (only for the display): distance
        # driven since the previous scan divided by the time between them.
        if self.previous_pose is None:
            ego_speed = 0.0
        else:
            previous_time, previous_pose = self.previous_pose
            moved = np.linalg.norm(ego_to_global[:2, 3] - previous_pose[:2, 3])
            ego_speed = moved / max(time - previous_time, 1e-3)
        self.previous_pose = (time, ego_to_global)
        if self.start_time is None:
            self.start_time = time

        return {
            "frame": frame,
            "scene_time": time - self.start_time,
            "ego_speed": ego_speed,
            "time": time,
            "ego_to_global": ego_to_global,
            "raw": raw,
            "points": points,          # downsampled
            "is_ground": is_ground,
            "height": height,          # above local ground, per point
            "ground_map": ground_map,
            "objects": objects,        # non-ground points that were clustered
            "labels": labels,          # cluster label per object point
            "clusters": clusters,
            "tracks": tracks,
        }
