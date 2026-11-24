"""
save_pipeline_steps.py

Saves one picture per pipeline step for a chosen moment of a scene, for
the visual walkthrough. All scans before that moment are processed too,
so the tracker has its history (IDs, speeds) like in the video.

Example:
    python save_pipeline_steps.py scene-0103 --time 8
    -> screenshots/scene-0103_8s/1_raw_3d.png ... all_steps.jpg
"""

import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")  # draw into files, no window
import matplotlib.pyplot as plt

import settings
from drawing import (cluster_view, diagnostic_grid, final_view, ground_view, raw_view,
                     render_frame)
from nuscenes_data import NuScenesMini, closest_in_time
from pipeline import VehicleDetector


def save_3d_view(result, path):
    """The raw scan as a 3D scatter plot, seen from behind and above our car."""
    raw = result["raw"]
    near = raw[(abs(raw[:, 0]) < 40) & (abs(raw[:, 1]) < 40)][::3]  # every 3rd point
    figure = plt.figure(figsize=(10, 7), dpi=110)
    axes = figure.add_subplot(projection="3d")
    axes.scatter(near[:, 0], near[:, 1], near[:, 2], c=near[:, 2], cmap="jet",
                 s=0.3, vmin=-0.5, vmax=3.5)
    axes.view_init(elev=28, azim=-160)
    axes.set_xlim(-40, 40)
    axes.set_ylim(-40, 40)
    axes.set_zlim(-5, 10)
    axes.set_box_aspect((1, 1, 0.2))
    axes.set_xlabel("x forward [m]")
    axes.set_ylabel("y left [m]")
    axes.set_title(f"Raw LiDAR scan ({len(raw)} points), colour = height")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Save pictures of every pipeline step")
    parser.add_argument("scene", help="scene name, e.g. scene-0103")
    parser.add_argument("--time", type=float, default=5.0, help="moment in seconds")
    parser.add_argument("--data", default=settings.DATA_ROOT, help="nuScenes mini folder")
    args = parser.parse_args()

    dataset = NuScenesMini(args.data)
    lidar_frames = dataset.frames(args.scene)[::settings.FRAME_STEP]
    camera_frames = dataset.frames(args.scene, "CAM_FRONT")
    first_time = lidar_frames[0]["timestamp"] / 1e6

    detector = VehicleDetector(dataset)
    for frame in lidar_frames:
        result = detector.process(frame)
        if frame["timestamp"] / 1e6 - first_time >= args.time:
            break

    folder = os.path.join("screenshots", f"{args.scene}_{args.time:g}s")
    os.makedirs(folder, exist_ok=True)
    camera_frame = closest_in_time(camera_frames, frame["timestamp"])

    save_3d_view(result, os.path.join(folder, "1_raw_3d.png"))
    pictures = {
        "2_raw_birdseye.png": raw_view(result),
        "3_ground_removed.png": ground_view(result),
        "4_clusters_and_candidates.png": cluster_view(result),
        "5_tracked_vehicles.png": final_view(result),
        "6_final_frame.jpg": render_frame(dataset, result, camera_frame, args.scene),
        "all_steps.jpg": diagnostic_grid(result),
    }
    for name, picture in pictures.items():
        cv2.imwrite(os.path.join(folder, name), picture)
    print(f"Saved {len(pictures) + 1} pictures to {folder}")


if __name__ == "__main__":
    main()
