"""
run_detection.py

Main script. Runs the vehicle detection on one nuScenes scene and writes
an annotated video (bird's-eye view + front camera + vehicle list).

Examples:
    python run_detection.py scene-0103
    python run_detection.py scene-0061 --diagnostic
    python run_detection.py scene-0655 --start 5 --end 12 --output output/parking.mp4
    python run_detection.py --list

Output goes to output/<scene>_vehicles.mp4 and, with --diagnostic, also
output/<scene>_diagnostic.mp4 (four pipeline steps side by side).
"""

import argparse
import os
import time
from collections import Counter

import cv2

import settings
from drawing import diagnostic_grid, render_frame
from nuscenes_data import NuScenesMini, closest_in_time
from pipeline import VehicleDetector


def parse_arguments():
    parser = argparse.ArgumentParser(description="Find vehicles in nuScenes LiDAR scans")
    parser.add_argument("scene", nargs="?", help="scene name, e.g. scene-0103")
    parser.add_argument("--list", action="store_true", help="list the scenes and exit")
    parser.add_argument("--data", default=settings.DATA_ROOT, help="nuScenes mini folder")
    parser.add_argument("--output", help="output video (default: output/<scene>_vehicles.mp4)")
    parser.add_argument("--diagnostic", action="store_true",
                        help="also write a video with the pipeline steps")
    parser.add_argument("--start", type=float, default=0, help="start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="end time in seconds")
    return parser.parse_args()


def main():
    args = parse_arguments()
    dataset = NuScenesMini(args.data)

    if args.list or not args.scene:
        for scene in dataset.scenes:
            print(f"{scene['name']}: {scene['description']}")
        return

    # [::FRAME_STEP] keeps every 2nd scan (10 per second instead of 20).
    lidar_frames = dataset.frames(args.scene)[::settings.FRAME_STEP]
    camera_frames = dataset.frames(args.scene, "CAM_FRONT")
    first_time = lidar_frames[0]["timestamp"] / 1e6
    fps = 20 / settings.FRAME_STEP   # the LiDAR spins 20 times per second

    os.makedirs("output", exist_ok=True)
    output_path = args.output or f"output/{args.scene}_vehicles.mp4"
    writer, diagnostic_writer = None, None

    detector = VehicleDetector(dataset)
    statuses = Counter()
    started = time.time()
    n_frames = 0

    for frame in lidar_frames:
        scene_time = frame["timestamp"] / 1e6 - first_time
        if args.end is not None and scene_time > args.end:
            break
        # Frames before --start are still processed (without output), so
        # the tracker already knows the vehicles when the clip begins.
        result = detector.process(frame)
        if scene_time < args.start:
            continue

        camera_frame = closest_in_time(camera_frames, frame["timestamp"])
        picture = render_frame(dataset, result, camera_frame, args.scene)
        # The video writer is made at the first frame, because only then
        # do we know the picture size.
        if writer is None:
            height, width = picture.shape[:2]
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (width, height))
        writer.write(picture)

        if args.diagnostic:
            grid = diagnostic_grid(result)
            if diagnostic_writer is None:
                height, width = grid.shape[:2]
                diagnostic_path = output_path.replace("_vehicles.mp4", "") + "_diagnostic.mp4"
                diagnostic_writer = cv2.VideoWriter(diagnostic_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                                    fps, (width, height))
            diagnostic_writer.write(grid)

        # Count confident and uncertain tracks for the summary at the end.
        statuses.update(t.status for t in result["tracks"] if t.status != "held")
        n_frames += 1

    if writer is None:
        print("No frames in the chosen time range.")
        return
    writer.release()
    print(f"Saved {output_path}")
    if diagnostic_writer is not None:
        diagnostic_writer.release()
        print(f"Saved {diagnostic_path}")

    seconds = time.time() - started
    print(f"{n_frames} scans in {seconds:.0f} s ({n_frames / seconds:.1f} scans per second)")
    print(f"per scan on average: {statuses['vehicle'] / n_frames:.1f} confident vehicles, "
          f"{statuses['uncertain'] / n_frames:.1f} uncertain candidates")


if __name__ == "__main__":
    main()
