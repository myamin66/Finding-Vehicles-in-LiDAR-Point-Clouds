"""
get_nuscenes_mini.py

Downloads the nuScenes mini dataset (about 4 GB, no login needed) and keeps
only what this project uses:

  v1.0-mini/*.json          the tables (scenes, poses, calibration, ...)
  LIDAR_TOP files           the LiDAR scans
  CAM_FRONT files           front camera pictures (only for display)

of the chosen scenes. The archive is read as a stream and never saved, so
you only need disk space for the kept files (~300 MB per scene).

Examples:
    python get_nuscenes_mini.py                       # the scenes used in the demos
    python get_nuscenes_mini.py --scenes all          # all 10 scenes (~3 GB)
    python get_nuscenes_mini.py --archive v1.0-mini.tgz   # use an already downloaded archive

nuScenes is (c) Motional, licensed CC BY-NC-SA 4.0 (non-commercial use).
"""

import argparse
import os
import tarfile
import urllib.request

import settings
from nuscenes_data import NuScenesMini

URL = "https://www.nuscenes.org/data/v1.0-mini.tgz"
DEMO_SCENES = ["scene-0061", "scene-0103", "scene-0553", "scene-0655", "scene-1077"]
CHANNELS = ["LIDAR_TOP", "CAM_FRONT"]


def wanted_files(data_root, scene_names):
    """File names (as stored in the archive) that belong to the scenes."""
    dataset = NuScenesMini(data_root)
    if scene_names == ["all"]:
        scene_names = dataset.scene_names()
    files = set()
    for scene in scene_names:
        for channel in CHANNELS:
            files.update(frame["filename"] for frame in dataset.frames(scene, channel))
    return files


def main():
    parser = argparse.ArgumentParser(description="Download the parts of nuScenes mini we need")
    parser.add_argument("--scenes", nargs="+", default=DEMO_SCENES,
                        help="scene names, or 'all' (default: the demo scenes)")
    parser.add_argument("--archive", help="local v1.0-mini.tgz instead of downloading")
    parser.add_argument("--data", default=settings.DATA_ROOT, help="where to put the data")
    args = parser.parse_args()

    if args.archive:
        stream = open(args.archive, "rb")
    else:
        print(f"Downloading {URL} (about 4 GB, only the needed parts are kept) ...")
        stream = urllib.request.urlopen(URL)

    keep = None   # filled in once the tables have been read
    kept = 0
    # "r|gz" = read the compressed archive front to back, as a stream.
    with tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = member.name

            if name.startswith("v1.0-mini/"):
                save(archive, member, args.data)
                continue

            # The tables come before the sensor files in the archive, so at
            # the first sensor file we can work out which files we want.
            if name.startswith(("samples/", "sweeps/")):
                if keep is None:
                    keep = wanted_files(args.data, args.scenes)
                    print(f"Keeping {len(keep)} files of {', '.join(args.scenes)}")
                if name in keep:
                    save(archive, member, args.data)
                    kept += 1
                    if kept % 500 == 0:
                        print(f"  {kept} / {len(keep)} files")

    print(f"Done: {kept} sensor files in {args.data}")


def save(archive, member, data_root):
    # Never write outside data_root, whatever the archive says.
    parts = member.name.split("/")
    if ".." in parts or os.path.isabs(member.name):
        return
    path = os.path.join(data_root, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with archive.extractfile(member) as source, open(path, "wb") as target:
        target.write(source.read())


if __name__ == "__main__":
    main()
