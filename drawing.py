"""
drawing.py

Everything that turns results into pictures, drawn with OpenCV.

Bird's-eye view (BEV): looking down from above, our car in the middle,
driving direction = up. 1 metre = PIXELS_PER_METRE pixels.

Colour language of the final view:
  dark grey points   ground
  light grey points  objects that are not vehicles
  coloured points    a tracked vehicle (one colour per ID)
  solid box          confident vehicle
  dashed box         uncertain vehicle (reason written next to it)
  thin grey box      held track: not seen in this scan, predicted position
"""

import cv2
import numpy as np

import settings
from tracker import motion_text

SCALE = settings.PIXELS_PER_METRE
BEV_SIZE = int(2 * settings.AREA_HALF_SIZE * SCALE)
CENTRE = BEV_SIZE // 2

BACKGROUND = (25, 25, 25)
GROUND = (75, 75, 75)
OBJECT = (170, 170, 170)
TEXT = (235, 235, 235)
UNCERTAIN = (0, 200, 255)     # amber (OpenCV colours are B, G, R)
HELD = (140, 140, 140)

# Distinct colours for track IDs (BGR).
PALETTE = [
    (255, 128, 0), (0, 220, 0), (255, 0, 255), (255, 255, 0), (0, 128, 255),
    (180, 120, 255), (0, 255, 180), (255, 80, 80), (80, 160, 255), (200, 255, 80),
]


def track_colour(track_id):
    return PALETTE[track_id % len(PALETTE)]


def to_pixels(xy):
    """Ego-frame (x forward, y left) -> image pixels (column, row)."""
    xy = np.atleast_2d(xy)
    # Forward (x) is up in the picture, so it lowers the row number.
    # Left (y) is left in the picture, so it lowers the column number.
    column = CENTRE - xy[:, 1] * SCALE
    row = CENTRE - xy[:, 0] * SCALE
    return np.column_stack([column, row]).astype(np.int32)


def put_text(image, text, position, colour=TEXT, size=0.45, thickness=1):
    """Text with a dark outline, readable on any background."""
    x, y = int(position[0]), int(position[1])
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0),
                thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, colour,
                thickness, cv2.LINE_AA)


def empty_bev():
    """Background with distance rings every 10 m and our car."""
    image = np.full((BEV_SIZE, BEV_SIZE, 3), BACKGROUND, np.uint8)
    for radius in range(10, int(settings.AREA_HALF_SIZE) + 1, 10):
        cv2.circle(image, (CENTRE, CENTRE), radius * SCALE, (55, 55, 55), 1, cv2.LINE_AA)
        put_text(image, f"{radius} m", (CENTRE + 4, CENTRE - radius * SCALE + 14),
                 (110, 110, 110), 0.4)
    draw_box(image, np.array([1.0, 0.0]), 4.6, 1.8, 0.0, (255, 255, 255), "solid", 2)
    return image


def draw_points(image, xy, colour, radius=1):
    """
    Draws points in one colour (or one colour per point, an (N, 3) array).
    radius 1 = single pixels (fast, for whole scans), bigger = small dots.
    """
    pixels = to_pixels(xy)
    on_image = ((pixels[:, 0] >= 0) & (pixels[:, 0] < BEV_SIZE)
                & (pixels[:, 1] >= 0) & (pixels[:, 1] < BEV_SIZE))
    pixels = pixels[on_image]
    if radius <= 1:
        # Writing straight into the pixel array is much faster than
        # calling cv2.circle for tens of thousands of points.
        colours = colour if np.ndim(colour) == 1 else np.asarray(colour)[on_image]
        image[pixels[:, 1], pixels[:, 0]] = colours
    else:
        for column, row in pixels:
            cv2.circle(image, (int(column), int(row)), radius, colour, -1)


def box_corners(centre, length, width, yaw):
    """The 4 corners (ego frame) of a box seen from above."""
    forward = np.array([np.cos(yaw), np.sin(yaw)]) * length / 2
    side = np.array([-np.sin(yaw), np.cos(yaw)]) * width / 2
    return np.array([centre + forward + side, centre + forward - side,
                     centre - forward - side, centre - forward + side])


def dashed_line(image, start, end, colour, thickness, dash=6):
    # OpenCV has no dashed lines: cut the line into pieces of about `dash`
    # pixels and draw every second piece.
    start, end = np.array(start, float), np.array(end, float)
    length = np.linalg.norm(end - start)
    n_pieces = max(1, int(length / dash))
    for i in range(0, n_pieces, 2):
        a = start + (end - start) * i / n_pieces
        b = start + (end - start) * min(i + 1, n_pieces) / n_pieces
        cv2.line(image, tuple(a.astype(int)), tuple(b.astype(int)), colour, thickness, cv2.LINE_AA)


def draw_box(image, centre, length, width, yaw, colour, style="solid", thickness=2):
    corners = to_pixels(box_corners(centre, length, width, yaw))
    for i in range(4):
        start, end = corners[i], corners[(i + 1) % 4]
        if style == "solid":
            cv2.line(image, tuple(start), tuple(end), colour, thickness, cv2.LINE_AA)
        else:
            dashed_line(image, start, end, colour, thickness)


# Pipeline stage views

def height_colour(z):
    """Blue (low) -> red (high) colours for raw points, as BGR."""
    level = np.clip((z + 0.5) / 4.0 * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(level.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)


def raw_view(result):
    """Stage 1: the raw scan coloured by height."""
    image = empty_bev()
    raw = result["raw"]
    draw_points(image, raw[:, :2], height_colour(raw[:, 2]))
    return image


def ground_view(result):
    """Stage 2: downsampled points, ground vs. objects."""
    image = empty_bev()
    points = result["points"]
    draw_points(image, points[result["is_ground"], :2], (40, 85, 150))  # brown (B, G, R)
    draw_points(image, points[~result["is_ground"], :2], (220, 220, 220))
    return image


def cluster_view(result):
    """Stage 3 + 4: every cluster in its own colour, boxes of candidates."""
    image = empty_bev()
    objects, labels = result["objects"], result["labels"]
    draw_points(image, objects[labels < 0, :2], (70, 70, 70))
    for cluster in result["clusters"]:
        # Cluster numbers change from scan to scan, so the colour only
        # has to make neighbouring clusters look different.
        colour = track_colour(int(cluster["members"][0]))
        xy = objects[cluster["members"], :2]
        if cluster["reject_reason"]:
            draw_points(image, xy, (120, 120, 120))
        else:
            draw_points(image, xy, colour, 2)
            draw_box(image, cluster["centre"], cluster["length"], cluster["width"],
                     cluster["yaw"], colour, "solid", 1)
    return image


def final_view(result):
    """Stage 5: tracked vehicles with IDs, confidence and motion."""
    image = empty_bev()
    points, objects = result["points"], result["objects"]
    draw_points(image, points[result["is_ground"], :2], GROUND)
    draw_points(image, objects[:, :2], OBJECT)

    # Uncertain first, so confident vehicles are drawn on top of them.
    order = {"held": 0, "uncertain": 1, "vehicle": 2}
    for track in sorted(shown_tracks(result["tracks"]), key=lambda t: order[t.status]):
        draw_track(image, track, objects)
    return image


def shown_tracks(tracks):
    """
    Tracks worth drawing: vehicles, uncertain ones that are not too weak,
    and held tracks that had been confirmed before. The rest (very weak
    candidates, held noise) would only clutter the picture.
    """
    shown = []
    for track in tracks:
        if track.status == "vehicle":
            shown.append(track)
        elif track.status == "uncertain" and track.confidence >= settings.MIN_SCORE_TO_SHOW:
            shown.append(track)
        elif track.status == "held" and track.was_vehicle:
            shown.append(track)
    return shown


def draw_track(image, track, objects):
    cluster = track.cluster
    colour = track_colour(track.id)

    if track.status == "held":
        # Not seen in this scan: thin grey dashed box at the predicted
        # position, so it can't be mistaken for a real detection.
        draw_box(image, track.ego_position, cluster["length"], cluster["width"],
                 cluster["yaw"], HELD, "dashed", 1)
        label_at = to_pixels(track.ego_position)[0]
        put_text(image, f"#{track.id} not seen", (label_at[0] + 10, label_at[1]), HELD, 0.38)
        return

    if track.status == "uncertain":
        draw_points(image, objects[cluster["members"], :2], UNCERTAIN, 1)
        draw_box(image, cluster["centre"], cluster["length"], cluster["width"],
                 cluster["yaw"], UNCERTAIN, "dashed", 1)
        # Only the score here; the reasons are in the list next to the view.
        corner = to_pixels(cluster["centre"])[0]
        put_text(image, f"?{track.confidence:.2f}", (corner[0] + 8, corner[1] + 4), UNCERTAIN, 0.36)
        return

    draw_points(image, objects[cluster["members"], :2], colour, 2)
    draw_box(image, cluster["centre"], cluster["length"], cluster["width"],
             cluster["yaw"], colour, "solid", 2)

    # Velocity arrow: where the vehicle will be in 1 second.
    if track.speed is not None and track.speed >= settings.STANDING_SPEED:
        start = to_pixels(cluster["centre"])[0]
        end = to_pixels(cluster["centre"] + track.ego_velocity)[0]
        cv2.arrowedLine(image, tuple(start), tuple(end), colour, 2, cv2.LINE_AA, tipLength=0.3)

    corner = to_pixels(cluster["centre"])[0]
    put_text(image, f"#{track.id} {track.confidence:.2f}", (corner[0] + 12, corner[1] - 4),
             colour, 0.45)


# Front camera picture (only to help the viewer, not used for detection)

def camera_view(dataset, result, camera_frame, width):
    """
    Front camera picture closest in time, with the points of every shown
    track projected into it. The LiDAR and the camera don't fire at exactly
    the same moment, so the points go LiDAR-time ego frame -> world ->
    camera-time ego frame -> camera, then through the lens (intrinsics).
    """
    image = cv2.imread(dataset.image_path(camera_frame))
    intrinsic, ego_to_camera = dataset.camera_matrices(camera_frame)
    lidar_ego_to_camera_ego = (np.linalg.inv(dataset.ego_to_global(camera_frame))
                               @ result["ego_to_global"])
    full_transform = ego_to_camera @ lidar_ego_to_camera_ego

    objects = result["objects"]
    for track in shown_tracks(result["tracks"]):
        if track.status == "held":
            continue
        colour = track_colour(track.id) if track.status == "vehicle" else UNCERTAIN
        xyz = objects[track.cluster["members"], :3]
        in_camera = xyz @ full_transform[:3, :3].T + full_transform[:3, 3]
        in_front = in_camera[:, 2] > 1.0           # z = viewing direction
        if not in_front.any():
            continue
        # Pinhole camera: multiply by the intrinsic matrix, then divide by
        # the depth (things further away end up closer to the image centre).
        pixels = in_camera[in_front] @ intrinsic.T
        pixels = (pixels[:, :2] / pixels[:, 2:3]).astype(int)
        for u, v in pixels:
            cv2.circle(image, (int(u), int(v)), 4, colour, -1)
        if track.status == "vehicle":
            top_left = pixels.min(axis=0)
            put_text(image, f"#{track.id}", (top_left[0], top_left[1] - 10), colour, 1.3, 3)

    height = int(image.shape[0] * width / image.shape[1])
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


# Complete output frame

PANEL_WIDTH = 480


def render_frame(dataset, result, camera_frame, scene_name):
    """Bird's-eye view on the left, camera + vehicle list on the right."""
    bev = final_view(result)
    panel = np.full((BEV_SIZE, PANEL_WIDTH, 3), (40, 40, 40), np.uint8)

    camera = camera_view(dataset, result, camera_frame, PANEL_WIDTH)
    panel[:camera.shape[0]] = camera
    y = camera.shape[0] + 26

    tracks = result["tracks"]
    vehicles = [t for t in tracks if t.status == "vehicle"]
    uncertain = [t for t in shown_tracks(tracks) if t.status == "uncertain"]
    put_text(panel, f"{scene_name}   t = {result['scene_time']:.1f} s", (12, y), TEXT, 0.55)
    y += 22
    put_text(panel, f"own speed {result['ego_speed'] * 3.6:.0f} km/h   "
                    f"vehicles {len(vehicles)}   uncertain {len(uncertain)}", (12, y), TEXT, 0.5)
    y += 14
    cv2.line(panel, (12, y), (PANEL_WIDTH - 12, y), (90, 90, 90), 1)
    y += 20

    # Vehicle list, closest first, then the uncertain ones.
    listed = sorted(vehicles, key=lambda t: t.distance) + sorted(uncertain, key=lambda t: t.distance)
    for track in listed[:12]:
        if track.status == "vehicle":
            colour, name = track_colour(track.id), f"#{track.id}"
            detail = motion_text(track)
        else:
            colour, name = UNCERTAIN, "?"
            detail = "? " + ", ".join(track.doubts[:2])
        put_text(panel, f"{name:>4} {track.distance:4.0f} m  {track.confidence:.2f}",
                 (12, y), colour, 0.45)
        put_text(panel, detail, (170, y), colour, 0.42)
        y += 19

    draw_legend(panel, BEV_SIZE - 92)
    return np.hstack([bev, panel])


def draw_legend(panel, y):
    cv2.line(panel, (12, y - 18), (PANEL_WIDTH - 12, y - 18), (90, 90, 90), 1)
    cv2.rectangle(panel, (14, y - 9), (34, y + 1), PALETTE[0], 2)
    put_text(panel, f"vehicle, score >= {settings.CONFIDENT_SCORE:.2f} (colour = ID, arrow = 1 s)",
             (44, y), TEXT, 0.4)
    y += 22
    dashed_line(panel, (14, y - 4), (34, y - 4), UNCERTAIN, 2, 4)
    put_text(panel, "uncertain: score and main reason for doubt", (44, y), UNCERTAIN, 0.4)
    y += 22
    dashed_line(panel, (14, y - 4), (34, y - 4), HELD, 1, 4)
    put_text(panel, "vehicle not seen in this scan (predicted)", (44, y), HELD, 0.4)
    y += 22
    put_text(panel, "grey points: ground (dark), other objects (light)", (44, y), (150, 150, 150), 0.4)


# Diagnostic picture: four pipeline steps side by side

def diagnostic_grid(result, tile_size=540):
    """2 x 2 grid: raw scan, ground removal, clusters, final result."""
    def tile(image, title):
        image = cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, tile_size - 30), (tile_size, tile_size), (0, 0, 0), -1)
        put_text(image, title, (10, tile_size - 10), TEXT, 0.55)
        return image

    top = np.hstack([
        tile(raw_view(result), "1. raw scan, colour = height"),
        tile(ground_view(result), "2. downsampled: ground (brown) / objects"),
    ])
    bottom = np.hstack([
        tile(cluster_view(result), "3+4. clusters, boxes = vehicle candidates"),
        tile(final_view(result), "5. tracked vehicles and uncertain ones"),
    ])
    return np.vstack([top, bottom])
