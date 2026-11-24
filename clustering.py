"""
clustering.py

Steps 3 and 4 of the pipeline:
  3. group the non-ground points into clusters (DBSCAN)
  4. fit a box around every cluster and decide whether it could be a vehicle

Clustering is done on the bird's-eye (x, y) positions only. A 32-beam LiDAR
has big vertical gaps between its laser rings far away, so in 3D a distant
car would fall apart into several horizontal slices. Seen from above those
slices lie on top of each other and stay one cluster.
"""

import numpy as np
from sklearn.cluster import DBSCAN

import settings


def cluster_points(points_xy):
    """
    DBSCAN: points closer than CLUSTER_EPS are neighbours, and chains of
    neighbours form one cluster. Points in thin areas get the label -1
    (noise). Returns one label per point.
    """
    if len(points_xy) == 0:
        return np.array([], dtype=int)
    dbscan = DBSCAN(eps=settings.CLUSTER_EPS, min_samples=settings.CLUSTER_MIN_SAMPLES)
    # fit_predict gives every point a cluster number (0, 1, 2, ...) or -1.
    return dbscan.fit_predict(points_xy)


def fit_box(points_xy):
    """
    Smallest rectangle (seen from above) around the points. I simply try
    angles from 0 to 90 degrees, turn the points by that angle, measure the
    axis-aligned rectangle and keep the angle that gives the smallest area.
    (Every rectangle repeats itself after 90 degrees, so that's enough.)

    Returns centre (x, y), length (long side), width (short side) and yaw
    (direction of the long side, radians).
    """
    # All 45 angles at once: one row per angle, one column per point.
    angles = np.radians(np.arange(0, 90, 2))
    cos, sin = np.cos(angles)[:, None], np.sin(angles)[:, None]
    x, y = points_xy[:, 0], points_xy[:, 1]
    # Turn the points by -angle so the candidate box becomes axis-aligned.
    # u is the position along the turned x axis, v along the turned y axis.
    u = x * cos + y * sin
    v = -x * sin + y * cos
    size_u = u.max(axis=1) - u.min(axis=1)
    size_v = v.max(axis=1) - v.min(axis=1)
    # At least 0.1 m per side, otherwise a cluster that is a straight line
    # (width 0) would always win with an area of 0.
    area = np.maximum(size_u, 0.1) * np.maximum(size_v, 0.1)

    best = int(np.argmin(area))
    angle, cos, sin = angles[best], cos[best, 0], sin[best, 0]
    size_u, size_v = size_u[best], size_v[best]
    centre_u = (u[best].min() + u[best].max()) / 2
    centre_v = (v[best].min() + v[best].max()) / 2
    # Turn the centre back into the normal x, y frame.
    centre = np.array([centre_u * cos - centre_v * sin,
                       centre_u * sin + centre_v * cos])

    # Length is always the long side, and yaw points along it.
    if size_u >= size_v:
        return centre, size_u, size_v, angle
    return centre, size_v, size_u, angle + np.pi / 2


def describe_clusters(points, height, labels):
    """
    One dictionary per cluster with everything the next steps need:
    which points belong to it, the box, height above ground, distance.
    """
    clusters = []
    for label in np.unique(labels):
        if label < 0:
            continue  # DBSCAN noise
        members = np.flatnonzero(labels == label)
        centre, length, width, yaw = fit_box(points[members, :2])
        clusters.append({
            "members": members,
            "n_points": len(members),
            "centre": centre,
            "length": float(length),
            "width": float(width),
            "yaw": float(yaw),
            # lowest and highest point, measured above the local ground
            "bottom": float(height[members].min()),
            "top": float(height[members].max()),
            # straight-line distance from our car to the box centre
            "distance": float(np.hypot(*centre)),
        })
    return clusters


def check_vehicle_rules(cluster):
    """
    Hard rules from settings.py. Returns "" when the cluster could be a
    vehicle, otherwise a short reason why not.
    """
    # The checks go from cheapest to most specific; the first one that
    # fails is the reason shown in the diagnostic video.
    if cluster["n_points"] < settings.MIN_VEHICLE_POINTS:
        return "too few points"
    if cluster["top"] < settings.MIN_VEHICLE_TOP:
        return "too low"
    if cluster["bottom"] > settings.MAX_VEHICLE_BOTTOM:
        return "floating (not on the ground)"
    if cluster["length"] < settings.MIN_VEHICLE_LENGTH:
        return "too small"
    if cluster["length"] > settings.MAX_VEHICLE_LENGTH:
        return "too long"
    if cluster["width"] > settings.MAX_VEHICLE_WIDTH:
        return "too wide"
    if cluster["length"] > settings.WALL_MIN_LENGTH and cluster["width"] < settings.WALL_MAX_WIDTH:
        return "long and thin (wall, fence?)"
    return ""


def inside(value, limits):
    return limits[0] <= value <= limits[1]


def shape_score(cluster):
    """
    How much does the cluster look like a vehicle, judged from one scan?
    Gives three parts between 0 and 1 and a list of reasons for doubt.

      points : more points = more evidence (far cars have few)
      size   : footprint of a car, or of a van / truck / bus
      height : roof height of a car. Taller clusters are a problem: most
               false alarms were walls, hedges and bushes, which reach the
               top of the body layer. But so do vans, trucks and buses. From
               one scan we can't tell them apart, so "tall" only lowers the
               score here; the tracker removes the doubt if it moves.
    """
    reasons = []

    points_part = min(1.0, cluster["n_points"] / settings.POINTS_FOR_FULL_SCORE)
    if points_part < 0.5:
        reasons.append("few points")

    length, width = cluster["length"], cluster["width"]
    if inside(length, settings.CAR_LENGTH) and inside(width, settings.CAR_WIDTH):
        size_part, kind = 1.0, "car"
    elif inside(length, settings.BIG_VEHICLE_LENGTH) and inside(width, settings.BIG_VEHICLE_WIDTH):
        size_part, kind = 1.0, "big vehicle"
    elif length < settings.CAR_LENGTH[0] or width < settings.CAR_WIDTH[0]:
        # Usually we only see one or two sides of a car (the LiDAR can't see
        # behind it), so the box is too short or too thin. Could also be a
        # bike, a bush or a small wall, so this gets only half the points.
        size_part, kind = 0.5, "partial view"
        reasons.append("partial view")
    else:
        size_part, kind = 0.3, "unusual size"
        reasons.append("unusual size")

    cluster["tall"] = cluster["top"] > settings.CAR_MAX_TOP
    if cluster["tall"]:
        height_part = settings.TALL_HEIGHT_SCORE
        reasons.append("tall: truck or wall?")
    elif cluster["top"] < settings.CAR_MIN_TOP:
        # Far away the upper laser rings can pass over the roof.
        height_part = 0.5
        reasons.append("low")
    else:
        height_part = 1.0

    parts = {"points": points_part, "size": size_part, "height": height_part}
    return parts, kind, reasons


def find_vehicle_candidates(points, height):
    """
    Runs steps 3 and 4. Returns (labels, clusters) where every cluster has
    "reject_reason" ("" for vehicle candidates) and, for candidates, the
    shape score parts.
    """
    labels = cluster_points(points[:, :2])
    clusters = describe_clusters(points, height, labels)
    for cluster in clusters:
        cluster["reject_reason"] = check_vehicle_rules(cluster)
        if not cluster["reject_reason"]:
            parts, kind, reasons = shape_score(cluster)
            cluster["score_parts"] = parts
            cluster["kind"] = kind
            cluster["doubts"] = reasons
    return labels, clusters
