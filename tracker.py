"""
tracker.py

Step 5: follow the vehicle candidates from scan to scan.

Why the global frame? Our own car moves. If we compared positions in the
ego frame, a parked car would seem to drive towards us at our own speed.
So every detection is first moved into the fixed world frame with the ego
pose of that scan ("ego-motion compensation"). There a parked car stays
put and a moving car moves by its own speed.

How a scan is handled:
  1. every track predicts where its vehicle should be now
     (last position + velocity * time passed)
  2. detections are matched to tracks, closest pairs first, but only when
     they are closer than MATCH_DISTANCE (greedy nearest neighbour)
  3. matched tracks take the new position; tracks without a match count a
     miss and are deleted after MAX_MISSES; detections without a track
     start a new track with a new ID

From a track's positions over the last second we get:
  * its speed over ground (is it parked or driving?)
  * how fast the distance to us changes (approaching / moving away)
"""

from collections import deque

import numpy as np

import settings
from nuscenes_data import apply_transform


class Track:
    def __init__(self, track_id, time, position, ego_position, cluster):
        self.id = track_id
        self.hits = 0     # how many scans it was detected in
        self.misses = 0   # scans in a row without a detection
        self.was_vehicle = False   # ever shown as a confident vehicle?
        # (time, vehicle position, our position), all in the world frame
        self.history = deque(maxlen=50)
        self.add(time, position, ego_position, cluster)

    def add(self, time, position, ego_position, cluster):
        self.history.append((time, position, ego_position))
        self.cluster = cluster
        self.hits += 1
        self.misses = 0

    @property
    def last_time(self):
        return self.history[-1][0]

    @property
    def position(self):
        return self.history[-1][1]

    def recent_history(self):
        """History entries of the last SPEED_WINDOW seconds."""
        newest = self.last_time
        return [h for h in self.history if newest - h[0] <= settings.SPEED_WINDOW]

    def velocity(self):
        """
        World-frame velocity (vx, vy) in m/s, from a straight-line fit
        through the recent positions. Fitting a line over ~1 s instead of
        using only the last two positions smooths out the jitter of the
        box centre. None while there is not enough history.
        """
        recent = self.recent_history()
        times = np.array([h[0] for h in recent])
        if len(recent) < 4 or times[-1] - times[0] < 0.4:
            return None
        positions = np.array([h[1] for h in recent])
        # polyfit(..., 1) fits a straight line: position = a * time + b.
        # The slope a is the speed in that direction.
        vx = np.polyfit(times, positions[:, 0], 1)[0]
        vy = np.polyfit(times, positions[:, 1], 1)[0]
        return np.array([vx, vy])

    def estimate_range_rate(self):
        """
        How fast the distance between us and the vehicle changes (m/s).
        Negative = getting closer. Uses our own position at the same times,
        so this is the relative speed along the line of sight.
        """
        recent = self.recent_history()
        times = np.array([h[0] for h in recent])
        if len(recent) < 4 or times[-1] - times[0] < 0.4:
            return None
        # Distance between the vehicle and us at each moment, then the
        # slope of that distance over time.
        distances = [np.linalg.norm(h[1] - h[2]) for h in recent]
        return float(np.polyfit(times, distances, 1)[0])

    def predict(self, time):
        """Where the vehicle should be at `time` if it keeps its velocity."""
        velocity = self.velocity()
        if velocity is None:
            return self.position
        return self.position + velocity * (time - self.last_time)


class VehicleTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def update(self, candidates, ego_to_global, time):
        """
        candidates: vehicle candidate clusters of this scan (ego frame)
        ego_to_global: 4x4 pose of our car for this scan
        time: scan time in seconds
        """
        # Our own position in the world, and every box centre moved from
        # the ego frame into the world frame (height is not needed).
        ego_position = ego_to_global[:2, 3]
        positions = []
        for cluster in candidates:
            centre = np.array([[cluster["centre"][0], cluster["centre"][1], 0.0]])
            positions.append(apply_transform(ego_to_global, centre)[0, :2])

        # 1 + 2: predict and match, closest pairs first.
        # Every track/detection pair that is close enough is a possible
        # match. After sorting by distance we take pairs from the top and
        # skip any track or detection that is already used, so each one is
        # matched at most once and the closest pairs win.
        pairs = []
        for t_index, track in enumerate(self.tracks):
            predicted = track.predict(time)
            for c_index, position in enumerate(positions):
                distance = np.linalg.norm(position - predicted)
                if distance < settings.MATCH_DISTANCE:
                    pairs.append((distance, t_index, c_index))
        pairs.sort()

        used_tracks, used_candidates = set(), set()
        for _, t_index, c_index in pairs:
            if t_index in used_tracks or c_index in used_candidates:
                continue
            self.tracks[t_index].add(time, positions[c_index], ego_position,
                                     candidates[c_index])
            used_tracks.add(t_index)
            used_candidates.add(c_index)

        # 3: misses, deletion and new tracks.
        # A track that got no detection is kept for a few scans (the car
        # may be hidden behind another one for a moment), then dropped.
        for t_index, track in enumerate(self.tracks):
            if t_index not in used_tracks:
                track.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= settings.MAX_MISSES]

        for c_index, cluster in enumerate(candidates):
            if c_index not in used_candidates:
                self.tracks.append(Track(self.next_id, time, positions[c_index],
                                         ego_position, cluster))
                self.next_id += 1

        for track in self.tracks:
            describe_track(track, ego_to_global, time)
        return self.tracks


def describe_track(track, ego_to_global, time):
    """
    Fills in what the display needs: position and motion in the ego frame,
    confidence score, status and the reasons for doubt.
    """
    # Where is it now, seen from our car? (for held tracks: the prediction)
    # The inverse of our pose turns world positions back into the ego frame.
    global_to_ego = np.linalg.inv(ego_to_global)
    position = track.predict(time) if track.misses > 0 else track.position
    track.ego_position = apply_transform(global_to_ego, np.array([[*position, 0.0]]))[0, :2]
    track.distance = float(np.linalg.norm(track.ego_position))

    velocity = track.velocity()
    track.speed = None if velocity is None else float(np.linalg.norm(velocity))
    # Velocity turned into the ego frame, so the arrow can be drawn.
    track.ego_velocity = None if velocity is None else global_to_ego[:2, :2] @ velocity
    track.range_rate = track.estimate_range_rate()

    # Confidence = average of the shape parts (from clustering.py) and a
    # tracking part that grows with the number of scans it was seen in.
    parts = dict(track.cluster["score_parts"])
    parts["tracking"] = min(1.0, track.hits / settings.HITS_FOR_FULL_SCORE)
    track.doubts = list(track.cluster["doubts"])

    # A tall cluster could be a wall or a truck. If it drives, it's a truck.
    is_moving = (track.speed is not None and track.speed > settings.MOVING_PROOF_SPEED
                 and track.hits >= settings.HITS_FOR_FULL_SCORE)
    if track.cluster["tall"] and is_moving:
        parts["height"] = 1.0
        track.doubts.remove("tall: truck or wall?")

    if track.hits < settings.CONFIRM_HITS:
        track.doubts.insert(0, "new track")

    track.score_parts = parts
    track.confidence = float(np.mean(list(parts.values())))

    # A tall cluster that stands still may be a parked truck, but just as
    # often it's a wall or a hedge. Its score is capped, so it can never
    # count as a confident vehicle, however many points it has.
    if "tall: truck or wall?" in track.doubts:
        track.confidence = min(track.confidence, settings.TALL_STANDING_MAX_SCORE)

    if track.misses > 0:
        track.status = "held"          # not seen in this scan
    elif track.hits >= settings.CONFIRM_HITS and track.confidence >= settings.CONFIDENT_SCORE:
        track.status = "vehicle"
    else:
        track.status = "uncertain"
    if track.status == "vehicle":
        track.was_vehicle = True


def motion_text(track):
    """Short human-readable motion description of a track."""
    if track.speed is None:
        return "motion: measuring..."
    if track.speed < settings.STANDING_SPEED:
        moving = "standing"
    else:
        moving = f"{track.speed * 3.6:.0f} km/h"

    rate = track.range_rate
    if rate < -settings.CLOSING_SPEED_LIMIT:
        relation = f"approaching {-rate:.1f} m/s"
    elif rate > settings.CLOSING_SPEED_LIMIT:
        relation = f"moving away {rate:.1f} m/s"
    else:
        relation = "same distance"
    return f"{moving}, {relation}"
