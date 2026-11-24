"""
settings.py

Every tuned number lives here, so changing a threshold never means
searching through the code. Distances are in metres, times in seconds.
All positions are in the ego frame (x forward, y left, z up, ground ~ 0).
"""


# Data

DATA_ROOT = "data/nuscenes-mini"

# The LiDAR records 20 scans per second. Using every 2nd one (10 per
# second) is plenty for tracking and halves the processing time.
FRAME_STEP = 2


# Step 1: area of interest and downsampling

# Square area around our car that we look at. Beyond ~40 m a car only
# gets a handful of points from the 32-beam LiDAR, too few to judge.
AREA_HALF_SIZE = 40.0

# Points on our own car (roof, mirrors) are removed with this box.
EGO_BOX_X = (-1.5, 3.5)
EGO_BOX_Y = (-1.2, 1.2)

# Voxel downsampling: space is cut into cubes of this size and every cube
# keeps one point (the average of its points). Close objects get thousands
# of points, far ones a few; this evens it out and makes clustering faster.
VOXEL_SIZE = 0.15


# Step 2: ground removal

# First try (kept in the README): one flat plane fitted with RANSAC. It
# worked near the car, but roads are not flat over 40 m (slopes, camber,
# kerbs), so far away cars looked "too low" or "floating" above the plane.
# Now the ground height is estimated locally on a grid:
#   - the area is cut into square cells of GROUND_CELL_SIZE
#   - in each cell the lowest point is a guess of the ground height
#   - the guess is replaced by the lowest guess of the 3 x 3 neighbouring
#     cells, because a cell completely covered by a car has no road points
GROUND_CELL_SIZE = 2.0

# Single points far below the road (reflections in puddles or windows)
# would pull the ground down, so points lower than this are ignored.
LOWEST_VALID_Z = -2.0

# Closer stray points below the road still happen. A cell whose lowest
# point is more than this below the median of its 8 neighbours is treated
# as such a stray point (real slopes don't drop 0.3 m from one 2 m cell
# to the next).
MAX_DIP_BELOW_NEIGHBOURS = 0.3

# Points less than this above the local ground are ground.
GROUND_MARGIN = 0.25

# Only the "vehicle body layer" is clustered: points up to this height
# above ground. Tree crowns, signs and traffic lights hang higher and
# would otherwise glue parked cars and trees into one big blob. Trucks
# and buses are taller, but their lower part is enough to find them.
BODY_LAYER_TOP = 2.5


# Step 3: clustering (DBSCAN on the bird's-eye x, y positions)

# Two points are "neighbours" when they are closer than CLUSTER_EPS.
# A point with at least CLUSTER_MIN_SAMPLES neighbours starts a cluster.
CLUSTER_EPS = 0.7
CLUSTER_MIN_SAMPLES = 4


# Step 4: which clusters could be vehicles

# Hard rules. A cluster failing any of these is not a vehicle at all.
MIN_VEHICLE_POINTS = 8         # after downsampling
MIN_VEHICLE_TOP = 0.9          # highest point; cars are about 1.5 m high
MAX_VEHICLE_BOTTOM = 0.8       # lowest point must be near the ground
MIN_VEHICLE_LENGTH = 1.4       # longest side of the box (rules out people, poles)
MAX_VEHICLE_LENGTH = 14.0      # an articulated bus is ~18 m, a normal bus 12 m
MAX_VEHICLE_WIDTH = 3.8        # shortest side (trucks 2.5 m + mirrors, box slack)

# Long and paper-thin = a wall or fence seen along its length.
WALL_MIN_LENGTH = 5.0
WALL_MAX_WIDTH = 1.0

# Typical sizes, used for the confidence score.
CAR_LENGTH = (3.2, 5.8)
CAR_WIDTH = (1.4, 2.3)
BIG_VEHICLE_LENGTH = (5.8, 14.0)   # vans, trucks, buses
BIG_VEHICLE_WIDTH = (2.0, 3.2)

# Roof height of a normal car (SUVs reach ~2 m).
CAR_MIN_TOP = 1.2
CAR_MAX_TOP = 2.1
# Height part of the score for clusters taller than a car (truck or wall?).
TALL_HEIGHT_SCORE = 0.3
# A tall cluster moving faster than this (and tracked for at least
# HITS_FOR_FULL_SCORE scans) is surely a vehicle, not a wall. Not lower:
# the visible piece of a wall changes as we drive past, so its box slides
# along and seems to move at a few km/h.
MOVING_PROOF_SPEED = 3.0
# Highest score a tall cluster can get while it is not seen moving.
TALL_STANDING_MAX_SCORE = 0.5

# A cluster this full of points gets the full "points" part of the score.
POINTS_FOR_FULL_SCORE = 60


# Step 5: tracking

# A new detection may be at most this far (metres) from where a track is
# expected to be, to count as the same vehicle.
MATCH_DISTANCE = 2.0
# A track is shown as "confirmed" after this many detections ...
CONFIRM_HITS = 3
# ... and deleted after this many scans in a row without a detection.
MAX_MISSES = 5
# Speeds are estimated from the positions over the last ~1 second.
SPEED_WINDOW = 1.0
# Below this speed (m/s) a vehicle is called "standing".
STANDING_SPEED = 1.0
# Range rate (m/s) below which we say "keeps distance".
CLOSING_SPEED_LIMIT = 0.5
# A track seen this many times gets the full "tracking" part of the score.
HITS_FOR_FULL_SCORE = 8

# Final decision on the confidence score (0..1).
CONFIDENT_SCORE = 0.65
# Uncertain candidates below this score are not drawn at all (clutter).
MIN_SCORE_TO_SHOW = 0.45


# Drawing

PIXELS_PER_METRE = 10          # bird's-eye view: 80 m x 80 m -> 800 x 800 px
