def get_centre_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_width_of_bbox(bbox):
    x1, _, x2, _ = bbox
    return int(x2 - x1)

# Using the formula to calculate distance
def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def measure_xy_distance(p1, p2):
    return p1[0] - p2[0], p1[1] - p2[1]

# Getting the foot position of the player
def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), y2