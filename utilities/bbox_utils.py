def get_centre_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_width_of_bbox(bbox):
    x1, _, x2, _ = bbox
    return int(x2 - x1)