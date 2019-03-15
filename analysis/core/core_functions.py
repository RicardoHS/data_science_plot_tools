import numpy as np

def compute_areas_between_lines(y1,y2):
    diff = y1 - y2
    positive_values = np.maximum(diff, 0)
    negative_values = np.minimum(diff, 0)
    y1_area = np.trapz(positive_values)[0]
    y2_area = np.trapz(negative_values)[0]
    return y1_area, y2_area