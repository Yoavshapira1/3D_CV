import cv2
import numpy as np
from utilities import plot_and_wait
import json
from clustering import find_clusters, D_proj, D_proj_vec, seg_to_line, seg_to_line_vec
from geometry import to_non_homogenous

# def angle_between_2_points(p1, p2):
#     if np.linalg.norm(p1) < np.linalg.norm(p2):
#         t = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
#     else:
#         t = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
#     t = math.degrees(min(np.abs(t), (2 * np.pi) - t))
#     return t
#
# def calcluate_theats(lines):
#     # shape of input: (N, 4)
#     thetas = []
#     for l in lines:
#         x1, y1, x2, y2 = map(int, l)
#         t = angle_between_2_points((x1, y1), (x2, y2))
#         thetas.append(t)
#     return thetas
#
#
# def segments_are_on_the_same_line(seg1, orient1, seg2, orient2, dist_th, degree_th=5):
#     _, _, seg1_end_x, seg1_end_y = seg1
#     seg2_start_x, seg2_start_y, _, _ = seg2
#     equal_orientation = np.abs(orient1-orient2) % 90 < degree_th
#     theta_between_segments = angle_between_2_points((seg2_start_y, seg1_end_y), (seg2_start_x, seg1_end_x))
#     theta_align = orient2 <= theta_between_segments <= orient1 or orient1 <= theta_between_segments <= orient2
#     distance = np.linalg.norm([seg2_start_x - seg1_end_x, seg2_start_y - seg1_end_y])
#     if equal_orientation and theta_align and distance < dist_th:
#         return True
#     return False
#
#
# def dist(p1, p2):
#     return np.sqrt((p1[1] - p2[0])**2 + (p1[1] - p2[1])**2)
#
#
# def merge_with_far_points(seg1, seg2):
#     if dist(seg1[:2], seg2[2:]) > dist(seg1[2:], seg2[2:]):
#         return seg1[:2] + seg2[2:]
#     return seg1[2:] + seg2[2:]
#
#
# def merge_segments(segments, orientations, dist_th):
#     # shape of input: (N, 4)
#     merged = True
#     while merged:
#         merged = False
#         for i in range(len(segments)):
#             for j in range(i + 1, len(segments)):
#                 if segments_are_on_the_same_line(segments[i], orientations[i], segments[j], orientations[j], dist_th):
#                     # Merge segments i and j
#                     segments[i] = merge_with_far_points(segments[i], segments[j])
#                     del segments[j]
#                     del orientations[j]
#                     merged = True
#                     break
#             if merged:
#                 break
#     return segments


def create_q_mat(nfa):
    F_min = np.min(nfa)
    F_max = np.max(nfa)
    q = 1 - ((nfa - F_min) / (F_max - F_min))
    return q


def plot_and_exit(lines, img):
    for l in lines:
        x1, y1, x2, y2 = map(int, l[0])
        cv2.line(img, (x1, y1), (x2, y2), 0, 1)
    plot_and_wait(img)


def final_points(clusters):
    vanishing_points = []
    for h in clusters:
        lines = seg_to_line_vec(h)
        min_sum, min_cross = np.inf, None
        for i, s_i in enumerate(h):
            for j, s_j in enumerate(h):
                if i < j:
                    l_i, l_j = seg_to_line(s_i), seg_to_line(s_j)
                    cross = np.cross(l_i, l_j)
                    sum = np.sum(D_proj_vec(cross, lines))
                    if sum < min_sum:
                        min_sum, min_cross = sum, cross
        vanishing_points.append(min_cross)
    return vanishing_points


def find_vanishing_points(img, plot_detected=False, iter=15):

    # create blank background for plot the lines
    blank = np.ones(img.shape, dtype=np.uint8) * 255

    # Create LSD detector
    lsd = cv2.createLineSegmentDetector(2, quant=1)
    lines, width, prec, nfa = lsd.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    # plot the detected lines
    if plot_detected:
        plot_and_exit(lines, blank)

    # clustering algorithm
    q = create_q_mat(nfa)
    lines = np.moveaxis(lines, 1, 0)[0]
    segments = np.array([[np.array([line[0], line[1], 1]), np.array([line[2], line[3], 1])] for line in lines])
    clusters, C = find_clusters(segments, q, iter=iter)
    for color, cluster in zip([(0, 0, 255), (0, 255, 0), (255, 0, 0)], clusters):
        for seg in cluster:
            p1, p2 = seg
            p1, p2 = to_non_homogenous(p1), to_non_homogenous(p2)
            cv2.line(img, p1, p2, color, 1)
    plot_and_wait(img)
    print("clustered")

    for name, clu in zip(['c1', 'c2', 'c3'], clusters):
        with open('%s.json'%name, 'w') as f:
            json.dump(np.array(clu).tolist(), f)
    print("dumped")

    v_points = final_points(clusters)
    print("v points found")

    return v_points
