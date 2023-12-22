# import cv2
# import numpy as np
# from utilities import plot_and_wait, draw_line_on_img
# import json
# from clustering import find_clusters, D_proj_vec, seg_to_line, seg_to_line_vec
# from geometry import to_non_homogenous
#
#
# def create_q_mat(nfa):
#     F_min = np.min(nfa)
#     F_max = np.max(nfa)
#     q = 1 - ((nfa - F_min) / (F_max - F_min))
#     return q
#
#
# def plot_and_exit(lines, img):
#     for l in lines:
#         x1, y1, x2, y2 = map(int, l[0])
#         cv2.line(img, (x1, y1), (x2, y2), 0, 1)
#     plot_and_wait(img)
#
#
# def final_points(clusters):
#     vanishing_points = []
#     for h in clusters:
#         lines = seg_to_line_vec(h)
#         min_sum, min_cross = np.inf, None
#         for i, s_i in enumerate(h):
#             for j, s_j in enumerate(h):
#                 if i < j:
#                     l_i, l_j = seg_to_line(s_i), seg_to_line(s_j)
#                     cross = np.cross(l_i, l_j)
#                     sum = np.sum(D_proj_vec(cross, lines))
#                     if sum < min_sum:
#                         min_sum, min_cross = sum, cross
#         vanishing_points.append(min_cross)
#     return vanishing_points
#
#
# def find_vertical_point(clusters, qs=None):
#     if qs is None:
#         qs = [np.ones(len(cluster)) for cluster in clusters]
#     angles = []
#     for q, cluster in zip(qs, clusters):
#         thetas = np.arctan2((cluster[:,1] - cluster[:,0])[:,1], (cluster[:,1] - cluster[:,0])[:,0])[:,None]
#         s_h = np.sum(q * np.sin(2 * thetas))
#         c_h = np.sum(q * np.cos(2 * thetas))
#         R_h = np.sqrt(s_h**2 + c_h**2) / np.sum(q)
#         sigma_h = np.sqrt(-2*np.log(R_h)) / 2
#         theta_h = np.arctan(s_h/c_h)
#         angles.append(min(theta_h, np.pi/2 - theta_h) + sigma_h)
#     angles = np.array(angles)
#     return angles.argmin()
#
#
# def find_vanishing_points(img, plot_detected=False, iter=15, quant=5):
#
#
#     lines = np.moveaxis(lines, 1, 0)[0]
#     segments = np.array([[np.array([line[0], line[1], 1]), np.array([line[2], line[3], 1])] for line in lines])
#     clusters, C = find_clusters(segments, q, iter=iter)
#     for color, cluster in zip([(0, 0, 255), (0, 255, 0), (255, 0, 0)], clusters):
#         for seg in cluster:
#             p1, p2 = seg
#             p1, p2 = to_non_homogenous(p1), to_non_homogenous(p2)
#             cv2.line(img, p1, p2, color, 1)
#     plot_and_wait(img)
#     print("clustered")
#
#     for name, clu in zip(['c1', 'c2', 'c3'], clusters):
#         with open('%s.json'%name, 'w') as f:
#             json.dump(np.array(clu).tolist(), f)
#     print("dumped")
#
#     v_points = final_points(clusters)
#     print("v points found")
#     idx = find_vertical_point(clusters)
#
#     print(f"the vertical point is {v_points[idx]} which is the {idx} point")
#     draw_line_on_img(img, to_non_homogenous(v_points[1]), to_non_homogenous(v_points[2]), color=(255,255,255), thickness=5, show=True)
#
#
#     return v_points
