import cv2
import numpy as np
import scipy
from utilities import plot_and_wait, draw_line_on_img
import json
from clustering import find_clusters, D_proj, D_proj_vec, seg_to_line, seg_to_line_vec
from geometry import to_non_homogenous

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


def laplacian(img):
    s = cv2.Laplacian(img, cv2.CV_16S, ksize=5)
    s = cv2 .convertScaleAbs(s)
    return img - s


def draw_hough_lines(img, lines):
    for p in lines:
        rho, theta = p[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


def find_vanishing_points(img, plot_detected=False, iter=15, quant=5, th=50):

    # create blank background for plot the lines
    blank = np.ones(img.shape, dtype=np.uint8) * 255

    plot_and_wait(img)

    # Edges
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 250, 300, apertureSize=3)
    plot_and_wait(edges)

    # Hough
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    # draw_hough_lines(blank, lines)
    # plot_and_wait(blank)

    # Probabilistic
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, th, maxLineGap=25, minLineLength=10)
    for l in lines:
        x1, y1, x2, y2 = l[0]
        cv2.line(blank, (x1, y1), (x2, y2), (0, 0, 255), 2)
    plot_and_wait(blank)

    # plot the detected lines
    if plot_detected:
        plot_and_exit(lines, blank)

    # clustering algorithm
    # q = create_q_mat(nfa)
    q = np.array([1. for l in lines])
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


# Good Results: P1080106, 1080119

# load the image
path = r"YorkUrbanDB/P1080093/P1080093.jpg"
img = cv2.imread(path)

# run clustering
v_points = [to_non_homogenous(p) for p in find_vanishing_points(img.copy(),
                                                                plot_detected=False,
                                                                iter=1000,
                                                                quant=5,
                                                                th=30)]

# # from loaded clusters
# v_points = load_clusters()

print(v_points)

# plot the vanishing lines (3 of them) on the image
draw_line_on_img(img, v_points[0], v_points[1], color=(0, 0, 255))
draw_line_on_img(img, v_points[1], v_points[2], color=(0, 255, 0))
draw_line_on_img(img, v_points[0], v_points[2], color=(255, 0, 0), show=True)