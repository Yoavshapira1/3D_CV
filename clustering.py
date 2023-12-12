import random
import time

import cv2
import numpy as np

def first_seed(segments, n=3):
    """
    receives homogenous segments and choose first seed and centroid
    """
    C = []
    chosen = random.sample(segments, 2*n)
    for i in range(n):
        h = chosen[i:i+2]
        l_h_0 = np.cross(h[0][0], h[0][1])
        l_h_1 = np.cross(h[1][0], h[1][1])
        c = np.cross(l_h_0, l_h_1)
        C.append(c)

    return C

def seg_to_line(segment):
    return np.cross(segment[0], segment[1])

def D_proj(c, l):
    return np.linalg.norm(np.dot(c, l)) / (np.linalg.norm(c) * (np.linalg.norm(l)))

def min_pseudo_centroid_cluster(line, C):
    """
    receives line and the centroids groups and return the minimal centroid
    returns the index of the optimal cluster
    """
    projs = []
    for c in C:
        projs.append(D_proj(c, line))
    return np.argmin(projs)


def find_theta_h(cluster, q_list):
    s_h, c_h, thetas = 0, 0, []
    for seg, q in zip(cluster, q_list):
        thetas.append(np.arctan2(seg[1][1] - seg[0][1], seg[1][0] - seg[0][0]))
        s_h += q * np.sin(2 * thetas[-1])
        c_h += q * np.cos(2 * thetas[-1])
    t_h = np.arctan(s_h / c_h) if c_h != 0 else np.pi / 2
    return t_h, thetas


def update_cluster_seeds(cluster):
    t_h, thetas = find_theta_h(cluster, [1]*len(cluster))
    angles = []
    for t in thetas:
        diff = abs(t - t_h) % (2*np.pi)
        angles.append(min(diff, (2*np.pi) - diff))
    min_angle = np.argmin(angles)
    alpha_h = cluster[min_angle]

    projs = []
    for seg in cluster:
        sum = 0
        for seg2 in cluster:
            if seg == seg2:
                continue
            a_cross_seg = np.cross(seg_to_line(alpha_h), seg_to_line(seg))
            a_cross_seg2 = np.cross(seg_to_line(alpha_h), seg_to_line(seg2))
            sum += D_proj(a_cross_seg, a_cross_seg2)
        projs.append(sum)
    beta_h = cluster[np.argmin(projs)]

    return alpha_h, beta_h


def build_clusters_from_centroids(segments, centroids):
    clusters = [[] for i in range(3)]
    for seg in segments:
        line = seg_to_line(seg)
        cluster_idx = min_pseudo_centroid_cluster(line, centroids)
        clusters[cluster_idx].append(seg)
    return clusters


def loop(segments, centroids):
    start = time.time()
    for i in range(15):
        clusters = build_clusters_from_centroids(segments, centroids)
        print([len(c) for c in clusters], "total: ", np.sum([len(c) for c in clusters]))
        centroids_new = []
        for c in clusters:
            a, b = update_cluster_seeds(c)
            centroids_new.append(np.cross(seg_to_line(a), seg_to_line(b)))
        if np.equal(centroids_new, centroids).all():
            break
        centroids = centroids_new
    print(time.time() - start)
    return clusters, centroids


def main(segments):
    centroids = first_seed(segments)
    clusters, C = loop(segments, centroids)
    return clusters, C

