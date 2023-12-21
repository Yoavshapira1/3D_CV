import random
import time
import cv2
import numpy as np
from utilities import draw_lines_and_centroids

def first_seed(segments, q, n):
    """
    receives homogenous segments and choose first seed and centroid
    """
    C = []
    idx = np.argsort(q.flatten())[-2*n:]
    chosen = segments[idx]
    for i in range(n):
        h = chosen[i:i+2]
        l_h_0 = np.cross(h[0][0], h[0][1])
        l_h_1 = np.cross(h[1][0], h[1][1])
        c = np.cross(l_h_0, l_h_1)
        C.append(c)
    return C


def seg_to_line(segment):
    return np.cross(segment[0], segment[1])


def seg_to_line_vec(segment):
    return np.cross(segment[:,0], segment[:,1], axisa=-1, axisb=-1)


def D_proj(c, l):
    return np.abs(np.dot(c, l)) / (np.linalg.norm(c) * (np.linalg.norm(l)))


def D_proj_vec(c, l_vec):
    dot_prod = np.sum(c * l_vec, axis=1)
    return np.abs(dot_prod) / (np.linalg.norm(c) * (np.linalg.norm(l_vec, axis=1)))


def D_proj_mat(c_vec, l_vec):
    dot_prod = np.abs(np.dot(l_vec, np.array(c_vec).T))
    c_norm = dot_prod / np.linalg.norm(c_vec, axis=1)
    line_norm = c_norm / np.linalg.norm(l_vec, axis=1)[:,None]
    return line_norm


def min_pseudo_centroid_cluster(line, C):
    """
    receives line and the centroids groups and return the minimal centroid
    returns the index of the optimal cluster
    """
    projs = []
    for c in C:
        projs.append(D_proj(c, line))
    return np.argmin(projs)


def find_theta_h(cluster, q):
    thetas = np.arctan2((cluster[:,1] - cluster[:,0])[:,1], (cluster[:,1] - cluster[:,0])[:,0])[:,None]
    s_h = np.sum(q * np.sin(2 * thetas))
    c_h = np.sum(q * np.cos(2 * thetas))
    t_h = np.arctan(s_h / c_h) #if c_h != 0 else np.pi / 2
    return t_h, thetas


def update_cluster_seeds(cluster, q):
    t_h, thetas = find_theta_h(cluster, q)
    diff = np.abs(thetas - t_h) % (2 * np.pi)
    radians = np.min(np.c_[diff, 2 * np.pi - diff], axis=1)
    min_angle = np.argmin(radians)
    alpha_h = cluster[min_angle]
    M = np.cross(seg_to_line(alpha_h), seg_to_line_vec(cluster))
    M = np.delete(M, min_angle, 0)
    M_proj_M_sum = np.argmin(np.sum(D_proj_mat(M, M), axis=0))
    min_sum = np.argmin(M_proj_M_sum)
    beta_h = cluster[min_sum if min_sum < min_angle else min_sum + 1]
    return alpha_h, beta_h


def build_clusters_from_centroids(segments, centroids, q, n):
    clusters, q_for_clusters = [[] for i in range(n)], [[] for i in range(n)]
    lines = seg_to_line_vec(segments)
    distances = D_proj_mat(centroids, lines)
    min_dist = np.argmin(distances, axis=1)

    for i in range(n):
        clusters[i] = segments[min_dist == i]
        q_for_clusters[i] = q[min_dist == i]
    return clusters, q_for_clusters


def is_converged(old, new):
    if old is None:
        return False

    for i in range(len(new)):
        if old[i].shape != new[i].shape:
            return False

        if not (new[i] == old[i]).all():
            return False

    print("converged")
    return True

def clusters_to_tuples(clusters):
    tuple_clusters = list()
    for cluster in clusters:
        tuple_cluster = list()
        sorted_cluster = sorted(cluster, key=lambda s: s[0][0]+s[0][1]+s[1][0]+s[1][1])
        for i in range(len(sorted_cluster)):
            seg = sorted_cluster[i]
            tuple_cluster.append((tuple(seg[0]), tuple(seg[1])))
        tuple_clusters.append(tuple(tuple_cluster))
    return tuple(tuple_clusters)

def centroid_to_tuple(centroids):
    tuple_centroids = list()
    sorted_centroids = sorted(centroids, key=lambda s: s[0] + s[1] + s[2])
    for c in sorted_centroids:
        tuple_centroids.append(tuple(c))
    return tuple(tuple_centroids)



def loop(segments, centroids, q, iter,n, img):
    start = time.time()
    centroids_set = set()
    centroids_tuple = centroid_to_tuple(centroids)
    centroids_set.add(centroids_tuple)

    for i in range(iter):
        clusters, q_for_clusters = build_clusters_from_centroids(segments, centroids, q, n)
        print([len(c) for c in clusters], "total: ", np.sum([len(c) for c in clusters]))
        centroids_new = []
        for i in range(len(clusters)):
            c, q_c = clusters[i], q_for_clusters[i]
            a, b = update_cluster_seeds(c, q_c)
            centroids_new.append(np.cross(seg_to_line(a), seg_to_line(b)))

        centroids_tuple = centroid_to_tuple(centroids_new)
        if centroids_tuple in centroids_set:
            if (centroids==np.array(centroids_new)).all():
                print("converged")
            print("Loop detected")
            break

        centroids = np.array(centroids_new)
        centroids_set.add(centroids_tuple)
        draw_lines_and_centroids(clusters, centroids, img)
    print(time.time() - start)
    return clusters, centroids


def find_clusters(segments, q, img, iter=50, n=3):
    centroids = first_seed(segments, q,n)
    clusters, C = loop(segments, centroids, q, iter,n, img)
    return clusters, C

