# This code implement the projective geometry calculation
# to find the height of the wav base on the object. Given:
# The horizon
# The object (low point "b" & high point "r")
# The wave (low point "b0" & high point "t0")
# The calculation process:
# 1) Connect "b" and "b0" with a line, and find its intersection with the horizon "v"
# 2) Connect "v" and "t0" with a line, and find its intersection with the object's line "t"
# 3) The height of the wave is W=d(b0, t0). Use cross-ratio to calculate W, using d(b, t) & d(b, r) & inf.
# All in all the formula is: W = ( |b-t| / |b-r| ) * H, where H is the object's height.
#
# All calculations here to find the intersections are
# done using homogenous coordinates and the cross product

import numpy as np


def to_homogenous(p):
    """Simply convert 2D point to homogenous coordinates"""
    return p[0], p[1], 1


def to_non_homogenous(p):
    """Convert homogenous coordinates to non-homogenous coordinates"""
    if len(p) < 3:
        return p
    if p[2] == 0:
        return None
    return p[0]//p[2], p[1]//p[2]


def find_intersection(l1_a, l1_b, l2_a, l2_b):
    """
    Finding intersection point P between 2 lines, each is defined with two points,
    using cross product and homogenous coordinates
    """

    # convert to homogenous
    line1_a_homo = to_homogenous(l1_a)
    line1_b_homo = to_homogenous(l1_b)
    line2_a_homo = to_homogenous(l2_a)
    line2_b_homo = to_homogenous(l2_b)

    # find the lines using cross product of the appropriate points
    line1 = np.cross(line1_a_homo, line1_b_homo)
    line2 = np.cross(line2_a_homo, line2_b_homo)

    # find the intersection with cross product with those two lines
    inters_homo = np.cross(line1, line2)
    return to_non_homogenous(inters_homo)


def calc_cr(b, t, r, R):
    """
    Calculating the height using cross ratio
    :param b: lower point of the object
    :param t: middle point of the object
    :param r: upper point of the object
    :param R: the real height of the object
    :return: the height of the object
    """
    b_t = np.abs(b[1] - t[1])
    b_r = np.abs(b[1] - r[1])
    H = (b_t / b_r) * R
    return H


def calculate_H(b, r, R, b0, t0, h1, h2):
    """
    calculating the height of the object - H
    :param b: lower point of the reference object
    :param r: upper point of the reference object
    :param R: the real height of the reference object
    :param b0: lower point of the object
    :param t0: upper point of the object
    :param h1: a point on the horizon
    :param h2: another point on the horizon
    :return: the height of the object
    """
    # v = find_intersection(b, b0, h1, h2)
    v = (825, 148)
    t = find_intersection(v, t0, b, r)
    print(t)
    print("our t:", (308, 321))
    H = calc_cr(b, t, r, R)
    return H
