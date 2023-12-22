# main code logic is here
import json

import numpy as np
import scipy.io
import cv2
from geometry import calculate_H, to_non_homogenous
from choose_coor import choose_cord
from ClusteringWithHoughLines import find_vanishing_points
from utilities import draw_line_on_img


def avg_x_axes(a, b):
    """Return points with same y values as b,r but with average x values for both
    So the line (a X b) is perpendicular to the x axes"""
    x_avg = (a[0] + b[0]) // 2
    a = (x_avg, a[1])
    b = (x_avg, b[1])
    return a, b


def choose_coordinates(image, message):
    print(message)
    points = choose_cord(image)
    while len(points) != 2:
        print("choose only the bottom and top of the object")
        points = choose_cord(image)
    points.sort(key=lambda point: point[1], reverse=True)
    return points


if __name__ == "__main__":
    # good images: 1080104, P1080106, 1080119

    # load the image
    path = r"Jaffa/Glz/Glz_resized.jpeg"
    img = cv2.imread(path)

    # run clustering
    h1, h2, vertical_p = find_vanishing_points(img,
                          plot_detected=False,
                          iter=1000,
                          segmentetion_algorithm="LSD"
                          )

    draw_line_on_img(img, h1, h2, color=(255, 0, 0), show=True)

    # b = (190, 400)
    # r = (189, 278)
    # b0 = (445, 584)
    # t0 = (451, 462)

    h1 = (int(1253.615038), int(318.433753))
    h2 = (int(-474.315634), int(293.168654))
    b, r = choose_coordinates(image=img, message="Please choose the bottom and top points of your reference object.")
    print(f"the top point is {r} and the bottom point is {b}")
    b0, t0 = choose_coordinates(image=img, message="Please choose the bottom and top points of the object you "
                                                   "wold like to measure.")
    H = int(input("Please enter the height of your reference object"))
    print(calculate_H(b, r, H, b0, t0, h1, h2))