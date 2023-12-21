# main code logic is here
import json

import numpy as np
import scipy.io
import cv2
from cv2 import imread
from PIL import Image
from geometry import calculate_H
from utilities import draw_line_on_img
from find_lines import find_vanishing_points, to_non_homogenous


def avg_x_axes(a, b):
    """Return points with same y values as b,r but with average x values for both
    So the line (a X b) is perpendicular to the x axes"""
    x_avg = (a[0] + b[0]) // 2
    a = (x_avg, a[1])
    b = (x_avg, b[1])
    return a, b


def load_clusters():
    # Opening JSON file
    clusters = []
    for name in ['c1', 'c2', 'c3']:
        with open('%s.json' % name, 'r') as f:
            data = json.load(f)
            clusters.append(np.array(data))
    from find_lines import final_points
    v_points = [to_non_homogenous(p) for p in final_points(clusters)]

    return v_points


def resize_image(original_image, new_width):
    # Get the original image dimensions
    original_height, original_width = original_image.shape[:2]

    # Calculate the proportional height based on the new width
    new_height = int((new_width / original_width) * original_height)

    # Resize the image
    return  cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

if __name__ == "__main__":
    # good images: 1080104

    # load the image
    path = r"YorkUrbanDB/P1080119/P1080119.jpg"
    path = r"pics/coordinate_system.png"
    img = cv2.imread(path)
    new_width = 700
    img = resize_image(img, new_width)



    # run clustering
    v_points = [to_non_homogenous(p) for p in find_vanishing_points(img,
                                                                    plot_detected=True,
                                                                    iter=1000,
                                                                    quant=0.1)]

    # # from loaded clusters
    # v_points = load_clusters()

    print(v_points)

    # plot the vanishing lines (3 of them) on the image
    draw_line_on_img(img, v_points[0], v_points[1], color=(0, 0, 255))
    draw_line_on_img(img, v_points[1], v_points[2], color=(0, 255, 0))
    draw_line_on_img(img, v_points[0], v_points[2], color=(255, 0, 0), show=True)

    # mat = scipy.io.loadmat(r'C:\Users\Dell\Desktop\University\Year4\3D ראייה ממוחשבת\project\YorkUrbanDB\P1080119\P1080119GroundTruthVP_CamParams.mat')
    # it = mat.items()
    # matrix = mat["data_opp"]

    # # find the horizon and save as 2 points
    # h1, h2 = find_horizon(img)
    #
    # # find the reference object and save as 2 points - lower and upper, and also ask for the object's height
    # b, r, H = sample_object(img)
    # # convert the points to ones that create a line perpendicular to the x axes
    # b, r = avg_x_axes(b, r)
    #
    # # find the wave - lower and upper points
    # b0, t0 = find_wave(img)
    # # convert the points to ones that create a line perpendicular to the x axes
    # b0, t0 = avg_x_axes(b0, t0)
    #
    # # # our example from ex2
    # # b = (158, 355)
    # # r = (155, 40)
    # # b0 = (262, 241)
    # # t0 = (264, 126)
    # # h1 = (435, 74)
    # # h2 = (241, 75)
    # # H = 124
    #
    # # calculate the heights of the wave
    # calculate_H(b, r, H, b0, t0, h1, h2)