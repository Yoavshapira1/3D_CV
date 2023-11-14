import cv2
import numpy as np


def rotate_image(image, angle):
    """
    Rotating a given image in a given angle
    The angle should be in degrees, and should be less than 90 degrees in order to prevent vertical flip
    :return: the rotated image
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image


def get_rotation_angle(h1, h2):
    """
    Calculating the angle (in degrees) that the image should be rotated for the horizon to be horizontal.
    This will make the objects perpendicular the plane, and then we can average the x-values to get a vertical
    line from 2 points (in relation to the plane)
    :param h1: points on the horizon
    :param h2: another point on the horizon
    :return: the angle, in degrees (angle <= 90)
    """
    dx = h2[0] - h1[0]
    dy = h2[1] - h1[1]

    # Calculate the angle in radians
    angle = np.arctan2(dy, dx)
    if np.abs(angle) >= np.pi // 2:
        angle = -(np.pi - angle)

    # Convert radians to degrees
    angle_deg = np.degrees(angle)

    return angle_deg


def rotate_point(point, center, angle):
    angle_rad = np.radians(angle)
    x, y = point[0] - center[0], point[1] - center[1]
    new_x = x * np.cos(angle_rad) - y * np.sin(angle_rad) + center[0]
    new_y = x * np.sin(angle_rad) + y * np.cos(angle_rad) + center[1]
    return int(new_x), int(new_y)


def rotate_two_points(h1, h2, image, angle):
    # Rotate the horizon points
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotated_horizon_point1 = rotate_point(h1, center, -angle)
    rotated_horizon_point2 = rotate_point(h2, center, -angle)
    return rotated_horizon_point1, rotated_horizon_point2


def perp_points(p1, p2) -> tuple[tuple, tuple]:
    """given 2 points (lower & higher points of an object), return points that create a perpendicular
    line related to the object's plane, by returning the same y values but averaged x values.
    reminder: we assume that the object is standing on the plane, hence should be perpendicular anyway"""
    avg_x = (p1[0] + p2[0]) // 2
    p1 = (avg_x, p1[1])
    p2 = (avg_x, p2[1])
    return p1, p2


def draw_point_on_img(img, p, txt):
    """arguments: image, text, and a point in non-homogeneous coordinates"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, txt, p, font, 1,
                (0, 255, 255), 2)
    cv2.imshow('image', img)


def draw_line_on_img(img, p1, p2, show=False):
    """arguments: image, and 2 points in non-homogeneous coordinates"""
    cv2.line(img, p1, p2, color=(0, 255, 255), thickness=1)
    if show:
        cv2.imshow('image', img)