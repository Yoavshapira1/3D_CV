# main code logic is here

from cv2 import imread
from geometry import calculate_W
from utilities import find_horizon, sample_object, find_wave


def avg_x_axes(a, b):
    """Return points with same y values as b,r but with average x values for both
    So the line (a X b) is perpendicular to the x axes"""
    x_avg = (a[0] + b[0]) // 2
    a = (x_avg, a[1])
    b = (x_avg, b[1])
    return a, b


if __name__ == "__main__":

    # load the image
    img = imread(r"pics/sea.jpg", 1)

    # find the horizon and save as 2 points
    h1, h2 = find_horizon(img)

    # find the reference object and save as 2 points - lower and upper, and also ask for the object's height
    b, r, H = sample_object(img)
    # convert the points to ones that create a line perpendicular to the x axes
    b, r = avg_x_axes(b, r)

    # find the wave - lower and upper points
    b0, t0 = find_wave(img)
    # convert the points to ones that create a line perpendicular to the x axes
    b0, t0 = avg_x_axes(b0, t0)

    # # our example from ex2
    # b = (158, 355)
    # r = (155, 40)
    # b0 = (262, 241)
    # t0 = (264, 126)
    # h1 = (435, 74)
    # h2 = (241, 75)
    # H = 124

    # calculate the heights of the wave
    calculate_W(b, r, H, b0, t0, h1, h2)