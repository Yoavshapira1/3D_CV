import cv2
import numpy as np


def find_horizon(img) -> [tuple, tuple]:
    """
    find the horizon line in the image, represented by 2 points
    :param img: image of a seashore
    :return: two points on the horizon line
    """
    import cv2
    import numpy as np

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 5, 50)
    cv2.imshow("", edges)
    cv2.waitKey(0)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.01:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sample_object(img) -> [tuple, tuple]:
    """
    given an img, ask the user for input to detect the reference object, and return the lower & upper points of it.
    The algorithm is: . . .
    :return: lowest and highest points of the reference object
    """
    pass


def find_wave(img) -> [tuple, tuple]:
    """
    given an img of a seashore, find the wave as two points - higher and lower.
    :return: lowest and highest points of the wave
    """
    pass