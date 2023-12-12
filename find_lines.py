import cv2
import numpy as np
from clustering import main
from geometry import to_non_homogenous

path = r"YorkUrbanDB\P1080119\P1080119.jpg"
img = cv2.imread(path, 0)
blank = np.ones(img.shape, dtype=np.uint8) * 255
def merge_segments(lines, max_length):
    merged_lines = []
    for line in lines:
        x1, y1, x2, y2 = map(int, line[0])
        length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

        if length > max_length:
            merged_lines.append([x1, y1, x2, y2])

    return merged_lines

# Apply Gaussian blur to emphasize larger structures
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Create LSD detector
lsd = cv2.createLineSegmentDetector(2, quant=1)
lines, _, _, _ = lsd.detect(img)

# Merge short line segments into longer ones
max_segment_length = 20  # Adjust this threshold based on your requirements
merged_lines = merge_segments(lines, max_segment_length)

# Draw merged lines on a new image
merged_img = np.ones_like(img) * 255
for line in merged_lines:
    x1, y1, x2, y2 = map(int, line)
    cv2.line(merged_img, (x1, y1), (x2, y2), 0, 1)  # Draw lines in black

# Show the result
cv2.imshow("Merged Lines", merged_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()



# probablistic
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 100, apertureSize=7)
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=75, minLineLength=25, maxLineGap=9)
new_lines = []
for line in lines:
    line = line[0]
    p1, p2 = (line[0], line[1]), (line[2], line[3])
    theta = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    if np.abs(theta) < 0.5 or np.abs(theta - np.pi / 2) < 0.5:
        new_lines.append(line)
        cv2.line(img, p1, p2, (0, 255, 0), 1)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()

segments = [[(line[0], line[1], 1), (line[2], line[3], 1)] for line in new_lines]
clusters, C = main(segments)
for color, cluster in zip([(0, 0, 255), (0, 255, 0), (255, 0, 0)], clusters):
    for seg in cluster:
        p1, p2 = seg
        p1, p2 = to_non_homogenous(p1), to_non_homogenous(p2)
        cv2.line(img, p1, p2, color, 1)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()