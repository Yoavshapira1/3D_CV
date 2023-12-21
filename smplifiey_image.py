import cv2
import numpy as np
from main import resize_image

def filter_lines(image_path, length_threshold=50):
    # Read the image
    img = cv2.imread(image_path)
    new_width = 500
    img = resize_image(img, new_width)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect lines using LSD
    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)
    lines_img, _, _, _ = lines.detect(blurred)

    # Filter lines based on length
    filtered_lines = [line[0] for line in lines_img if np.linalg.norm(line[0][0:2] - line[0][2:4]) > length_threshold]

    # Create an empty image to draw the filtered lines
    filtered_img = np.zeros_like(img)

    # Draw the filtered lines on the image
    for line in filtered_lines:
        x1, y1, x2, y2 = map(int, line)
        cv2.line(filtered_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the original and filtered images
    cv2.imshow('Original Image', img)
    cv2.imshow('Filtered Lines', filtered_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage with a length threshold of 50 pixels
filter_lines(r"pics/3Dchess.jpg", length_threshold=10)