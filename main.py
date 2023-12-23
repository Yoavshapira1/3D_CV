# main code logic is here
import cv2
from geometry import calculate_H, calculate_height_using_camera_matrix
from choose_coor import choose_cord
from ClusteringWithHoughLines import find_vanishing_points


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
        points = [cord for cord in choose_cord(image)]
    points.sort(key=lambda point: point[1], reverse=True)

    x = (points[0][0]+points[1][0]) // 2
    points = [(x, points[0][1]), (x,points[1][1])]
    return points

def draw_hight_on_image(img, start_point, end_point, height, placement, color):
    thickness = 2
    cv2.rectangle(img, start_point, end_point, color, thickness)

    # Write the object height on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text = f'{height:.2f}'
    outline_thickness = 4
    text_position = (start_point[0]+placement[0], start_point[1]+placement[1])  # Adjust the position above the object
    cv2.putText(img, text, text_position, font, font_scale, (0, 0, 0), outline_thickness + font_thickness)
    text_position = (start_point[0]+placement[0], start_point[1]+placement[1])  # Adjust the position above the object
    cv2.putText(img, text, text_position, font, font_scale, color, font_thickness)

def draw_real_height(img, start_point, height, placement, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text = f'{height:.2f}'
    outline_thickness = 4
    text_position = (
    start_point[0] + placement[0], start_point[1] + placement[1])  # Adjust the position above the object
    cv2.putText(img, text, text_position, font, font_scale, (0, 0, 0), outline_thickness + font_thickness)
    text_position = (
    start_point[0] + placement[0], start_point[1] + placement[1])  # Adjust the position above the object
    cv2.putText(img, text, text_position, font, font_scale, color, font_thickness)


def extract_points_from_text(file_path):
    try:
        with open(file_path, 'r') as file:
            text = file.readlines()
            h1 = tuple([int(float(num)) for num in text[1][:-2].strip("(").split(",")])
            h2 = tuple([int(float(num)) for num in text[2][:-2].strip("(").split(",")])
            vertical = tuple([int(float(num)) for num in text[4][:-2].strip("(").split(",")])
        return h1, h2, vertical


    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")


if __name__ == "__main__":
    # good images: 1080104, P1080106, 1080119

    # load the image
    path = r"pics/stairs1.jpg"
    img = cv2.imread(path)

    # run clustering
    # h1, h2, vertical_p = find_vanishing_points(img,
    #                       plot_detected=False,
    #                       iter=1000,
    #                       segmentetion_algorithm="LSD"
    #                       )

    # b = (190, 400)
    # r = (189, 278)
    # b0 = (445, 584)
    # H = 166
    # t0 = (451, 462)
    name = path.split('/')[-1].split(".")[0]
    input_path = f"input_files/{name}/{name}.txt"
    h1, h2, vertical = extract_points_from_text(input_path)
    b, r = choose_coordinates(image=img, message="Please choose the bottom and top points of your reference object.")
    print(f"the top point is {r} and the bottom point is {b}")
    b0, t0 = choose_coordinates(image=img, message="Please choose the bottom and top points of the object you "
                                              "wold like to measure.")
    H = int(input("Please enter the height of your reference object"))
    R = int(input("Please enter the height of the object you like to measure if unknown enter -1"))
    cross_res = calculate_H(b, r, H, b0, t0, h1, h2)
    algebric_res = calculate_height_using_camera_matrix(b, r, H, b0, t0, h1, h2, vertical)
    print(f"The measured height of your object using cross is: {cross_res}")
    print(f"The measured height of your object using camera matrix is: "
          f"{algebric_res}")
    print(f"the average is {(algebric_res+ cross_res)/2}")
    height_img = img.copy()
    draw_hight_on_image(height_img, t0, b0, cross_res, (5, 20), (255, 255, 0))
    draw_hight_on_image(height_img, r, b, H, (5, 20), (154,250,0))
    if R > 0:
        draw_real_height(height_img, b0,R, (0, -10), (200, 135, 0))
    mt0, mb0 = (t0[0]-5, t0[1]), (b0[0]-5, b0[1])
    draw_hight_on_image(height_img, mt0, mb0, algebric_res, (-117, 20), (255,0,255))
    # Display the image
    cv2.imshow('Object Height on Image', height_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
