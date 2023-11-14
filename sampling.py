# importing the module
import cv2
import numpy as np
from utilities import *
from geometry import *


click_count = 0
points = []

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):

	global click_count, points

	# checking for left mouse clicks
	if event == cv2.EVENT_LBUTTONDOWN:

		click_count += 1

		# save the coordinates in list
		points.append((x, y))
		print((x, y))

		# displaying the counter of the point
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img, str(click_count), (x,y), font,
					1, (255, 0, 0), 2)
		cv2.imshow('image', img)

	# checking for right mouse clicks
	if event==cv2.EVENT_RBUTTONDOWN:

		# displaying the coordinates
		# on the Shell
		print(f"[{x},{y}]", end=", ")

		# displaying the coordinates
		# on the image window
		font = cv2.FONT_HERSHEY_SIMPLEX
		b = img[y, x, 0]
		g = img[y, x, 1]
		r = img[y, x, 2]
		cv2.putText(img, str(b) + ',' +
					str(g) + ',' + str(r),
					(x,y), font, 1,
					(255, 255, 0), 2)
		cv2.imshow('image', img)


def display_and_wait(img):
	cv2.imshow('image', img)
	cv2.setMouseCallback('image', click_event)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == "__main__":

	# reading the image
	img = cv2.imread(r"pics/beer3.jpeg", 1)
	img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))

	# get inputs for finding the horizon
	display_and_wait(img)

	# find the horizon using 4 inputs
	points = [(338, 636), (718, 563), (464, 494), (257, 515)]
	a1, a2, a3, a4 = [to_homogenous(p) for p in points]
	h1 = find_intersection(a1, a2, a4, a3)
	h2 = find_intersection(a1, a4, a2, a3)

	# rotate the image using the horizon. also draw the line on the image
	rotation_angle = get_rotation_angle(h1, h2)
	img = rotate_image(img, rotation_angle)
	h1, h2 = rotate_the_horizon(h1, h2, img, rotation_angle)
	draw_line_on_img(img, h1, h2)

	# get input for the rest of the points
	display_and_wait(img)

	b = (624, 643)
	r = (623, 216)

	b, r = perp_points(b, r)
	H_box = 6.5
	H_beer = 22.5

	b0_far, t0_far = (86, 431), (94, 177)
	b0_close, t0_close = (457, 441), (458, 147)
	b0, t0 = perp_points(b0_close, t0_close)

	W_pred = calculate_W(b, r, H_beer, b0, t0, h1, h2)
	W = 74
	print("real: ", W)
	print("pred: ", W_pred)

	print("the horizon :",h1, h2)

	# line: b -> b0 -> v
	v = find_intersection(b, b0, h1, h2)
	draw_line_on_img(img, to_non_homogenous(b), to_non_homogenous(v))

	# line: v -> t0 -> t
	t = find_intersection(v, t0, b, r)
	draw_line_on_img(img, to_non_homogenous(v), to_non_homogenous(t))

	# line: t -> r -> b
	draw_line_on_img(img, to_non_homogenous(t), to_non_homogenous(b))

	for p, txt in zip ([b, r, t, t0, b0, v], ["b", "r", "t", "t0", "b0", "v"]):
		draw_point_on_img(img, p, txt)

	cv2.waitKey(0)
	cv2.destroyAllWindows()


