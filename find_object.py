import cv2
import numpy as np

from choose_coor import choose_cord

START_THRESHOLD = 150
MIN_POINTS_IN_CONTOURS = 100
OBJ_MESSAGE = "Pleas chose the highest and lowest points of the picture object."

#[(546, 316), (546, 441)]


def show_image(name, image):
	cv2.imshow(name, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def find_threshold(img_gray):
	threshold = START_THRESHOLD
	while threshold != -1:
		ret, thresh = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY_INV)
		show_image("Thresholded Image", thresh)
		threshold = int(input("Pleas enter new threshold, if you are happy with the one presented enter -1"))

	return thresh


def find_all_image_contours(image):
	img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresh = find_threshold(img_gray)
	contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
	return contours


def create_a_sorted_array_of_contours_near_a_point(contours, given_point):
	def distance_from_point(contour):
		if len(contour) < MIN_POINTS_IN_CONTOURS:
			return float("inf")
		# Calculate the centroid of the contour
		M = cv2.moments(contour)
		if M["m00"] != 0:
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			# Calculate distance from the given point to the centroid of the contour
			return np.sqrt((cX - given_point[0]) ** 2 + (cY - given_point[1]) ** 2)
		return float("inf")

	return sorted(contours, key=distance_from_point)


def cant_find_object(i, sorted_contours):
	ans = input("Can the last choice prove sufficient? [Y/n]")
	if not ans or ans.lower() == "y":
		return sorted_contours[0:i - 1]
	else:
		print("looks like the algorithm can't detect the object in your image.")
		return None


def find_all_relevant_contours(image, sorted_contours):
	for i in range(1, len(sorted_contours)):
		image_copy = image.copy()
		chosen_contours = sorted_contours[0:i]
		cv2.drawContours(image=image_copy, contours=chosen_contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
						 lineType=cv2.LINE_AA)
		show_image('Closest Contour', image_copy)
		ans = input("Dose this contours detect your object [Y/l/g]")
		if not ans or ans.lower() == "y":
			return chosen_contours
		if ans == "g":
			return cant_find_object(i, sorted_contours)
	return cant_find_object(i, sorted_contours)


def manually_pick_two_points(image, message, given_point):
	print(message)
	coords = choose_cord(image)[0:2]
	coords.sort(key=lambda item: item[1])
	return [(given_point[0], coords[0][1]), (given_point[0], coords[1][1])]

def find_object_height(chosen_contours, given_point):
	y_cords = np.array([cord[0,1] for contours in chosen_contours for cord in contours])
	return [(given_point[0], y_cords.min()), (given_point[0], y_cords.max())]



def run():
	image = cv2.imread('Pictures/img1.jpg')
	image = cv2.resize(image, (960, 540))
	contours = find_all_image_contours(image)
	given_point = choose_cord(image)[0]
	sorted_contours = create_a_sorted_array_of_contours_near_a_point(contours, given_point)
	chosen_contours = find_all_relevant_contours(image, sorted_contours)
	if chosen_contours:
		return find_object_height(chosen_contours, given_point)
	else:
		return manually_pick_two_points(image, OBJ_MESSAGE, given_point)


if __name__ == "__main__":
	print(run())
