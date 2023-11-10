import cv2

IMAGE_TO_READ = 'Pictures/img1.jpg'


def choose_cord(img):
	chosen_point = []

	def click_event(event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			chosen_point.append((x, y))

	cv2.imshow('image', img)

	cv2.setMouseCallback('image', click_event)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return chosen_point  # Return the list of chosen point coordinates


if __name__ == "__main__":
	img = cv2.imread(IMAGE_TO_READ)
	img = cv2.resize(img, (960, 540))
	chosen_points = choose_cord(img)
	print("Chosen Points:", chosen_points)
