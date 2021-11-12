import cv2
import time


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

# Source data : Video File
video_file = 'test5.mp4'
# Read the source video file
cap = cv2.VideoCapture(video_file)

# pre trained classifiers
car_classifier = 'cars.xml'
pedestrian_classifier = 'pedestrian.xml'

# Classified Trackers
car_tracker = cv2.CascadeClassifier(car_classifier)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_classifier)


start = time.time()
while True:
    # start reading video file frame by frame like an image

	(read_successful, frame) = cap.read()
	
	#convert to grey scale image
	frame_temp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	start = time.time()
	frame_temp = rescale_frame(frame, percent=40)
	# Detect Cars, Pedestrians
	cars = car_tracker.detectMultiScale(frame_temp,1.1,2)
	print("Car Detected at loactions: ")
	print(cars)
	pedestrians = pedestrian_tracker.detectMultiScale(frame_temp,1.1,2)
	print("Pedestrain Detected at loactions: ")
	print (pedestrians)
	end = time.time()

	# Draw rectangle around the cars
	for (x, y, w, h) in cars:
		cv2.rectangle(frame_temp, (x, y), (x + w, y + h), (0, 255, 255), 2)
		cv2.putText(frame_temp, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	# Draw square around the pedestrians
	for (x, y, w, h) in pedestrians:
		cv2.rectangle(frame_temp, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv2.putText(frame_temp, 'Human', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	# display the imapge with the face spotted
	cv2.imshow('Detect Objects On Road',frame_temp)
	# capture key
	key = cv2.waitKey(1)
	# Stop incase Esc is pressed
	if key == 27:
		break
print ("The time to Detect the obejct %s seconds" %(end - start))
# Release video capture object
cap.release()


