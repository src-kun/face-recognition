import cv2
import dlib
from recognition import face_id_by_frame, face_id_by_frame2
import face_recognition
# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Open the input movie file
#video_capture = cv2.VideoCapture("hamilton_clip.mp4")
#length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('output.avi', fourcc, 29.97, (640, 360))

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0
face_detection = dlib.get_frontal_face_detector()
description = dlib.face_recognition_model_v1('model/face_model.dat')
face_alignment = dlib.shape_predictor('model/face_alignment.dat')
process_this_frame = True

def read_names_lib(filename):
	f = open(filename,'r')
	r = f.readlines()
	f.close()
	return r
last_names = []
from datetime import datetime
while True:
	# Grab a single frame of video
	ret, frame = video_capture.read()
	frame_number += 1
	
	# Quit when the input video file ends
	if not ret:
		break
	face_names = []
	start = datetime.now()
	rgb_frame, face_names = face_id_by_frame(frame, last_names)
	print face_names
	end = datetime.now()
	print (end - start).seconds
	
	last_names = face_names[:]
	 # Resize frame of video to 1/4 size for faster face recognition processing
	small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

	# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
	rgb_small_frame = small_frame[:, :, ::-1]
	face_locations = face_recognition.face_locations(rgb_small_frame)
	
	# Display the results
	for (top, right, bottom, left), name in zip(face_locations, face_names):
		# Scale back up face locations since the frame we detected in was scaled to 1/4 size
		top *= 4
		right *= 4
		bottom *= 4
		left *= 4

		# Draw a box around the face
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

		# Draw a label with a name below the face
		cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

	# Display the resulting image
	#cv2.imshow('Video', frame)

	# Hit 'q' on the keyboard to quit!
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
