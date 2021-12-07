from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import face_recognition




ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-s", "--show", type=int, default=0,
	help="whether or not to show result video")
ap.add_argument("-f", "--file", type=str, default="result.txt",
	help="file to store result")
args = vars(ap.parse_args())


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] Reading Video...")
vs = cv2.VideoCapture(args["video"])
show_video = args["show"]
outF = open(args["file"], "w")

total = 0
# loop over the frames from the video stream
while True:
    # Grab a single frame of video
	ret, frame = vs.read()
    # Convert the image from BGR color (which OpenCV uses) to RGB   
    # color (which face_recognition uses)
	rgb_frame = frame[:, :, ::-1]
    # Find all the faces in the current frame of video
	face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

	if face_landmarks_list:
		total += 1
		timestamps = vs.get(cv2.CAP_PROP_POS_MSEC)
		print("[INFO] Faces found at %s fps"%timestamps)
		if show_video:
			for face_landmarks in face_landmarks_list:
					for facial_feature in face_landmarks.keys():
							for (x,y) in face_landmarks[facial_feature]:
								if show_video:
									cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
			cv2.putText(frame, str(timestamps), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF
		face_landmarks_list.append({"timestamp": timestamps})
		outF.write(str(face_landmarks_list))
		outF.write("\n")

outF.close()
if show_video:
	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()


