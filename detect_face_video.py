import cv2
import numpy as np

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 2 options: one is use your webcam of your computer, the other is input the video
# Option 1: To capture video from webcam. 
# cap = cv2.VideoCapture(0)

# Option2: To use a video file as input 
cap = cv2.VideoCapture('production2.mp4')

def rescale_frame(frame, percent=75):
    scale_percent=75
    width=int(frame.shape[1]*scale_percent/100)
    height=int(frame.shape[0]*scale_percent/100)
    dim=(width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

while True:
    # Read the frame
    _, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display
    frame=rescale_frame(frame, percent=40)
    cv2.imshow('frame', frame)

    #Stop if escape key is pressed
    k = cv2.waitKey(100) & 0xff
    if k==27:
        break
        
# Release the VideoCapture object
cap.release()