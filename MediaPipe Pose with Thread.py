from __future__ import print_function
from imutils.video import WebcamVideoStream
import cv2
import mediapipe as mp
import socket
import time

prev_frame_time = 0
new_frame_time = 0

mp_drawing = mp.solutions.drawing_utils

mp_pose = mp.solutions.pose

width, height = 1280, 720

cap = WebcamVideoStream(src=0).start()

# Communication with UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

serverAddressPort = ("127.0.0.1", 1234)

data = []

# Initiate hand model
with mp_pose.Pose(static_image_mode=False,
                  model_complexity=1,
                  enable_segmentation=True,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5,
                  smooth_segmentation=True,
                  smooth_landmarks=True
                  ) as pose:
    while True:
    
        # Read
        frame = cap.read()
        
        # Calculate FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        
        # FPS message
        #cv2.putText(frame, "FPS: {:.2f}".format(fps), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        # Make Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                  )
        
        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Get the pose landmarks 
        for idnum, poses in enumerate (results.pose_world_landmarks.landmark):
            #print(idnum)
            #print(poses)
            x = poses.x * width
            y = poses.y * height
            z = poses.z * width
            data.append(x)
            data.append(height - y)
            data.append(z)
        #print(data)
        
        # Pass the landmarks to unity
        sock.sendto(str.encode(str(data)),serverAddressPort)
        
        data = []
        image = cv2.resize(image, (0, 0), None, 0.5, 0.5)         
        cv2.imshow('Webcam', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
cap.stop()
