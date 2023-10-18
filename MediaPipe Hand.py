from __future__ import print_function
from imutils.video import WebcamVideoStream
import cv2
import mediapipe as mp
import time
import socket
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

prev_frame_time = 0
new_frame_time = 0
width, height = 1280, 720

# For webcam input:
cap = WebcamVideoStream(src=0).start()

# Communication with UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 1234)

data = []
# Initiate hand model
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
  while True:
    image = cap.read()
    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # FPS message
    cv2.putText(image, "FPS: {:.2f}".format(fps), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
    
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #print('hand_landmarks:', hand_landmarks)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                      )
            for landmark in hand_landmarks.landmark:
                x = landmark.x * width
                y = landmark.y * height
                z = landmark.z * width
                data.append(x)
                data.append(height - y)
                data.append(z)
    # Pass the landmarks to unity
    sock.sendto(str.encode(str(data)),serverAddressPort)
    data = []
    # Flip the image horizontally for a selfie-view display.
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)    
    image = cv2.resize(image, (0, 0), None, 0.5, 0.5)      
    cv2.imshow('Webcam', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.stop()
cv2.destroyAllWindows()
