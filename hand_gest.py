# import necessary packages

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)


# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]

    # show the prediction on the frame
    text = className
    text_position = (10, 50)
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 1
    text_color = (0, 0, 255)  # Red color
    text_thickness = 2
    text_background_color = (255, 255, 255)  # White color

    # Get the width and height of the text box
    (text_width, text_height), _ = cv2.getTextSize(text, text_font, text_scale, text_thickness)

    # Calculate the coordinates for the text box
    box_coords = ((text_position[0], text_position[1] - text_height - 10), 
                (text_position[0] + text_width + 10, text_position[1] + 10))

    # Draw the text box with white background
    cv2.rectangle(frame, box_coords[0], box_coords[1], text_background_color, cv2.FILLED)

    # Draw the text on the frame
    cv2.putText(frame, text, text_position, text_font, text_scale, text_color, text_thickness, cv2.LINE_AA)
    # Show the final output
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()