import cv2
import numpy as np
import mediapipe as mp

# Constants
DRAW_COLOR = (255, 0, 0)  # Default blue color
ERASE_COLOR = (255, 255, 255)  # White color for erasing
BRUSH_THICKNESS = 5
ERASER_THICKNESS = 20

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize variables
canvas = None
current_color = DRAW_COLOR
last_point = None
brush_thickness = BRUSH_THICKNESS

# Function to display color and brush size buttons
def draw_buttons(frame):
    colors = {
        'Blue': ((10, 10), (80, 60), (255, 0, 0)),
        'Green': ((100, 10), (170, 60), (0, 255, 0)),
        'Red': ((190, 10), (260, 60), (0, 0, 255)),
        'Erase': ((280, 10), (350, 60), ERASE_COLOR),
        'Reset': ((370, 10), (440, 60), (0, 0, 0)),
        'Brush+': ((10, 80), (80, 130), (200, 200, 200)),
        'Brush-': ((100, 80), (170, 130), (200, 200, 200))
    }
    for label, ((x1, y1), (x2, y2), color) in colors.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return colors

# Start video capture
cap = cv2.VideoCapture(0)

# Create a blank canvas
ret, frame = cap.read()
if not ret:
    print("Unable to access the camera.")
    cap.release()
    cv2.destroyAllWindows()
    exit()
else:
    canvas = np.ones_like(frame) * 255  # White canvas

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Draw buttons
    colors = draw_buttons(frame)

    # Handle hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get fingertip coordinates (index finger tip)
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)

            # Check if fingertip is over a button
            for label, ((x1, y1), (x2, y2), color) in colors.items():
                if x1 < cx < x2 and y1 < cy < y2:
                    if label == 'Erase':
                        current_color = ERASE_COLOR
                    elif label == 'Brush+':
                        brush_thickness = min(brush_thickness + 5, 50)  # Increase brush size
                    elif label == 'Brush-':
                        brush_thickness = max(brush_thickness - 5, 5)  # Decrease brush size
                    elif label == 'Reset':
                        canvas = np.ones_like(frame) * 255  # Reset canvas
                    else:
                        current_color = color

            # Draw on canvas
            if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z < -0.1:  # Nearer to camera
                if last_point is not None:
                    cv2.line(canvas, last_point, (cx, cy), current_color,
                              ERASER_THICKNESS if current_color == ERASE_COLOR else brush_thickness)
                last_point = (cx, cy)
            else:
                last_point = None

    # Blend the canvas onto the frame
    blended = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Display the frame
    cv2.imshow('Whiteboard', blended)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()