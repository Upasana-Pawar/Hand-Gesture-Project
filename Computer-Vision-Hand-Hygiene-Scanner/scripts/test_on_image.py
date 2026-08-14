import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

# Get the project root directory (parent of scripts directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', 'gesture_classifier.pkl')
X_MAX_PATH = os.path.join(PROJECT_ROOT, 'model', 'X_max.npy')

# Load the trained model
model = joblib.load(MODEL_PATH)
X_max = np.load(X_MAX_PATH)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Extract keypoints from hand landmarks
def extract_keypoints(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()

# Test gesture classification on an image
def classify_gesture(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = extract_keypoints(hand_landmarks)
            keypoints = keypoints / X_max
            gesture = model.predict([keypoints])[0]

            # Display the gesture
            print(f"Gesture detected: {gesture}")
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Gesture Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
classify_gesture(os.path.join(PROJECT_ROOT, 'data', 'Images', '1.jpg'))
classify_gesture(os.path.join(PROJECT_ROOT, 'data', 'Images', '2.jpg'))
classify_gesture(os.path.join(PROJECT_ROOT, 'data', 'Images', '3.jpg'))
classify_gesture(os.path.join(PROJECT_ROOT, 'data', 'Images', '4.jpg'))
classify_gesture('data/Images/5.jpg')
classify_gesture('data/Images/6.jpg')
