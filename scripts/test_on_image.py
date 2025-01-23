import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load the trained model
model = joblib.load('model/gesture_classifier.pkl')
X_max = np.load('model/X_max.npy')

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
classify_gesture('data/Images/1.jpg')
classify_gesture('data/Images/2.jpg')
classify_gesture('data/Images/3.jpg')
classify_gesture('data/Images/4.jpg')
classify_gesture('data/Images/5.jpg')
classify_gesture('data/Images/6.jpg')
