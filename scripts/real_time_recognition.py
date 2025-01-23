import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

# Load the trained model for gesture classification
model = joblib.load('model/gesture_classifier.pkl')

# Initialize MediaPipe Hands for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Define the required sequence of gestures for hand washing based on landmarks
REQUIRED_GESTURES = [
    "Start",      
    "Rubbing Palm",           
    "Scrubbing Fingers",     
    "Rinsing Hands",        
    "End"                   
]

# To track the progress and gestures performed
performed_gestures = []
start_time = None
final_time = None

def extract_keypoints(landmarks):
    """Extract hand landmarks (X, Y, Z coordinates) from the hand landmarks."""
    keypoints = []
    for i, landmark in enumerate(landmarks.landmark):
        keypoints.extend([landmark.x, landmark.y, landmark.z])
    return np.array(keypoints)  # Flattened keypoints array

def real_time_recognition():
    # Start video capture
    video = cv2.VideoCapture(0)
    global start_time, final_time
    last_gesture = "No gesture detected"

    with hands as hand_detector:
        while True:
            ret, image = video.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Convert the image to RGB for MediaPipe processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hand_detector.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract the landmarks and make predictions
                    keypoints = extract_keypoints(hand_landmarks)
                    predicted_gesture = model.predict([keypoints])[0]

                    # Only add the gesture to performed_gestures if it's not already there
                    if predicted_gesture not in performed_gestures:
                        performed_gestures.append(predicted_gesture)

                    # Start the timer when the first gesture is detected
                    if start_time is None:
                        start_time = time.time()

                    # Display the detected gesture on the screen
                    cv2.putText(image, f'Gesture: {predicted_gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # If the first gesture has been performed, calculate elapsed time
            if start_time is not None:
                # Check if all gestures are completed
                if len(performed_gestures) == len(REQUIRED_GESTURES):
                    # Once all gestures are done, freeze the time
                    if final_time is None:
                        final_time = time.time() - start_time  # Freeze the timer once done

                if final_time is None:
                    elapsed_time = time.time() - start_time
                    cv2.putText(image, f"Time: {elapsed_time:.2f} seconds", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(image, f"Total Time: {final_time:.2f} seconds", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Check if all required gestures are performed in the correct order
            if len(performed_gestures) == len(REQUIRED_GESTURES):  # Check if all gestures have been performed
                cv2.putText(image, f"Well done! Your hands are clean.", 
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:  # Some gestures are missing
                missing_gestures = [gesture for gesture in REQUIRED_GESTURES if gesture not in performed_gestures]
                cv2.putText(image, f"Missing Gestures: {', '.join(missing_gestures)}", 
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display the frame
            cv2.imshow("Hand Gesture Recognition", image)

            # Break loop on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Set window to fullscreen (optional)
    cv2.namedWindow("Hand Gesture Recognition", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Hand Gesture Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    real_time_recognition()
