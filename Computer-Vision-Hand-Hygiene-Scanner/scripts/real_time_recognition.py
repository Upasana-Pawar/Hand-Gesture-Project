import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import argparse
import sys
import os

# Get the project root directory (parent of scripts directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', 'gesture_classifier.pkl')
X_MAX_PATH = os.path.join(PROJECT_ROOT, 'model', 'X_max.npy')

# Load the trained model for gesture classification
model = joblib.load(MODEL_PATH)
# Load normalization factors saved during training
try:
    X_max = np.load(X_MAX_PATH)
except Exception:
    X_max = None

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
# Simple smoothing: recent predictions buffer to reduce flicker
recent_predictions = []
RECENT_WINDOW = 5
CONFIRM_THRESHOLD = 3

def extract_keypoints(landmarks):
    """Extract hand landmarks (X, Y, Z coordinates) from the hand landmarks."""
    keypoints = []
    for i, landmark in enumerate(landmarks.landmark):
        keypoints.extend([landmark.x, landmark.y, landmark.z])
    return np.array(keypoints)  # Flattened keypoints array

def find_working_camera(preferred_index=None, max_search=6):
    """Try to open the preferred camera index; if not available, scan indices 0..max_search-1 and return the first that opens."""
    backends = [cv2.CAP_ANY]
    # On Windows, try DirectShow backend which often works with external webcams
    if sys.platform.startswith("win"):
        backends.insert(0, cv2.CAP_DSHOW)

    def try_open(index):
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap is None or not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                continue

            # quick frame read test
            ok, _ = cap.read()
            if not ok:
                try:
                    cap.release()
                except Exception:
                    pass
                continue

            return cap, index
        return None, None

    # Try preferred index first
    if preferred_index is not None:
        cap, idx = try_open(preferred_index)
        if cap is not None:
            return cap, idx

    # Fall back to scanning indices
    for i in range(max_search):
        cap, idx = try_open(i)
        if cap is not None:
            return cap, idx

    return None, None


def real_time_recognition(camera_index=None):
    global start_time, final_time
    
    # Start video capture (selected camera or auto-detect)
    cap, used_index = find_working_camera(preferred_index=camera_index)
    if cap is None:
        print("Warning: initial camera open attempt failed. Trying more backends for the requested index...")
        # Try more aggressive backends list per-platform
        alt_backends = [
            cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY
        ]
        cap = None
        used_index = camera_index if camera_index is not None else 0
        for b in alt_backends:
            try:
                temp = cv2.VideoCapture(used_index, b)
                if temp is not None and temp.isOpened():
                    ok, _ = temp.read()
                    if ok:
                        cap = temp
                        print(f"Opened camera index {used_index} with backend {b}")
                        break
                try:
                    temp.release()
                except Exception:
                    pass
            except Exception:
                continue

        if cap is None:
            print("Error: no camera could be opened. Check camera connection, drivers, and OS permissions.")
            return
        print(f"Using camera index: {used_index}")
    else:
        print(f"Using camera index: {used_index}")

    video = cap

    with hands as hand_detector:
        while True:
            ret, image = video.read()
            if not ret or image is None:
                # Attempt to reopen the camera once
                print("Failed to grab frame; attempting to reopen camera...")
                try:
                    video.release()
                except Exception:
                    pass
                cap2, idx2 = find_working_camera(preferred_index=used_index)
                if cap2 is not None:
                    video = cap2
                    print(f"Reopened camera index {idx2}")
                    ret, image = video.read()
                    if not ret:
                        print("Still failed to read from camera after reopen. Exiting.")
                        break
                else:
                    print("Unable to reopen camera. Exiting.")
                    break

            # Convert the image to RGB for MediaPipe processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hand_detector.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract the landmarks and make predictions
                    keypoints = extract_keypoints(hand_landmarks)

                    # Normalize using training X_max factors if available
                    if X_max is not None and X_max.shape[0] == keypoints.shape[0]:
                        keypoints = keypoints / X_max

                    predicted_gesture = model.predict([keypoints])[0]

                    # Add to recent predictions buffer and apply simple confirmation
                    recent_predictions.append(predicted_gesture)
                    if len(recent_predictions) > RECENT_WINDOW:
                        recent_predictions.pop(0)

                    # Confirm gesture if it appears at least CONFIRM_THRESHOLD times in the recent window
                    if recent_predictions.count(predicted_gesture) >= CONFIRM_THRESHOLD:
                        if predicted_gesture not in performed_gestures:
                            performed_gestures.append(predicted_gesture)

                        # Start the timer when the first confirmed gesture is detected
                        if start_time is None:
                            start_time = time.time()

                    # Display the detected gesture on the screen (most recent)
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
    parser = argparse.ArgumentParser(description="Real-time hand gesture recognition")
    parser.add_argument("--camera-index", type=int, default=None, help="Preferred camera index (integer)")
    args = parser.parse_args()

    real_time_recognition(camera_index=args.camera_index)
