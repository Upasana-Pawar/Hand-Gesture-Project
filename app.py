import os
import sys
import streamlit as st
from PIL import Image
import numpy as np
import joblib
import time
import cv2
from pathlib import Path

# Set page config FIRST
st.set_page_config(
    page_title="Hand Hygiene Scanner - Real-Time",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import MediaPipe
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except (ImportError, OSError) as e:
    HAS_MEDIAPIPE = False

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent / "Computer-Vision-Hand-Hygiene-Scanner"
MODEL_PATH = PROJECT_ROOT / 'model' / 'gesture_classifier.pkl'
X_MAX_PATH = PROJECT_ROOT / 'model' / 'X_max.npy'

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load(str(MODEL_PATH))
        X_max = np.load(str(X_MAX_PATH))
        return model, X_max
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Initialize MediaPipe Hands
@st.cache_resource
def init_mediapipe():
    if not HAS_MEDIAPIPE:
        return None, None, None
    
    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        mp_drawing = mp.solutions.drawing_utils
        return hands, mp_hands, mp_drawing
    except Exception as e:
        return None, None, None

def extract_keypoints(landmarks):
    """Extract hand landmarks (X, Y, Z coordinates)."""
    keypoints = []
    for i, landmark in enumerate(landmarks.landmark):
        keypoints.extend([landmark.x, landmark.y, landmark.z])
    return np.array(keypoints)

def process_frame(frame, model, X_max, hands, mp_hands, mp_drawing, recent_predictions, RECENT_WINDOW=5, CONFIRM_THRESHOLD=3):
    """Process a single frame and detect gestures."""
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    predicted_gesture = None
    frame_copy = frame.copy()
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract the landmarks and make predictions
            keypoints = extract_keypoints(hand_landmarks)
            
            # Normalize using training X_max factors
            if X_max is not None and X_max.shape[0] == keypoints.shape[0]:
                keypoints = keypoints / X_max
            
            predicted_gesture = model.predict([keypoints])[0]
            
            # Add to recent predictions buffer and apply simple confirmation
            recent_predictions.append(predicted_gesture)
            if len(recent_predictions) > RECENT_WINDOW:
                recent_predictions.pop(0)
            
            # Display the detected gesture on the frame
            cv2.putText(frame_copy, f'Gesture: {predicted_gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame_copy, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    return frame_copy, predicted_gesture, recent_predictions

# Main app
st.title("🖐️ Hand Hygiene Scanner - Real-Time")
st.write("Real-time hand washing gesture recognition")

# Load model and MediaPipe
model, X_max = load_model()
hands, mp_hands, mp_drawing = init_mediapipe()

# Check dependencies
if model is None or X_max is None:
    st.error("❌ Could not load the gesture model.")
    st.stop()

if not HAS_MEDIAPIPE or hands is None:
    st.error("❌ MediaPipe not available. Please install it: pip install mediapipe")
    st.stop()

# Required hand washing gestures sequence
REQUIRED_GESTURES = [
    "Start",      
    "Rubbing Palm",           
    "Scrubbing Fingers",     
    "Rinsing Hands",        
    "End"                   
]

# Session state for tracking
if 'performed_gestures' not in st.session_state:
    st.session_state.performed_gestures = []
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'final_time' not in st.session_state:
    st.session_state.final_time = None
if 'recent_predictions' not in st.session_state:
    st.session_state.recent_predictions = []

# Sidebar controls
st.sidebar.title("⚙️ Settings")
st.sidebar.write("### Hand Washing Sequence:")
for i, gesture in enumerate(REQUIRED_GESTURES, 1):
    if gesture in st.session_state.performed_gestures:
        st.sidebar.write(f"✅ {i}. {gesture}")
    else:
        st.sidebar.write(f"⭕ {i}. {gesture}")

if st.sidebar.button("🔄 Reset Session"):
    st.session_state.performed_gestures = []
    st.session_state.start_time = None
    st.session_state.final_time = None
    st.session_state.recent_predictions = []
    st.rerun()

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📹 Live Feed")
    camera_placeholder = st.empty()
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Could not access webcam. Please check permissions.")
        st.stop()
    
    # Get camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    RECENT_WINDOW = 5
    CONFIRM_THRESHOLD = 3
    frame_count = 0
    max_frames = 1000  # Process up to 1000 frames
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        
        if not ret:
            st.error("Failed to read from camera")
            break
        
        # Flip frame for selfie view
        frame = cv2.flip(frame, 1)
        
        # Process frame
        processed_frame, predicted_gesture, st.session_state.recent_predictions = process_frame(
            frame, model, X_max, hands, mp_hands, mp_drawing, 
            st.session_state.recent_predictions, RECENT_WINDOW, CONFIRM_THRESHOLD
        )
        
        # Confirm gesture if it appears multiple times in recent window
        if predicted_gesture and len(st.session_state.recent_predictions) > 0:
            if st.session_state.recent_predictions.count(predicted_gesture) >= CONFIRM_THRESHOLD:
                if predicted_gesture not in st.session_state.performed_gestures:
                    st.session_state.performed_gestures.append(predicted_gesture)
                
                # Start timer on first gesture
                if st.session_state.start_time is None:
                    st.session_state.start_time = time.time()
        
        # Calculate elapsed time
        if st.session_state.start_time is not None:
            if len(st.session_state.performed_gestures) == len(REQUIRED_GESTURES):
                if st.session_state.final_time is None:
                    st.session_state.final_time = time.time() - st.session_state.start_time
                elapsed = st.session_state.final_time
            else:
                elapsed = time.time() - st.session_state.start_time
            
            cv2.putText(processed_frame, f"Time: {elapsed:.2f}s", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Check completion status
        if len(st.session_state.performed_gestures) == len(REQUIRED_GESTURES):
            cv2.putText(processed_frame, "✅ Well done! Your hands are clean.", 
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            missing = [g for g in REQUIRED_GESTURES if g not in st.session_state.performed_gestures]
            text = f"Missing: {', '.join(missing[:2])}"
            cv2.putText(processed_frame, text, 
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display frame
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        camera_placeholder.image(processed_frame_rgb, use_container_width=True)
        
        frame_count += 1
        
        # Check for stop condition
        if frame_count >= max_frames:
            break
    
    cap.release()

with col2:
    st.subheader("📊 Progress")
    
    st.write(f"**Gestures Completed:** {len(st.session_state.performed_gestures)}/{len(REQUIRED_GESTURES)}")
    
    if st.session_state.performed_gestures:
        st.write("**Detected Gestures:**")
        for gesture in st.session_state.performed_gestures:
            st.write(f"✅ {gesture}")
    
    if st.session_state.final_time:
        st.write(f"**Total Time:** {st.session_state.final_time:.2f}s")
        st.success("🎉 Hand washing complete!")
    elif st.session_state.start_time:
        elapsed = time.time() - st.session_state.start_time
        st.write(f"**Elapsed Time:** {elapsed:.2f}s")
    
    missing_gestures = [g for g in REQUIRED_GESTURES if g not in st.session_state.performed_gestures]
    if missing_gestures:
        st.warning(f"**Still needed:** {', '.join(missing_gestures)}")

# Footer
st.divider()
st.markdown("""
### 📋 How to Use:
1. **Allow webcam access** when prompted
2. **Position your hands** in front of the camera
3. **Perform the required gestures** in sequence
4. The app will track your progress and show the total time
5. **Click "Reset Session"** to start over

### 🖐️ Required Gestures:
1. Start
2. Rubbing Palm
3. Scrubbing Fingers
4. Rinsing Hands
5. End
""")

