import streamlit as st
import numpy as np
import joblib
import os
from pathlib import Path
import tempfile
from PIL import Image
import io

# Import with error handling
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    st.error("OpenCV not available. Please check deployment logs.")

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    st.error("MediaPipe not available. Please check deployment logs.")

# Set page config
st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="🖐️",
    layout="wide"
)

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent / "Computer-Vision-Hand-Hygiene-Scanner"
MODEL_PATH = PROJECT_ROOT / 'model' / 'gesture_classifier.pkl'
X_MAX_PATH = PROJECT_ROOT / 'model' / 'X_max.npy'

# Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load(str(MODEL_PATH))
    X_max = np.load(str(X_MAX_PATH))
    return model, X_max

# Initialize MediaPipe Hands
@st.cache_resource
def init_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils
    return hands, mp_hands, mp_drawing

def extract_keypoints(landmarks):
    """Extract hand landmarks (X, Y, Z coordinates)."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()

def process_image(image, model, X_max, hands, mp_hands, mp_drawing):
    """Process image and detect hand gestures."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    gestures_detected = []
    annotated_image = image.copy()
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = extract_keypoints(hand_landmarks)
            keypoints_normalized = keypoints / X_max
            gesture = model.predict([keypoints_normalized])[0]
            gestures_detected.append(gesture)
            
            # Draw landmarks on the image
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
    
    return annotated_image, gestures_detected

# Main app
st.title("🖐️ Hand Gesture Recognition System")
st.write("Detect and classify hand washing gestures using computer vision")

# Check dependencies
if not HAS_CV2 or not HAS_MEDIAPIPE:
    st.error("⚠️ Required dependencies are not available. Please wait for the deployment to complete and reload the page.")
    st.stop()

# Load model and MediaPipe
try:
    model, X_max = load_model()
    hands, mp_hands, mp_drawing = init_mediapipe()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar for options
st.sidebar.title("Options")
option = st.sidebar.radio(
    "Choose input method:",
    ["📷 Upload Image", "🎥 Webcam"]
)

if option == "📷 Upload Image":
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        # Read and process the image
        image = cv2.imread(tmp_path)
        
        if image is not None:
            # Process the image
            annotated_image, gestures = process_image(image, model, X_max, hands, mp_hands, mp_drawing)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, channels="BGR", use_column_width=True)
            
            with col2:
                st.subheader("Detected Gestures")
                st.image(annotated_image, channels="BGR", use_column_width=True)
            
            # Display detected gestures
            st.subheader("Detection Results")
            if gestures:
                st.success(f"✅ Gestures detected: {', '.join(gestures)}")
                
                # Create a summary
                for i, gesture in enumerate(gestures, 1):
                    st.info(f"Hand {i}: **{gesture}**")
            else:
                st.warning("⚠️ No hands detected in the image")
        
        # Clean up temporary file
        os.unlink(tmp_path)

elif option == "🎥 Webcam":
    st.subheader("Real-time Webcam Detection")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.write("### Settings")
        confidence = st.slider("Detection Confidence", 0.0, 1.0, 0.7, 0.05)
        max_frames = st.number_input("Max frames to process", 1, 300, 100)
    
    with col1:
        stframe = st.empty()
        info_placeholder = st.empty()
    
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        frame_count = 0
        gesture_history = []
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                st.error("Failed to capture frame from webcam")
                break
            
            # Flip the frame for selfie view
            frame = cv2.flip(frame, 1)
            
            # Process frame
            annotated_frame, gestures = process_image(frame, model, X_max, hands, mp_hands, mp_drawing)
            
            # Add text to frame
            if gestures:
                gesture_history.extend(gestures)
                text = f"Gesture: {', '.join(gestures)}"
                cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Display info
            with info_placeholder.container():
                st.write(f"**Frames Processed:** {frame_count + 1}/{max_frames}")
                if gesture_history:
                    st.write(f"**Latest Gesture:** {gesture_history[-1]}")
            
            frame_count += 1
        
        cap.release()
        
        # Summary
        st.subheader("Session Summary")
        if gesture_history:
            st.write(f"**Total gestures detected:** {len(gesture_history)}")
            st.write(f"**Unique gestures:** {', '.join(set(gesture_history))}")
        else:
            st.info("No gestures detected in this session")
    else:
        st.error("Unable to access webcam. Please check your camera permissions.")

# Footer
st.divider()
st.markdown("""
### 📋 About
This application uses:
- **MediaPipe** for hand detection
- **OpenCV** for image processing
- **Scikit-learn** for gesture classification
- **Streamlit** for the web interface

Trained to recognize hand washing gestures and sequences.
""")
