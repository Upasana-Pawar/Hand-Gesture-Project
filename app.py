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
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    st.error("MediaPipe not available. Please wait for deployment to complete.")

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

def process_image(image_pil, model, X_max, hands, mp_hands, mp_drawing):
    """Process image and detect hand gestures."""
    # Convert PIL Image to numpy array
    image_np = np.array(image_pil)
    
    # Convert BGR to RGB if needed
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_rgb = image_np
    else:
        image_rgb = image_np
    
    results = hands.process(image_rgb)
    
    gestures_detected = []
    annotated_image = image_np.copy()
    
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
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
    
    return Image.fromarray(annotated_image.astype('uint8')), gestures_detected

# Main app
st.title("🖐️ Hand Gesture Recognition System")
st.write("Detect and classify hand washing gestures using computer vision")

# Check dependencies
if not HAS_MEDIAPIPE:
    st.error("⚠️ MediaPipe is not available. Please wait for deployment to complete and reload.")
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
st.sidebar.info("ℹ️ Upload an image to detect hand gestures")

# Image upload section
st.subheader("Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and process the image using PIL
    image_pil = Image.open(uploaded_file)
    
    # Process the image
    annotated_image_pil, gestures = process_image(image_pil, model, X_max, hands, mp_hands, mp_drawing)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image_pil, use_column_width=True)
    
    with col2:
        st.subheader("Detected Gestures")
        st.image(annotated_image_pil, use_column_width=True)
    
    # Display detected gestures
    st.subheader("Detection Results")
    if gestures:
        st.success(f"✅ Gestures detected: {', '.join(gestures)}")
        
        # Create a summary
        for i, gesture in enumerate(gestures, 1):
            st.info(f"Hand {i}: **{gesture}**")
    else:
        st.warning("⚠️ No hands detected in the image")

# Footer
st.divider()
st.markdown("""
### 📋 About
This application uses:
- **MediaPipe** for hand detection
- **Pillow** for image processing
- **Scikit-learn** for gesture classification
- **Streamlit** for the web interface

Trained to recognize hand washing gestures and sequences.
""")
