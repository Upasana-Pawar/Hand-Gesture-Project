import os
import sys

# Disable OpenGL and GUI libraries before any imports
os.environ['DISPLAY'] = ''
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import streamlit as st
from PIL import Image
import numpy as np
import joblib
from pathlib import Path

# Set page config FIRST before any other st calls
st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="🖐️",
    layout="wide"
)

# Import with error handling
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except (ImportError, OSError) as e:
    HAS_MEDIAPIPE = False
    error_msg = str(e)
    if "libGL" in error_msg or "cannot open shared object" in error_msg:
        st.warning(f"⚠️ Graphics libraries not available on this system. Please wait...")
    else:
        st.warning(f"⚠️ MediaPipe initialization: {error_msg}")

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
        return None, None

# Initialize MediaPipe Hands
@st.cache_resource
def init_mediapipe():
    if not HAS_MEDIAPIPE:
        return None, None, None
    
    try:
        mp_hands = mp.solutions.hands
        # Use CPU only, disable GPU
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        mp_drawing = mp.solutions.drawing_utils
        return hands, mp_hands, mp_drawing
    except Exception as e:
        st.warning(f"Could not initialize MediaPipe: {e}")
        return None, None, None

def extract_keypoints(landmarks):
    """Extract hand landmarks (X, Y, Z coordinates)."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()

def process_image(image_pil, model, X_max, hands, mp_hands, mp_drawing):
    """Process image and detect hand gestures."""
    if model is None:
        return image_pil, []
    
    # Convert PIL Image to numpy array
    image_np = np.array(image_pil)
    
    # If no hand detection available, just return original image
    if hands is None or mp_hands is None:
        return image_pil, ["Hand detection unavailable"]
    
    # Convert to RGB if needed
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_rgb = image_np
    else:
        image_rgb = image_np
    
    try:
        results = hands.process(image_rgb)
    except Exception as e:
        return Image.fromarray(image_np.astype('uint8')), []
    
    gestures_detected = []
    annotated_image = image_np.copy()
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = extract_keypoints(hand_landmarks)
            keypoints_normalized = keypoints / X_max
            gesture = model.predict([keypoints_normalized])[0]
            gestures_detected.append(gesture)
            
            # Draw landmarks on the image
            try:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
            except:
                pass
    
    return Image.fromarray(annotated_image.astype('uint8')), gestures_detected

# Main app
st.title("🖐️ Hand Gesture Recognition System")
st.write("Detect and classify hand washing gestures using computer vision")

# Load model and MediaPipe
model, X_max = load_model()
hands, mp_hands, mp_drawing = init_mediapipe()

# Check dependencies
if model is None or X_max is None:
    st.error("❌ Could not load the gesture model. Please try refreshing the page.")
    st.stop()

if not HAS_MEDIAPIPE or hands is None:
    st.info("ℹ️ Hand detection is currently unavailable, but you can still upload images to see predictions.")
    model_available = True
    hands = None
else:
    model_available = True

# Sidebar for options
st.sidebar.title("Options")
st.sidebar.info("ℹ️ Upload an image to detect hand gestures")

# Image upload section
st.subheader("Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
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
            st.subheader("Analysis Result")
            st.image(annotated_image_pil, use_column_width=True)
        
        # Display detected gestures
        st.subheader("Detection Results")
        if gestures and gestures[0] != "Hand detection unavailable":
            st.success(f"✅ Gestures detected: {', '.join(gestures)}")
            
            # Create a summary
            for i, gesture in enumerate(gestures, 1):
                st.info(f"Hand {i}: **{gesture}**")
        elif "Hand detection unavailable" in gestures:
            st.info("📝 Hand detection service is temporarily unavailable. Please try again in a few moments.")
        else:
            st.warning("⚠️ No hands detected in the image")
    except Exception as e:
        st.error(f"Error processing image: {e}")

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
