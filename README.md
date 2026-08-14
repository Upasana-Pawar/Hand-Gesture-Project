# 🧼 Computer Vision Hand Hygiene Scanner

An AI-powered Computer Vision system designed to monitor handwashing gestures in real time and provide visual feedback to improve hand hygiene compliance.

Developed as part of an MSc Dissertation in Applied Computer Science and Artificial Intelligence, this project combines MediaPipe hand tracking, Machine Learning, and real-time gesture recognition to evaluate handwashing techniques based on key hand hygiene actions.

---

# 📌 Overview

Hand hygiene is one of the most effective methods for preventing the spread of infectious diseases. Despite widespread awareness, many individuals fail to perform handwashing correctly, often missing critical steps required for effective cleaning.

This project explores how Artificial Intelligence and Computer Vision can be used to recognize handwashing gestures and provide real-time feedback to users. By analyzing hand landmarks captured through a webcam, the system identifies specific handwashing actions and guides users through a structured hand hygiene workflow.

The project demonstrates the feasibility of using AI-powered gesture recognition for healthcare, public hygiene education, and future compliance monitoring systems.

---

# 🎯 Project Objectives

The primary objectives of this project are:

* Develop a Computer Vision-based hand gesture recognition system.
* Detect and classify handwashing gestures in real time.
* Provide visual feedback to improve handwashing technique.
* Explore the application of AI in healthcare and hygiene monitoring.
* Create a foundation for future hand hygiene compliance systems.

---

# ✨ Features

✅ Real-time hand tracking using MediaPipe

✅ Extraction of 21 hand landmarks

✅ Machine Learning-based gesture classification

✅ WHO-inspired handwashing workflow

✅ Webcam-based live recognition

✅ Visual feedback system

✅ Progress tracking through handwashing stages

✅ Gesture sequence validation

---

# 🏥 Problem Statement

Poor hand hygiene remains one of the leading causes of infection transmission in healthcare and public environments.

Many existing solutions focus only on handwashing duration and fail to assess whether the correct handwashing techniques are performed.

This project addresses that gap by using Computer Vision and Machine Learning to recognize handwashing gestures and provide immediate visual feedback, encouraging proper hand hygiene practices.

---

# 🧠 How It Works

The system follows the workflow below:

```text
Webcam
   ↓
MediaPipe Hand Detection
   ↓
21 Hand Landmarks Extraction
   ↓
Feature Normalization
   ↓
Random Forest Classifier
   ↓
Gesture Prediction
   ↓
Visual Feedback
```

---

# 🖥️ System Architecture

The application consists of three primary components:

### 1. Hand Detection

MediaPipe Hands detects and tracks hand landmarks in real time.

### 2. Gesture Classification

A Random Forest classifier processes normalized landmark coordinates and predicts the current handwashing gesture.

### 3. Feedback System

The recognized gesture is displayed on-screen, allowing users to monitor their handwashing progress.

---

# 🧰 Technology Stack

## Programming Language

* Python

## Computer Vision

* MediaPipe
* OpenCV

## Machine Learning

* Scikit-Learn
* Random Forest Classifier

## Data Processing

* NumPy
* Pandas

## Model Persistence

* Joblib

## Development Tools

* Visual Studio Code
* Git
* GitHub

---

# 📂 Project Structure

```text
Computer-Vision-Hand-Hygiene-Scanner
│
├── data
│   ├── Images
│   └── hand_washing_data.csv
│
├── model
│   ├── gesture_classifier.pkl
│   └── X_max.npy
│
├── scripts
│   ├── train_model.py
│   ├── test_on_image.py
│   └── real_time_recognition.py
│
├── docs
│   ├── architecture.png
│   ├── screenshots
│   └── demo.gif
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

# 🤖 Machine Learning Pipeline

### Data Collection

The dataset contains hand landmark coordinates generated from MediaPipe's 21-point hand tracking model.

### Feature Extraction

For every frame:

* 21 landmarks are detected.
* Each landmark contains:

  * X coordinate
  * Y coordinate
  * Z coordinate

Total Features:

```text
21 landmarks × 3 coordinates = 63 features
```

### Data Normalization

Feature values are normalized before model training.

Saved normalization file:

```text
model/X_max.npy
```

### Model Training

The system trains a Random Forest Classifier using labeled hand gesture data.

Training script:

```bash
python scripts/train_model.py
```

### Model Storage

Trained model:

```text
model/gesture_classifier.pkl
```

---

# 🖐️ Supported Gestures

The current prototype supports the following gestures:

| Gesture           | Description                      |
| ----------------- | -------------------------------- |
| Start             | Beginning of handwashing process |
| Rubbing Palm      | Palm-to-palm rubbing             |
| Scrubbing Fingers | Finger cleaning motion           |
| Rinsing Hands     | Hand rinsing gesture             |
| End               | Completion of handwashing        |

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/Computer-Vision-Hand-Hygiene-Scanner.git

cd Computer-Vision-Hand-Hygiene-Scanner
```

## Create Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv

source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

## Train the Model

```bash
python scripts/train_model.py
```

## Test on Images

```bash
python scripts/test_on_image.py
```

## Run Real-Time Recognition

```bash
python scripts/real_time_recognition.py
```

The webcam will open automatically and begin tracking hand gestures.

---

# 📊 Results

### Prototype Performance

| Metric            | Value |
| ----------------- | ----- |
| Accuracy          | 24%   |
| Weighted F1 Score | 0.25  |

The project successfully demonstrated the feasibility of using MediaPipe landmarks and Machine Learning for hand hygiene monitoring.

The results highlight the importance of larger datasets and more advanced temporal models for achieving production-level performance.

---

# ⚠️ Current Limitations

### Dataset Size

The dataset is relatively small and contains limited gesture diversity.

### Synthetic Data

The project primarily uses AI-generated landmark data rather than large-scale real-world recordings.

### Environmental Sensitivity

Performance can be affected by:

* Lighting conditions
* Camera quality
* Background complexity

### Gesture Similarity

Certain gestures share similar landmark patterns, leading to occasional misclassification.

---

# 🔮 Future Improvements

### Data Collection

* Build a real-world gesture dataset
* Increase participant diversity
* Capture gestures under different lighting conditions

### Deep Learning

* Replace Random Forest with:

  * LSTM Networks
  * Temporal CNNs
  * Transformer-based architectures

### User Experience

* Audio feedback
* Mobile application
* Improved visual guidance

### Deployment

* Streamlit Web Application
* FastAPI Backend
* Docker Containerization
* Cloud Deployment

### Healthcare Applications

* Compliance monitoring
* Healthcare staff training
* Public health education
* Smart hygiene stations

---

# 📸 Demo

1. Output of the project when it is active

 (1)\data\Images\ss\Hand Gesture Recognition.png




---

# 📚 Research Context

This project was developed as part of an MSc Dissertation in Applied Computer Science and Artificial Intelligence.

The research investigates the use of Computer Vision and Machine Learning techniques to improve hand hygiene practices through gesture recognition and real-time feedback systems.

---

# 👩‍💻 Author

**Upasana Pawar**

MSc Applied Computer Science and Artificial Intelligence

University of Bradford


---

# 📄 License

This project is licensed under the MIT License.

See the LICENSE file for details.

---

# ⭐ Acknowledgements

* Google MediaPipe Team
* OpenCV Community
* Scikit-Learn Contributors
* University of Bradford
* Dissertation Supervisor Viktor Doychinov

