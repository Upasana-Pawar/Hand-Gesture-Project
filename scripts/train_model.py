import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# File paths
DATA_PATH = "data/hand_washing_data.csv"  # Path to your CSV file
MODEL_DIR = "model/"
MODEL_FILE = os.path.join(MODEL_DIR, "gesture_classifier.pkl")
X_MAX_FILE = os.path.join(MODEL_DIR, "X_max.npy")

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load the dataset
print("Loading dataset...")
data = pd.read_csv(DATA_PATH)

# Separate features (X) and labels (y)
X = data.iloc[:, :-1].values  # All columns except the last (keypoints)
y = data['label'].values  # Last column (gesture labels)

# Normalize the features (scale each feature to range [0, 1])
X_max = X.max(axis=0)
X_max[X_max == 0] = 1  # Prevent division by zero
X_normalized = X / X_max

# Save the normalization factors
np.save(X_MAX_FILE, X_max)
print(f"Normalization factors saved to: {X_MAX_FILE}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Train the model
print("Training the Random Forest model...")
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Save the trained model
joblib.dump(classifier, MODEL_FILE)
print(f"Trained model saved to: {MODEL_FILE}")

# Evaluate the model
print("Evaluating the model on the test set...")
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Training completed successfully!")
