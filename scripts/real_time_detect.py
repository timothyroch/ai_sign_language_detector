import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the trained model
MODEL_PATH = 'models/sign_model.keras'  # Update path if necessary
print("Loading the model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# Initialize MediaPipe Hands for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Define class labels
CLASS_LABELS = ['L', 'A', 'C']  # Update with your actual labels

# Start webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

print("Starting real-time sign language detection... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from the webcam.")
        break

    # Flip the frame horizontally for a mirror-like effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract bounding box around the hand
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Add padding to the bounding box
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # Extract the hand region from the frame
            hand_region = frame[y_min:y_max, x_min:x_max]

            # Preprocess the hand region
            hand_region_resized = cv2.resize(hand_region, (120, 120))
            hand_region_normalized = hand_region_resized / 255.0
            hand_region_reshaped = np.expand_dims(hand_region_normalized, axis=0)

            # Make predictions
            # Make predictions
            predictions = model.predict(hand_region_reshaped)

# Debug print for predictions
            print(f"Raw predictions: {predictions}")

            predicted_index = np.argmax(predictions)
            print(f"Predicted index: {predicted_index}")
            print(f"Class labels length: {len(CLASS_LABELS)}")

            predictions = model.predict(hand_region_reshaped)
            predicted_label = CLASS_LABELS[np.argmax(predictions)]

            # Display the prediction on the frame
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, predicted_label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Real-Time Sign Language Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
