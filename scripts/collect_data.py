import os
import time
import uuid
import cv2

# Define the path where images will be saved
IMAGES_PATH = os.path.join('data', 'raw_images')  # Adjust folder name as needed
os.makedirs(IMAGES_PATH, exist_ok=True)

# Number of images to capture per gesture
number_images = 30

# Label for the gesture (e.g., 'A', 'B', etc.)
gesture_label = input("Enter the label for the gesture (e.g., 'A', 'B', etc.): ").strip()

# Create a subfolder for the specific gesture
gesture_path = os.path.join(IMAGES_PATH, gesture_label)
os.makedirs(gesture_path, exist_ok=True)

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print(f"Starting to collect images for gesture '{gesture_label}'...")

try:
    for imgnum in range(number_images):
        print(f'Collecting image {imgnum + 1}/{number_images}')

        # Capture the frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture an image.")
            continue

        # Generate a unique filename
        imgname = os.path.join(gesture_path, f'{gesture_label}_{uuid.uuid4()}.jpg')

        # Save the image
        cv2.imwrite(imgname, frame)

        # Display the frame to the user
        cv2.imshow('Frame', frame)

        # Pause for a moment before capturing the next image
        time.sleep(0.5)

        # Exit early if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting image collection early.")
            break

    print(f"Image collection for gesture '{gesture_label}' completed.")

finally:
    # Release the webcam and close any open windows
    cap.release()
    cv2.destroyAllWindows()
