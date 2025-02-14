import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil

# Set paths
RAW_DATA_PATH = os.path.join('data', 'raw_images')
PROCESSED_DATA_PATH = os.path.join('data', 'processed_images')

# Ensure the processed data folder exists
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# Preprocessing and augmentation settings
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Preprocess and save data
def preprocess_and_save_data():
    print("Starting data preprocessing and augmentation...")
    
    for label in os.listdir(RAW_DATA_PATH):
        label_path = os.path.join(RAW_DATA_PATH, label)
        processed_label_path = os.path.join(PROCESSED_DATA_PATH, label)
        os.makedirs(processed_label_path, exist_ok=True)

        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Skipping invalid image: {img_path}")
                continue

            # Resize image to a fixed size (e.g., 120x120)
            img_resized = cv2.resize(img, (120, 120))

            # Convert to numpy array and expand dimensions for augmentation
            img_array = np.expand_dims(img_resized, axis=0)

            # Generate augmented images
            i = 0
            for batch in datagen.flow(img_array, batch_size=1, save_to_dir=processed_label_path, 
                                      save_prefix=label, save_format='jpg'):
                i += 1
                if i >= 10:  # Save 10 augmented images per original image
                    break

    print("Data preprocessing and augmentation completed.")

# Clean processed data folder if needed
def clean_processed_data():
    if os.path.exists(PROCESSED_DATA_PATH):
        shutil.rmtree(PROCESSED_DATA_PATH)
        os.makedirs(PROCESSED_DATA_PATH)
        print("Processed data folder cleaned.")

# Main function
if __name__ == "__main__":
    clean_choice = input("Do you want to clean the processed data folder before starting? (y/n): ").strip().lower()
    if clean_choice == 'y':
        clean_processed_data()
    
    preprocess_and_save_data()
