import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
processed_path = os.path.join('data', 'processed_images')
train_path = os.path.join(processed_path, 'train')
val_path = os.path.join(processed_path, 'val')
test_path = os.path.join(processed_path, 'test')

# Clear existing train/val/test directories if they exist
for folder in [train_path, val_path, test_path]:
    if os.path.exists(folder):
        shutil.rmtree(folder)  # Remove old splits to prevent data accumulation
    os.makedirs(folder)  # Recreate empty directories

# Loop through all classes (folders like 'A', 'L', 'C') in processed_images
for class_name in os.listdir(processed_path):
    class_path = os.path.join(processed_path, class_name)

    # Skip train/val/test folders
    if not os.path.isdir(class_path) or class_name in ['train', 'val', 'test']:
        continue

    # Get all images for the current class
    all_images = [img for img in os.listdir(class_path) if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(all_images)} images in {class_path}")

    if len(all_images) == 0:
        print(f"No images found for class '{class_name}'. Skipping...")
        continue

    # Create class-specific directories inside train/val/test
    os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_path, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_path, class_name), exist_ok=True)

    # Split into train (70%), val (15%), and test (15%)
    train_imgs, temp_imgs = train_test_split(all_images, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    # Move images to respective folders
    for img in train_imgs:
        shutil.move(os.path.join(class_path, img), os.path.join(train_path, class_name, img))

    for img in val_imgs:
        shutil.move(os.path.join(class_path, img), os.path.join(val_path, class_name, img))

    for img in test_imgs:
        shutil.move(os.path.join(class_path, img), os.path.join(test_path, class_name, img))

    print(f"Data split completed for class '{class_name}'.")

print("All data splitting completed!")
