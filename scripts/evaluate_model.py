import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set the paths
MODEL_PATH = os.path.join('models', 'sign_model.keras')  # Adjust if your model has a different name
TEST_DATA_PATH = os.path.join('data', 'processed_images', 'test')         # Path to the test dataset

# Load the trained model
print("Loading the trained model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# Load test data
print("Loading test data...")
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DATA_PATH,
    target_size=(120, 120),  # Adjust to match your model's input size
    batch_size=32,
    class_mode='categorical',  # Assuming multi-class classification
    shuffle=False
)

# Evaluate the model on the test data
print("Evaluating the model...")
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Get predictions
print("Generating predictions...")
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Classification report
print("\nClassification Report:")
class_labels = list(test_generator.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=class_labels))

# Confusion matrix
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
