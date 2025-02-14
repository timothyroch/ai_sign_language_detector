import os
import numpy as np
import cv2
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Utility to load and preprocess an image
def load_and_preprocess_image(image_path, target_size=(120, 120)):
    """
    Load an image from a given path and preprocess it for the model.
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Target size to resize the image (default is (120, 120)).
    Returns:
        np.ndarray: Preprocessed image ready for inference.
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize to range [0, 1]
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Utility to display sample images
def display_sample_images(data_path, num_samples=5):
    """
    Display a few sample images from each class in the dataset.
    Args:
        data_path (str): Path to the dataset folder.
        num_samples (int): Number of samples to display per class.
    """
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if os.path.isdir(label_path):
            print(f"Class: {label}")
            sample_images = os.listdir(label_path)[:num_samples]
            fig, axes = plt.subplots(1, len(sample_images), figsize=(15, 5))
            for i, img_file in enumerate(sample_images):
                img_path = os.path.join(label_path, img_file)
                img = cv2.imread(img_path)[:, :, ::-1]  # Convert BGR to RGB
                axes[i].imshow(img)
                axes[i].axis('off')
            plt.show()

# Utility to plot training history
def plot_training_history(history):
    """
    Plot training and validation accuracy and loss.
    Args:
        history (tf.keras.callbacks.History): Training history object.
    """
    plt.figure(figsize=(12, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Utility to evaluate the model and generate a confusion matrix
def evaluate_model(model, test_generator, class_labels):
    """
    Evaluate the model on the test set and display a confusion matrix.
    Args:
        model (tf.keras.Model): Trained model.
        test_generator (tf.keras.preprocessing.image.DirectoryIterator): Test data generator.
        class_labels (list): List of class labels.
    """
    print("Evaluating the model...")
    y_pred = np.argmax(model.predict(test_generator), axis=1)
    y_true = test_generator.classes

    # Classification report
    print("\nClassification Report:")
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

# Utility to save a model
def save_model(model, path):
    """
    Save the model to the specified path.
    Args:
        model (tf.keras.Model): Trained model.
        path (str): Path to save the model.
    """
    model.save(path)
    print(f"Model saved to: {path}")

# Utility to load a model
def load_model(path):
    """
    Load a saved model from the specified path.
    Args:
        path (str): Path to the saved model file.
    Returns:
        tf.keras.Model: Loaded model.
    """
    model = tf.keras.models.load_model(path)
    print(f"Model loaded from: {path}")
    return model
