import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image # Used for opening image files
import os 
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 10

cifar10_class_names = [
    'airplane',  'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck']

# Loading the trained model
try:
    loaded_model = keras.models.load_model("cifar10_mobilenetv2_transfer_learning.keras")
    print("Model 'cifar10_mobilenetv2_transfer_learning.keras' loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'cifar10_mobilenetv2_transfer_learning.keras' exists in the current directory.")
    exit()

def preprocess_for_model(img_tensor):
    img_tensor = tf.cast(img_tensor, tf.float32) / 255.0
    img_tensor = tf.image.resize(img_tensor, (IMG_HEIGHT, IMG_WIDTH))
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    return img_tensor

# Function for getting images from a directory
def predict_from_directory(directory_path, num_images_to_predict=None):
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return

    image_paths = []
    # Cycling through directory for all images
    for root, _, files in os.walk(directory_path):
        for file in files:
            # Checks the extensions
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print(f"No image files found in '{directory_path}'.")
        return

    if num_images_to_predict and num_images_to_predict < len(image_paths):
        np.random.shuffle(image_paths)
        image_paths = image_paths[:num_images_to_predict]

    plt.figure(figsize=(15, 6 * ((len(image_paths) + 4) // 5))) # Adjusting the image size dynamically
    
    for i, img_path in enumerate(image_paths):
        try:
            original_img_pil = Image.open(img_path).convert('RGB')
            
            # Processing image for the model
            img_tensor_for_processing = tf.constant(np.array(original_img_pil))
            processed_img_for_model = preprocess_for_model(img_tensor_for_processing)

            # Getting the actual prediction
            predictions = loaded_model.predict(processed_img_for_model, verbose=0) # verbose=0 to suppress predict output

            # Getting the predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class_name = cifar10_class_names[predicted_class_idx]
            confidence = predictions[0][predicted_class_idx] * 100

            # Displaying the image with prediction
            ax = plt.subplot((len(image_paths) + 4) // 5, 5, i + 1) 
            plt.imshow(original_img_pil) 
            
            true_label_folder = os.path.basename(os.path.dirname(img_path))
            true_label_display = ""
            if true_label_folder in cifar10_class_names: # Check if folder name is a known class
                true_label_display = f"True: {true_label_folder}\n"
                color = 'green' if true_label_folder == predicted_class_name else 'red'
            else:
                color = 'black' # Default color if true label is unknown

            plt.title(f"{true_label_display}Pred: {predicted_class_name}\n({confidence:.2f}%)", color=color)
            plt.axis('off')

        except Exception as e:
            print(f"Could not process image '{img_path}': {e}")

    plt.tight_layout()
    plt.show()

 # Main stuff
if __name__ == "__main__":
    test_images_directory = "planes" # Folder of input images


    print(f"\n--- Predicting for images in directory: {test_images_directory} ---")
    # Prediction func
    predict_from_directory(test_images_directory, num_images_to_predict=10) # int = set number of images, none = all images

