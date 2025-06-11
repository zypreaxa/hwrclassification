import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import numpy as np

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_CLASSES = 10

# Load CIFAR-10 dataset
(train_images_raw, train_labels_raw), (test_images_raw, test_labels_raw) = keras.datasets.cifar10.load_data()

# Convert labels to one-hot encoding (do this once for all labels)
train_labels_one_hot = keras.utils.to_categorical(train_labels_raw, NUM_CLASSES)
test_labels_one_hot = keras.utils.to_categorical(test_labels_raw, NUM_CLASSES)

AUTOTUNE = tf.data.AUTOTUNE

def preprocess_image_and_label(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    return image, label

def prepare_dataset(images, labels):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(preprocess_image_and_label, num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    return dataset

train_ds = prepare_dataset(train_images_raw, train_labels_one_hot)
test_ds = prepare_dataset(test_images_raw, test_labels_one_hot)

print(f"Train dataset size: {len(train_images_raw)} images")
print(f"Test dataset size: {len(test_images_raw)} images")
print(f"Image shape after resizing: {IMG_HEIGHT}x{IMG_WIDTH}")

base_model = MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = keras.Model(inputs, outputs)

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- Training the top layers (feature extraction phase) ---")
initial_epochs = 10

history = model.fit(
    train_ds,
    epochs=initial_epochs,
    validation_data=test_ds
)

def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

plot_history(history)

print("\n--- Starting Fine-tuning (unfreezing some base layers) ---")

base_model.trainable = True

fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine_tune = model.fit(
    train_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=test_ds
)

def plot_combined_history(history_initial, history_fine_tune):
    acc = history_initial.history['accuracy'] + history_fine_tune.history['accuracy']
    val_acc = history_initial.history['val_accuracy'] + history_fine_tune.history['val_accuracy']

    loss = history_initial.history['loss'] + history_fine_tune.history['loss']
    val_loss = history_initial.history['val_loss'] + history_fine_tune.history['val_loss']

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Combined Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, max(max(loss), max(val_loss)) * 1.1])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Combined Training and Validation Loss')
    plt.show()

plot_combined_history(history, history_fine_tune)

print("\n--- Evaluating the final model ---")
loss, accuracy = model.evaluate(test_ds)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

model.save("cifar10_mobilenetv2_transfer_learning.keras")
print("Model saved to cifar10_mobilenetv2_transfer_learning.keras")