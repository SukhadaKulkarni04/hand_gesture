import cv2
import numpy as np
from squeezenet import SqueezeNet
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential
import tensorflow as tf
import os

# Define the path to save images
IMG_SAVE_PATH = 'train_images'

# Define the class mappings for the gestures
CLASS_MAP = {
    "down": 0,
    "left": 1,
    "right": 2,
    "up": 3,
    "none": 4,
}

# Number of classes
NUM_CLASSES = len(CLASS_MAP)

# Mapper function to convert class names to their corresponding integer values
def mapper(val):
    return CLASS_MAP[val]

# Define the model architecture
def get_model():
    model = Sequential([
        SqueezeNet(input_shape=(227, 227, 3), include_top=False),
        Dropout(0.5),
        Convolution2D(NUM_CLASSES, (1, 1), padding='valid'),
        Activation('relu'),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ])
    return model

# Load images from the directory
dataset = []
for directory in os.listdir(IMG_SAVE_PATH):
    path = os.path.join(IMG_SAVE_PATH, directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        # Skip hidden files
        if item.startswith("."):
            continue
        img = cv2.imread(os.path.join(path, item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (227, 227))
        dataset.append([img, directory])

# Separate data and labels
data, labels = zip(*dataset)
labels = list(map(mapper, labels))

# One-hot encode the labels
labels = np_utils.to_categorical(labels, NUM_CLASSES)

# Define the model
model = get_model()

# Compile the model with updated 'learning_rate' argument
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Check the shapes of data and labels
print("Data shape:", np.array(data).shape)  # Should be (num_samples, 227, 227, 3)
print("Labels shape:", np.array(labels).shape)  # Should be (num_samples, 5)

# Start training
model.fit(np.array(data), np.array(labels), epochs=3)

# Save the model for later use
model.save("hand_gesture.h5")
