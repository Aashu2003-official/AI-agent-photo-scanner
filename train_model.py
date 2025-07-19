import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
from tensorflow.keras.utils import img_to_array, to_categorical
from sklearn.model_selection import train_test_split

IMG_SIZE = 64
DATA_DIR = "train"
labels = {"me": 0, "mom": 1}

data = []
target = []

for label_name in labels:
    path = os.path.join(DATA_DIR, label_name)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img_to_array(img)
        data.append(img)
        target.append(labels[label_name])

data = np.array(data, dtype="float32") / 255.0
target = to_categorical(target)

x_train, x_val, y_train, y_val = train_test_split(data, target, test_size=0.2, random_state=42)

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(2, activation="softmax")  
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=8)



IMG_SIZE = 64
labels = ["me", "mom"]

def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image '{image_path}'")
        return
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0]
    label_index = np.argmax(prediction)
    print(f"Predicted: {labels[label_index]} ({prediction[label_index]*100:.2f}%)")


def predict_folder(folder_path):
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue
        print(f"Testing: {img_path}")
        predict_image(img_path)

predict_folder("test")