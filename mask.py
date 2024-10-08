# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import cv2

# Extracting dataset
from zipfile import ZipFile
dataset="D:/faisal-VS/faisal project/self_driving_car_project/face mask/face-mask-dataset.zip"
with ZipFile(dataset,'r') as zip:
    zip.extractall()
    print("The dataset is extracted")

# Listing the dataset directory
with_mask_file = os.listdir('D:/faisal-VS/faisal project/data/with_mask')
without_mask_file = os.listdir('D:/faisal-VS/faisal project/data/without_mask')

# Labels for images
with_mask_label = [1] * len(with_mask_file)
without_mask_label = [0] * len(without_mask_file)
labels = with_mask_label + without_mask_label

# Processing and resizing images
data = []
with_mask_path = "D:/faisal-VS/faisal project/data/with_mask/"
without_mask_path = "D:/faisal-VS/faisal project/data/without_mask/"

for img_file in with_mask_file:
    image = Image.open(with_mask_path + img_file)
    image = image.resize((128, 128))
    image = image.convert('RGB')
    image = np.array(image)
    data.append(image)

for img_file in without_mask_file:
    image = Image.open(without_mask_path + img_file)
    image = image.resize((128, 128))
    image = image.convert('RGB')
    image = np.array(image)
    data.append(image)

# Convert data and labels to numpy arrays
x = np.array(data)
y = np.array(labels)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Scaling the data
x_train_scaled = x_train / 255.0
x_test_scaled = x_test / 255.0

# Building CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation='sigmoid')
])

# Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

# Training the model
history = model.fit(x_train_scaled, y_train, validation_split=0.1, epochs=5)
model.save('D:/faisal-VS/faisal project/self_driving_car_project/face mask/face_mask_model.h5')
# Evaluating the model
loss, accuracy = model.evaluate(x_test_scaled, y_test)
print("Test Accuracy =", accuracy)

# Plotting training history
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='validation_loss')
plt.legend()
plt.show()

plt.plot(history.history['acc'], label='train_accuracy')
plt.plot(history.history['val_acc'], label='validation_accuracy')
plt.legend()
plt.show()

# Predictive system
input_image_path = input("Path of image to predict: ")
input_image = cv2.imread(input_image_path)
cv2.imshow("Input Image", input_image)  # Use cv2.imshow instead of cv2_imshow
cv2.waitKey(0)
cv2.destroyAllWindows()

input_image_resize = cv2.resize(input_image, (128, 128))
input_image_scaled = input_image_resize / 255.0
input_image_scaled = np.reshape(input_image_scaled, [1, 128, 128, 3])
input_predict = model.predict(input_image_scaled)
input_pred_label = np.argmax(input_predict)

if input_pred_label == 1:
    print("The person in the image is wearing a mask.")
else:
    print("The person in the image is not wearing a mask.")