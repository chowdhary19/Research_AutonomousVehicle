

# Autonomous Vehicle Steering Model

## Collaborators
- **Prateek Rathore**
- **Utkarsh Varshney**

## GitHub Author
- **Yuvraj Singh Chowdhary**

---

### Introduction

This project involves developing a model for autonomous vehicle steering using a convolutional neural network (CNN). The model is based on the NVIDIA architecture, which is well-suited for this type of task. Below is the detailed implementation of the project, including data loading, preprocessing, augmentation, model training, and evaluation.

### Libraries and Dependencies

First, we import the necessary libraries and dependencies:

```python
import os
import ntpath
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from imgaug import augmenters as iaa
import random
import matplotlib.image as mpimg
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.layers import Activation
import tensorflow as tf
```

### Load Data

Load the data from the `driving_log.csv` file and update file paths:

```python
# Define the data directory
datadir = 'track'

# Define the column names
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']

# Load the data
data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names=columns)

# Update file paths
data['center'] = data['center'].apply(lambda x: os.path.join(datadir, 'IMG', ntpath.basename(x)))
data['left'] = data['left'].apply(lambda x: os.path.join(datadir, 'IMG', ntpath.basename(x)))
data['right'] = data['right'].apply(lambda x: os.path.join(datadir, 'IMG', ntpath.basename(x)))
```

### Balance Data

Balance the data to avoid overfitting to a particular steering angle:

```python
# Define number of bins and samples per bin
num_bins = 25
samples_per_bin = 2000

# Create a histogram of the steering angles
hist, bins = np.histogram(data['steering'], num_bins)
center = (bins[:-1] + bins[1:]) * 0.5

# List to store the indices to be removed
remove_list = []

# Balance the data
for j in range(num_bins):
    list_ = []
    for i in range(len(data['steering'])):
        if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
            list_.append(i)
    list_ = shuffle(list_)
    list_ = list_[samples_per_bin:]
    remove_list.extend(list_)

# Drop the unbalanced data
data.drop(data.index[remove_list], inplace=True)
```

### Load Images and Steering Angles

Define a function to load images and steering angles:

```python
def load_img_steering(df):
    image_paths = []
    steerings = []
    correction = 0.2

    for _, row in df.iterrows():
        image_paths.append(row['center'])
        steerings.append(float(row['steering']))

        image_paths.append(row['left'])
        steerings.append(float(row['steering']) + correction)

        image_paths.append(row['right'])
        steerings.append(float(row['steering']) - correction)

    return np.asarray(image_paths), np.asarray(steerings)

# Load image paths and steering angles
image_paths, steerings = load_img_steering(data)
```

### Split Data

Split the data into training and validation sets:

```python
# Split the data
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)
```

### Data Augmentation

Define functions for data augmentation:

```python
# Function for zoom augmentation
def zoom(image):
    zoom_factor = np.random.uniform(1, 1.3)
    zoom = iaa.Affine(scale=(1, zoom_factor))
    return zoom.augment_image(image)

# Function for pan augmentation
def pan(image):
    translate_range = {"x": (-10, 10), "y": (-10, 10)}
    pan = iaa.Affine(translate_px=translate_range)
    return pan.augment_image(image)

# Function for random brightness augmentation
def img_random_brightness(image):
    brightness = iaa.Multiply((0.2, 1.2))
    return brightness.augment_image(image)

# Function for random flip augmentation
def img_random_flip(image, steering_angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

# Function for adding random shadow
def add_random_shadow(image):
    top_y = 320 * np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = 0 * image_hls[:, :, 1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1
    random_bright = .5
    cond1 = shadow_mask == 1
    cond0 = shadow_mask == 0
    if np.random.randint(2) == 1:
        image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
    else:
        image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright
    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    return image

# Function for random augmentation
def random_augment(image, steering_angle):
    image = mpimg.imread(image)
    image = zoom(image)
    image = pan(image)
    image = img_random_brightness(image)
    image, steering_angle = img_random_flip(image, steering_angle)
    return image, steering_angle
```

### Model Architecture

Define the NVIDIA model architecture:

```python
# Define the NVIDIA model architecture
def nvidia_model():
    model = Sequential()
    model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(Flatten())

    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss='mse', optimizer=optimizer)
    return model

# Initialize the model
model = nvidia_model()
print(model.summary())
```

### GPU Monitoring

Monitor GPU performance:

```python
!nvidia-smi
```

### Batch Generator

Define a batch generator for training and validation:

```python
# Define the batch generator
def batch_generator(image_paths, steering_ang, batch_size, istraining):
    while True:
        batch_img = []
        batch_steering = []

        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)

            if istraining:
                im, steering = random_augment(image_paths[random_index], steering_ang[random_index])
            else:
                im = mpimg.imread(image_paths[random_index])
                steering = steering_ang[random_index]

            im = img_preprocess(im)
            batch_img.append(im)
            batch_steering.append(steering)

        yield (np.asarray(batch_img), np.asarray(batch_steering))

# Define preprocessing function
def img_preprocess(img):
    img = img[50:140,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.resize(img, (200, 66))
    img = img.astype(np.float32) / 255.0
    return img
```

### Training the Model

Train the model using the defined batch generator:

```python
# Define batch size and steps per epoch
batch_size_cpu = 32
steps_per_epoch_cpu = len(X_train) // batch_size_cpu
validation_steps_cpu = len(X_valid) // batch_size_cpu

# Define

 early stopping and learning rate reduction callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Train the model
history = model.fit(batch_generator(X_train, y_train, batch_size_cpu, True),
                    steps_per_epoch=steps_per_epoch_cpu,
                    epochs=30,
                    validation_data=batch_generator(X_valid, y_valid, batch_size_cpu, False),
                    validation_steps=validation_steps_cpu,
                    callbacks=[early_stopping, reduce_lr],
                    verbose=1)
```

### Evaluate Model

Evaluate the model and calculate performance metrics:

```python
# Make predictions on the validation set
predictions = model.predict(batch_generator(X_valid, y_valid, batch_size_cpu, False), steps=validation_steps_cpu)

# Calculate performance metrics
mse = mean_squared_error(y_valid, predictions)
mae = mean_absolute_error(y_valid, predictions)
r2 = r2_score(y_valid, predictions)

print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)
print('R2 Score:', r2)
```

---

This document provides a detailed walkthrough of the code implementation for the autonomous vehicle steering model.
