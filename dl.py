import tensorflow as tf
from keras._tf_keras.keras.utils import image_dataset_from_directory
from keras._tf_keras.keras import Sequential, layers, models 
import matplotlib.pyplot as plt 
import numpy as np
img_size = (224, 224) 
batch_size = 32

train_ds = image_dataset_from_directory(
    "MelSpectrograms",
    labels="inferred",
    label_mode="int", 
    validation_split=0.2,  
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)
val_ds = image_dataset_from_directory(
    "MelSpectrograms",
    labels="inferred",
    label_mode="int",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)
class_names = train_ds.class_names
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = Sequential([
    layers.RandomFlip("horizontal"), 
    layers.RandomZoom(0.1),       
    layers.RandomContrast(0.1),        
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

import matplotlib.pyplot as plt
for images, labels in train_ds.take(1):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[labels[i]])
        plt.axis("off")

num_classes = len(class_names)

cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=img_size + (3,)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])
cnn_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
history = cnn_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15
)
















