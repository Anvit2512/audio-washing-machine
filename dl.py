import tensorflow as tf
from keras._tf_keras.keras.utils import image_dataset_from_directory
from keras._tf_keras.keras import Sequential, layers, models
import numpy as np
import os
img_size = (224, 224)
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

normalization_layer = layers.Rescaling(1./255)

def preprocess_ds(ds, training=True):
    ds = ds.map(lambda x, y: (normalization_layer(x), y))
    if training:
        ds = ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    else:
        ds = ds.cache().prefetch(buffer_size=AUTOTUNE)
    return ds

def build_model(num_classes):
    model = models.Sequential([
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
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

stage1_train = image_dataset_from_directory(
    "MelSpectrograms",
    labels="inferred",
    label_mode="int",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)
stage1_val = image_dataset_from_directory(
    "MelSpectrograms",
    labels="inferred",
    label_mode="int",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

stage1_classes = stage1_train.class_names
print("Stage 1 Classes:", stage1_classes)

stage1_train = preprocess_ds(stage1_train, training=True)
stage1_val = preprocess_ds(stage1_val, training=False)

print("Training Stage 1 (Normal vs Abnormal)...")
stage1_model = build_model(len(stage1_classes))
stage1_model.fit(stage1_train, validation_data=stage1_val, epochs=10)

abnormal_train = image_dataset_from_directory(
    "MelSpectrograms/00 - Abnormal",
    labels="inferred",
    label_mode="int",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)
abnormal_val = image_dataset_from_directory(
    "MelSpectrograms/00 - Abnormal",
    labels="inferred",
    label_mode="int",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

abnormal_classes = abnormal_train.class_names
print("Stage 2 Abnormal Classes:", abnormal_classes)

abnormal_train = preprocess_ds(abnormal_train, training=True)
abnormal_val = preprocess_ds(abnormal_val, training=False)

print("Training Stage 2 (Abnormal)...")
abnormal_model = build_model(len(abnormal_classes))
abnormal_model.fit(abnormal_train, validation_data=abnormal_val, epochs=10)

normal_train = image_dataset_from_directory(
    "MelSpectrograms/01 - Normal",
    labels="inferred",
    label_mode="int",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)
normal_val = image_dataset_from_directory(
    "MelSpectrograms/01 - Normal",
    labels="inferred",
    label_mode="int",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

normal_classes = normal_train.class_names
print("Stage 2 Normal Classes:", normal_classes)

normal_train = preprocess_ds(normal_train, training=True)
normal_val = preprocess_ds(normal_val, training=False)

print("Training Stage 2 (Normal)...")
normal_model = build_model(len(normal_classes))
normal_model.fit(normal_train, validation_data=normal_val, epochs=10)

class HierarchicalClassifier:
    def __init__(self, stage1_model, abnormal_model, normal_model,
                 stage1_classes, abnormal_classes, normal_classes, img_size=(224, 224)):
        self.img_size = img_size
        self.stage1_model = stage1_model
        self.abnormal_model = abnormal_model
        self.normal_model = normal_model
        self.stage1_classes = stage1_classes
        self.abnormal_classes = abnormal_classes
        self.normal_classes = normal_classes

    def preprocess(self, image_path):
        img = tf.keras.utils.load_img(image_path, target_size=self.img_size)
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, 0)
        return img_array

    def predict(self, image_path):
        img_array = self.preprocess(image_path)
        stage1_pred = self.stage1_model.predict(img_array)
        stage1_idx = np.argmax(stage1_pred)
        main_class = self.stage1_classes[stage1_idx]
        if main_class == "00 - Abnormal":
            sub_pred = self.abnormal_model.predict(img_array)
            sub_idx = np.argmax(sub_pred)
            sub_class = self.abnormal_classes[sub_idx]
        else:
            sub_pred = self.normal_model.predict(img_array)
            sub_idx = np.argmax(sub_pred)
            sub_class = self.normal_classes[sub_idx]
        return {
            "stage1_class": main_class,
            "stage1_confidence": float(np.max(stage1_pred)),
            "stage2_class": sub_class,
            "stage2_confidence": float(np.max(sub_pred)),
            "final_prediction": f"{main_class} → {sub_class}"
        }

# classifier = HierarchicalClassifier(stage1_model, abnormal_model, normal_model,
#                                     stage1_classes, abnormal_classes, normal_classes)

# result = classifier.predict("MelSpectrograms/00 - Abnormal/00-2 - Dehydration mode noise/01.png")
# print(result["final_prediction"])


# Save models after training
os.makedirs("saved_models", exist_ok=True)
stage1_model.save("saved_models/stage1_model.h5")
abnormal_model.save("saved_models/abnormal_model.h5")
normal_model.save("saved_models/normal_model.h5")

print("✅ Models saved in 'saved_models/' folder")
