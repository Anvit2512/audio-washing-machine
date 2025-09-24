import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import models

# ========== Utility: Save mel spectrogram ==========
def save_mel_spectrogram(file_path, save_dir="temp_specs", sr=22050, n_mels=128, hop_length=512, n_fft=2048):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(file_path).replace(".wav", ".png"))

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='magma')
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return save_path

# ========== Hierarchical Classifier ==========
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
        stage1_pred = self.stage1_model.predict(img_array, verbose=0)
        stage1_idx = np.argmax(stage1_pred)
        main_class = self.stage1_classes[stage1_idx]

        if main_class == "00 - Abnormal":
            sub_pred = self.abnormal_model.predict(img_array, verbose=0)
            sub_idx = np.argmax(sub_pred)
            sub_class = self.abnormal_classes[sub_idx]
        else:
            sub_pred = self.normal_model.predict(img_array, verbose=0)
            sub_idx = np.argmax(sub_pred)
            sub_class = self.normal_classes[sub_idx]

        return {
            "stage1_class": main_class,
            "stage1_confidence": float(np.max(stage1_pred)),
            "stage2_class": sub_class,
            "stage2_confidence": float(np.max(sub_pred)),
            "final_prediction": f"{main_class} → {sub_class}"
        }

# ========== Load Models ==========
stage1_model = models.load_model("saved_models/stage1_model.h5")
abnormal_model = models.load_model("saved_models/abnormal_model.h5")
normal_model = models.load_model("saved_models/normal_model.h5")

# Define class lists (same order as training!)
stage1_classes = ["00 - Abnormal", "01 - Normal"]
abnormal_classes = os.listdir("MelSpectrograms/00 - Abnormal")
normal_classes = os.listdir("MelSpectrograms/01 - Normal")

classifier = HierarchicalClassifier(stage1_model, abnormal_model, normal_model,
                                    stage1_classes, abnormal_classes, normal_classes)

# ========== Example Inference ==========
audio_file = "C:/Users/dell/3D Objects/Samsung Prism/Brain\Audio/audio-washing-machine/Washing machine/00 - Abnormal/00-2 - Dehydration mode noise/04.wav"   # 🔹 Replace with your audio file
spec_path = save_mel_spectrogram(audio_file)

result = classifier.predict(spec_path)
print("🎯 Final Prediction:", result["final_prediction"])
print("Stage 1:", result["stage1_class"], "| Confidence:", result["stage1_confidence"])
print("Stage 2:", result["stage2_class"], "| Confidence:", result["stage2_confidence"])
