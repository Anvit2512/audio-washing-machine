import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def save_mel_spectrogram(file_path, save_dir, sr=22050, n_mels=128, hop_length=512, n_fft=2048):
    y, sr = librosa.load(file_path, sr=sr, mono=True)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(file_path).replace(".wav", ".png"))

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='magma')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return save_path

def process_dataset(wav_root, output_root, sr=22050, n_mels=128):
    for root, dirs, files in os.walk(wav_root):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, wav_root)
                save_dir = os.path.join(output_root, rel_path)
                save_mel_spectrogram(file_path, save_dir, sr=sr, n_mels=n_mels)

wav_root = "Washing machine" 
output_root = "MelSpectrograms" 
process_dataset(wav_root, output_root)



