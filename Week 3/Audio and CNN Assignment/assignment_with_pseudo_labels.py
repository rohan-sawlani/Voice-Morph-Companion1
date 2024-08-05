import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from sklearn.preprocessing import LabelEncoder

class AudioDataset:
    def __init__(self, csv_path, audio_dir, n_mels=128, max_length=128):
        self.csv_path = csv_path
        self.audio_dir = audio_dir
        self.n_mels = n_mels
        self.max_length = max_length
        self.metadata = pd.read_csv(csv_path)
        self.label_encoder = LabelEncoder()
        self.labels_encoded = self.label_encoder.fit_transform(self.metadata['Instrument (in full)'])  # Adjust column name if necessary
        self.model = self.build_model()

    def build_model(self):
        # Define a simple CNN model (untrained)
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.n_mels, self.max_length, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(len(np.unique(self.labels_encoded)), activation='softmax'))
        return model

    def load_audio_file(self, file_path):
        audio, sr = librosa.load(file_path, sr=None)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=self.n_mels)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Pad or truncate the Mel spectrogram to the fixed length
        if mel_spectrogram_db.shape[1] > self.max_length:
            mel_spectrogram_db = mel_spectrogram_db[:, :self.max_length]
        else:
            padding = self.max_length - mel_spectrogram_db.shape[1]
            mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, padding)), mode='constant')

        return audio, mel_spectrogram_db

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = os.path.join(self.audio_dir, row['Path'])  # Adjust column name if necessary
        audio, mel_spectrogram = self.load_audio_file(file_path)
        ground_truth = self.label_encoder.transform([row['Instrument (in full)']])[0]  # Adjust column name if necessary

        # Reshape audio and mel_spectrogram
        audio = audio[np.newaxis, :]
        mel_spectrogram = mel_spectrogram[np.newaxis, :, :]

        # Predict pseudo label using the untrained model (randomly initialized weights)
        mel_spectrogram_input = mel_spectrogram[..., np.newaxis]  # Add channel dimension
        pseudo_label = self.model.predict(mel_spectrogram_input)
        pseudo_label = np.argmax(pseudo_label, axis=1)[0]

        return {
            'file': row['Path'],
            'audio': audio,
            'mel': mel_spectrogram,
            'gt': ground_truth,
            'pseudo': pseudo_label
        }

# Example usage:
csv_path = "D:\e-club python ml assignment\Voice-Morph-Companion\Voice-Morph-Companion1\Week 3\Audio and CNN Assignment\TInySOL Dataset\TinySOL\TinySOL_metadata (1).csv"
audio_dir = "D:\e-club python ml assignment\Voice-Morph-Companion\Voice-Morph-Companion1\Week 3\Audio and CNN Assignment\TInySOL Dataset\TinySOL"
metadata = pd.read_csv(csv_path)
dataset = AudioDataset(csv_path, audio_dir)

# Get item at index 0
item = dataset[0]
#print(item)

def build_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Parameters
input_shape = (128, 128, 1)  # Adjust based on the dataset
num_classes = len(np.unique(dataset.labels_encoded))

# Build the model
model = build_cnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
from sklearn.model_selection import train_test_split

# Parameters
n_mels = 128
max_length = 128  # Adjust this based on the average length of your Mel spectrograms

# Function to load and preprocess audio files
def load_audio_file(file_path, n_mels=128, max_length=128):
    audio, sr = librosa.load(file_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Pad or truncate the Mel spectrogram to the fixed length
    if mel_spectrogram_db.shape[1] > max_length:
        mel_spectrogram_db = mel_spectrogram_db[:, :max_length]
    else:
        padding = max_length - mel_spectrogram_db.shape[1]
        mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, padding)), mode='constant')

    return mel_spectrogram_db

# Load and preprocess the dataset
audio_data = []
pseudo_labels=[]



for index, row in metadata.iterrows():
    file_path = os.path.join(audio_dir, row['Path'])  # Adjust column name if necessary
    mel_spectrogram_db = load_audio_file(file_path, n_mels, max_length)
    audio_data.append(mel_spectrogram_db)
    #jlabels.append(row['Instrument (in full)'])  # Adjust column name if n


for i in range(len(dataset)):
    item = dataset[i]
    #audio_data.append(item['mel'])
    pseudo_labels.append(item['pseudo'])


audio_data = np.array(audio_data)

pseudo_labels = np.array(pseudo_labels)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(audio_data, pseudo_labels, test_size=0.2, random_state=42)

# Train the model
history_pseudo = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# Save the model
model.save('instrument_classification_cnn_from_scratch.h5')

import matplotlib.pyplot as plt

# Assuming `history_pseudo` contains the training history for the model trained with pseudo labels

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history_pseudo.history['accuracy'], label='Train Accuracy')
plt.plot(history_pseudo.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy (Pseudo Labels)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history_pseudo.history['loss'], label='Train Loss')
plt.plot(history_pseudo.history['val_loss'], label='Validation Loss')
plt.title('Model Loss (Pseudo Labels)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
