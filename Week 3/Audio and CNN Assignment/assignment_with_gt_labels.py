import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load the metadata CSV file
metadata_path = 'D:\\e-club python ml assignment\\Voice-Morph-Companion\\Voice-Morph-Companion1\\Week 3\\Audio and CNN Assignment\\TInySOL Dataset\\TinySOL\\TinySOL_metadata (1).csv'
audio_dir = 'D:\\e-club python ml assignment\\Voice-Morph-Companion\\Voice-Morph-Companion1\\Week 3\\Audio and CNN Assignment\\TInySOL Dataset\\TinySOL'  # Directory where audio files are stored
metadata = pd.read_csv(metadata_path)

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
labels = []

for index, row in metadata.iterrows():
    file_path = os.path.join(audio_dir, row['Path'])  # Adjust column name if necessary
    mel_spectrogram_db = load_audio_file(file_path, n_mels, max_length)
    audio_data.append(mel_spectrogram_db)
    labels.append(row['Instrument (in full)'])  # Adjust column name if necessary

audio_data = np.array(audio_data)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Reshape audio data to fit the CNN input
audio_data = audio_data[..., np.newaxis]

# Split the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(audio_data, labels_encoded, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(n_mels, max_length, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(np.unique(labels_encoded)), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history_gt = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# Save the model
model.save('instrument_classification_cnn.h5')

import matplotlib.pyplot as plt

# Assuming `history_gt` contains the training history for the model trained with ground truth labels

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history_gt.history['accuracy'], label='Train Accuracy')
plt.plot(history_gt.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy (Ground Truth Labels)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history_gt.history['loss'], label='Train Loss')
plt.plot(history_gt.history['val_loss'], label='Validation Loss')
plt.title('Model Loss (Ground Truth Labels)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
