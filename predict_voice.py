import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import joblib

# ------------------ Parameters ------------------
DURATION = 5          # seconds
SAMPLE_RATE = 22050*2 # Hz

# ------------------ Load model and preprocessors ------------------
model = tf.keras.models.load_model("model_augmented.h5")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ------------------ Record audio ------------------
print("Recording started...")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
sd.wait()
print("Recording finished!")

# ------------------ Preprocess audio ------------------
audio = audio.flatten()  # convert to 1D

# Extract features: MFCC + Chroma + Spectral Contrast + Tonnetz
mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
stft = np.abs(librosa.stft(audio))
chroma = librosa.feature.chroma_stft(S=stft, sr=SAMPLE_RATE)
spec_contrast = librosa.feature.spectral_contrast(S=stft, sr=SAMPLE_RATE)
tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=SAMPLE_RATE)

# Combine features
features = np.vstack([mfccs, chroma, spec_contrast, tonnetz]).T

# Pad or truncate to match training time_steps
time_steps = model.input_shape[1]
if features.shape[0] < time_steps:
    features = np.pad(features, ((0, time_steps - features.shape[0]), (0,0)), mode='constant')
elif features.shape[0] > time_steps:
    features = features[:time_steps, :]

# Scale features
features_2d = features.reshape(1, -1)
features_scaled_2d = scaler.transform(features_2d)
features_scaled = features_scaled_2d.reshape(1, time_steps, features.shape[1])

# ------------------ Make prediction ------------------
pred_probs = model.predict(features_scaled)
pred_index = np.argmax(pred_probs)
pred_label = label_encoder.inverse_transform([pred_index])[0]

print(f"Predicted emotion: {pred_label}")
