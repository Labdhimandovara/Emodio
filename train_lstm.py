import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib

# ==============================
# CONFIG
# ==============================
RAVDESS_PATH = r"C:\dsp_project\act 1-24"
CREMAD_PATH = r"C:\dsp_project\AudioWAV"

DURATION = 5          # seconds
SAMPLE_RATE = 22050*2
N_MFCC = 40

# Only common emotions across both datasets
COMMON_EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]

# RAVDESS emotion mapping
RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# CREMA-D emotion mapping
CREMAD_EMOTIONS = {
    "NE": "neutral",
    "HA": "happy",
    "SA": "sad",
    "AN": "angry",
    "FE": "fearful",
    "DI": "disgust"
}

features = []
labels = []

print("Script started!")

# ==============================
# DATA AUGMENTATION FUNCTIONS
# ==============================
def add_noise(data):
    noise_amp = 0.005 * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape)

def pitch_shift(data, sr):
    n_steps = np.random.randint(-2, 3)  # random semitones shift
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)

def time_stretch(data):
    rate = np.random.uniform(0.9, 1.1)  # speed change between 90% - 110%
    return librosa.effects.time_stretch(y=data, rate=rate)

# ==============================
# FEATURE EXTRACTION FUNCTION
# ==============================
def extract_features(file_path, sample_rate):
    X, sr = librosa.load(file_path, res_type='kaiser_fast', duration=DURATION, sr=sample_rate, mono=True)
    
    # Apply augmentation randomly
    if np.random.rand() < 0.3:
        X = add_noise(X)
    if np.random.rand() < 0.3:
        X = pitch_shift(X, sr)
    if np.random.rand() < 0.3:
        X = time_stretch(X)
    
    # MFCC
    mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=N_MFCC)
    # Chroma
    stft = np.abs(librosa.stft(X))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    # Spectral Contrast
    spec_contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr)
    
    # Combine all features
    combined = np.vstack([mfccs, chroma, spec_contrast, tonnetz]).T
    return combined

# ==============================
# FEATURE EXTRACTION: RAVDESS
# ==============================
for actor_folder in os.listdir(RAVDESS_PATH):
    actor_folder_path = os.path.join(RAVDESS_PATH, actor_folder)
    if not os.path.isdir(actor_folder_path):
        continue
    for file in os.listdir(actor_folder_path):
        file_path = os.path.join(actor_folder_path, file)
        emotion_code = file.split("-")[2]
        emotion = RAVDESS_EMOTIONS.get(emotion_code)
        if emotion not in COMMON_EMOTIONS:
            continue
        try:
            feats = extract_features(file_path, SAMPLE_RATE)
            features.append(feats)
            labels.append(emotion)
        except Exception as e:
            print(f"Skipping RAVDESS file {file_path} due to error: {e}")

# ==============================
# FEATURE EXTRACTION: CREMA-D
# ==============================
for file in os.listdir(CREMAD_PATH):
    if not file.endswith(".wav"):
        continue
    try:
        emotion_code = file.split("_")[2]  # e.g., "ANG" in "1001_DFA_ANG_XX.wav"
        emotion = CREMAD_EMOTIONS.get(emotion_code)
        if emotion not in COMMON_EMOTIONS:
            continue
        file_path = os.path.join(CREMAD_PATH, file)
        feats = extract_features(file_path, SAMPLE_RATE)
        features.append(feats)
        labels.append(emotion)
    except Exception as e:
        print(f"Skipping CREMA-D file {file_path} due to error: {e}")

print("Feature extraction completed for both datasets with augmentation.")

# ==============================
# PREPROCESSING
# ==============================
max_len = max([f.shape[0] for f in features])
features_padded = np.array([np.pad(f, ((0, max_len - f.shape[0]), (0,0)), mode='constant') for f in features])

X = features_padded
y = np.array(labels)

if len(X) == 0 or len(y) == 0:
    raise ValueError("No valid audio files found. Check your dataset paths and files!")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Scale features
num_samples, time_steps, num_features = X.shape
X_2d = X.reshape(num_samples, time_steps*num_features)
scaler = StandardScaler()
X_scaled_2d = scaler.fit_transform(X_2d)
X_scaled = X_scaled_2d.reshape(num_samples, time_steps, num_features)

# Save scaler & label encoder
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

# ==============================
# LSTM MODEL
# ==============================
model = Sequential()
model.add(LSTM(128, input_shape=(X_scaled.shape[1], X_scaled.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint("model_augmented.h5", monitor='loss', save_best_only=True, verbose=1)

# ==============================
# TRAIN
# ==============================
model.fit(X_scaled, y_categorical, epochs=100, batch_size=32, callbacks=[checkpoint])

print("Training completed! Model saved as model_augmented.h5")
