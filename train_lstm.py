import os
import argparse
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
RAVDESS_PATH = r"C:\Users\mando\Downloads\emodio\ravdess"
CREMAD_PATH = r"C:\Users\mando\Downloads\emodio\cremad\AudioWAV"

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
    "ANG": "angry",
    "HAP": "happy",
    "SAD": "sad",
    "FEA": "fearful",
    "DIS": "disgust",
    "NEU": "neutral"
}



features = []
labels = []

print("Script started!")

def main():
    parser = argparse.ArgumentParser(description="Train LSTM on emotion audio datasets")
    parser.add_argument('--data-dir', type=str, default='', help='Root folder with subfolders per emotion (overrides dataset-specific paths)')
    parser.add_argument('--ravdess-path', type=str, default=RAVDESS_PATH, help='Path to RAVDESS root')
    parser.add_argument('--cremad-path', type=str, default=CREMAD_PATH, help='Path to CREMA-D wav folder')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--synthetic-test', action='store_true', help='Run a synthetic-data smoke test instead of loading files')
    parser.add_argument('--classes', nargs='+', default=[], help='List of class names to use for synthetic test (default: common emotions)')
    parser.add_argument('--synthetic-samples', type=int, default=20, help='Samples per class for synthetic test')
    args = parser.parse_args()

    global RAVDESS_PATH, CREMAD_PATH
    RAVDESS_PATH = args.ravdess_path
    CREMAD_PATH = args.cremad_path

    # -----------------------------
    # Load features
    # -----------------------------
    features = []
    labels = []

    if args.data_dir:
        print(f"Loading data from folder-per-class root: {args.data_dir}")
        feats, labs = load_data_from_folder_per_class(args.data_dir, SAMPLE_RATE)
        features.extend(feats)
        labels.extend(labs)
    elif args.synthetic_test:
        class_names = args.classes if len(args.classes) > 0 else COMMON_EMOTIONS
        print(f"Generating synthetic dataset for classes: {class_names} ({args.synthetic_samples} samples/class)")
        feats, labs = synthetic_dataset(class_names, samples_per_class=args.synthetic_samples)
        features.extend(feats)
        labels.extend(labs)
    else:
        print("Using dataset-specific paths for RAVDESS and CREMA-D")

        # ==============================
        # FEATURE EXTRACTION: RAVDESS
        # ==============================
        if os.path.exists(RAVDESS_PATH):
            for actor_folder in os.listdir(RAVDESS_PATH):
                actor_folder_path = os.path.join(RAVDESS_PATH, actor_folder)
                if not os.path.isdir(actor_folder_path):
                    continue
                for file in os.listdir(actor_folder_path):
                    # skip non-wav files and malformed names
                    if not file.lower().endswith('.wav'):
                        continue
                    parts = file.split("-")
                    if len(parts) < 3:
                        continue
                    file_path = os.path.join(actor_folder_path, file)
                    emotion_code = parts[2]
                    emotion = RAVDESS_EMOTIONS.get(emotion_code)
                    if emotion not in COMMON_EMOTIONS:
                        continue
                    try:
                        feats = extract_features(file_path, SAMPLE_RATE)
                        features.append(feats)
                        labels.append(emotion)
                    except Exception as e:
                        print(f"Skipping RAVDESS file {file_path} due to error: {e}")
        else:
            print(f"RAVDESS path not found: {RAVDESS_PATH} (skipping)")

        # ==============================
        # FEATURE EXTRACTION: CREMA-D
        # ==============================
        if os.path.exists(CREMAD_PATH):
            for file in os.listdir(CREMAD_PATH):
                if not file.lower().endswith(".wav"):
                    continue
                try:
                    parts = file.split("_")
                    if len(parts) < 3:
                        continue
                    emotion_code = parts[2].upper().strip()  # ANG, HAP, SAD, FEA, DIS, NEU
                    emotion = CREMAD_EMOTIONS.get(emotion_code)

                    if emotion not in COMMON_EMOTIONS:
                        continue

                    file_path = os.path.join(CREMAD_PATH, file)
                    feats = extract_features(file_path, SAMPLE_RATE)
                    features.append(feats)
                    labels.append(emotion)

                except Exception as e:
                    print(f"Skipping CREMA-D file {file} due to error: {e}")
        else:
            print(f"CREMA-D path not found: {CREMAD_PATH} (skipping)")



    # -----------------------------
    # PREPROCESSING
    # -----------------------------
    if len(features) == 0 or len(labels) == 0:
        raise ValueError("No valid audio files found. Check your dataset paths and files!")

    max_len = max([f.shape[0] for f in features])
    features_padded = np.array([np.pad(f, ((0, max_len - f.shape[0]), (0,0)), mode='constant') for f in features])

    X = features_padded
    y = np.array(labels)

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
    model.fit(X_scaled, y_categorical, epochs=args.epochs, batch_size=args.batch_size, callbacks=[checkpoint])

    print("Training completed! Model saved as model_augmented.h5")


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


def synthetic_dataset(class_names, samples_per_class=20, duration=DURATION, sample_rate=SAMPLE_RATE):
    """Generate a tiny synthetic dataset using noise + augmentations for smoke-testing the pipeline."""
    feats = []
    labs = []
    for label in class_names:
        for i in range(samples_per_class):
            # start with low-amplitude noise as a base 'audio'
            data = np.random.normal(0, 0.01, int(duration * sample_rate))
        

            # extract features directly from array using librosa (no file)
            try:
                mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=N_MFCC)
                stft = np.abs(librosa.stft(data))
                chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
                spec_contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
                tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sample_rate)
                combined = np.vstack([mfccs, chroma, spec_contrast, tonnetz]).T
                feats.append(combined)
                labs.append(label)
            except Exception as e:
                print(f"Synthetic sample generation failed: {e}")
    return feats, labs


def load_data_from_folder_per_class(root_dir, sample_rate):
    """Load audio files organized as root_dir/<label>/*.wav"""
    feats = []
    labs = []
    if not root_dir:
        return feats, labs
    if not os.path.exists(root_dir):
        print(f"Warning: data dir not found: {root_dir}")
        return feats, labs

    for label in os.listdir(root_dir):
        label_dir = os.path.join(root_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if not fname.lower().endswith('.wav'):
                continue
            file_path = os.path.join(label_dir, fname)
            try:
                f = extract_features(file_path, sample_rate)
                feats.append(f)
                labs.append(label)
            except Exception as e:
                print(f"Skipping file {file_path} due to error: {e}")
    return feats, labs


if __name__ == '__main__':
    main()

