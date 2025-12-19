import argparse
import os
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import joblib

# ------------------ Parameters ------------------
DURATION = 5                 # seconds
SAMPLE_RATE = 22050         # FIXED universal rate
N_MFCC = 40


# -------------------------------------------------
# Load model, scaler, label encoder
# -------------------------------------------------
def load_model_and_tools(model_path="model_augmented.h5",
                         scaler_path="scaler.pkl",
                         le_path="label_encoder.pkl"):

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    if not os.path.exists(le_path):
        raise FileNotFoundError(f"Label encoder file not found: {le_path}")

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(le_path)

    return model, scaler, label_encoder


# -------------------------------------------------
# Feature extraction (no tonnetz)
# -------------------------------------------------
def extract_features_from_audio(audio, sample_rate, n_mfcc=N_MFCC):

    # Normalize audio (VERY IMPORTANT)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

    stft = np.abs(librosa.stft(audio))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    spec_contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)

    # No tonnetz (it breaks prediction)
    features = np.vstack([mfccs, chroma, spec_contrast]).T
    return features


# -------------------------------------------------
# Prepare feature shape to match training
# -------------------------------------------------
def prepare_features_for_model(features, model, scaler):

    # Model input shape → (None, timesteps, features)
    _, time_steps, num_features = model.input_shape

    # Pad or trim time steps
    if features.shape[0] < time_steps:
        features = np.pad(features,
                          ((0, time_steps - features.shape[0]), (0, 0)),
                          mode="constant")
    else:
        features = features[:time_steps, :]

    # Scale using the same scaler used in training
    feat_2d = features.reshape(1, -1)
    feat_scaled_2d = scaler.transform(feat_2d)

    feat_scaled = feat_scaled_2d.reshape(1, time_steps, num_features)
    return feat_scaled


# -------------------------------------------------
# Predict from audio file
# -------------------------------------------------
def predict_from_file(file_path, model, scaler, label_encoder):

    audio, sr = librosa.load(file_path,
                             sr=SAMPLE_RATE,
                             duration=DURATION,
                             mono=True)

    features = extract_features_from_audio(audio, SAMPLE_RATE)
    features_scaled = prepare_features_for_model(features, model, scaler)

    probs = model.predict(features_scaled)[0]
    top_idx = np.argsort(probs)[::-1]

    results = [(label_encoder.inverse_transform([i])[0], float(probs[i]))
               for i in top_idx]

    return results


# -------------------------------------------------
# Predict from microphone
# -------------------------------------------------
def predict_from_recording(model, scaler, label_encoder):
    print("Recording started...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1)
    sd.wait()
    print("Recording finished!")

    audio = audio.flatten()

    features = extract_features_from_audio(audio, SAMPLE_RATE)
    features_scaled = prepare_features_for_model(features, model, scaler)

    probs = model.predict(features_scaled)[0]
    top_idx = np.argsort(probs)[::-1]

    results = [(label_encoder.inverse_transform([i])[0], float(probs[i]))
               for i in top_idx]

    return results


# -------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict emotion from audio")
    parser.add_argument("--file", "-f", type=str,
                        help="Path to audio file (wav, mp3). If omitted, microphone recording is used.")
    parser.add_argument("--model", type=str, default="model_augmented.h5")
    parser.add_argument("--scaler", type=str, default="scaler.pkl")
    parser.add_argument("--le", type=str, default="label_encoder.pkl")

    args = parser.parse_args()

    model, scaler, label_encoder = load_model_and_tools(
        args.model, args.scaler, args.le
    )

    try:
        if args.file:
            results = predict_from_file(args.file, model, scaler, label_encoder)
        else:
            results = predict_from_recording(model, scaler, label_encoder)

        print("\nTop predictions:")
        for label, prob in results[:5]:
            print(f"  {label}: {prob:.4f}")

        print(f"\nPredicted Emotion: {results[0][0]}")

    except Exception as e:
        print(f"\nError during prediction: {e}")


if __name__ == "__main__":
    main()

