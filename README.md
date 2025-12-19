# Emodio: AI Vocal Biomarkers For Telepathy
 
Transform voices into emotional insights using AI.

## Overview

**Emodio – Telepathy** is a real-time voice emotion recognition system that analyzes human speech and predicts emotional states using deep learning.  
The project provides a **complete end-to-end pipeline**, including model training, prediction, and a live graphical user interface (GUI).

This system is built as a **research and deployment prototype**, focusing on real-time processing and interpretability rather than clinical accuracy.

##  What is Telepathy?

In this project, **Telepathy** refers to the AI’s ability to infer human emotional states from vocal patterns.  
Using LSTM-based deep learning models, the system identifies emotional cues embedded in speech.

### Supported Emotions
- Neutral  
- Happy  
- Sad  
- Angry  
- Fearful

## Current Status

 **Prototype Stage**

- Trained on limited datasets  
- Accuracy is not clinically validated  
- Intended as a **baseline research pipeline** for emotion recognition  

---

##  Project Structure

```text
emodio/
│
├── realtime_gui_clean.py      # Real-time emotion recognition GUI
├── predict_voice.py           # Feature extraction and prediction logic
├── train_lstm.py              # LSTM model training script
│
├── model_augmented.h5         # Trained LSTM model
├── scaler.pkl                 # Feature scaler
├── label_encoder.pkl          # Emotion label encoder
│
├── emodio.png / elephant.jpg  # Optional GUI logo
├── requirements.txt
└── README.md

```
## Quick Start

### Install Dependencies
```bash
pip install numpy sounddevice librosa tensorflow scikit-learn joblib
```

### Train Model (requires datasets)
```bash
python3 train_lstm.py
```

### Run Prediction
```bash
python3 predict_voice.py
```

## Run Real-Time GUI
```bash
python realtime_gui.py
```

## Datasets Used

This project uses publicly available emotion speech datasets:

-RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
-CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)

## Business Potential

This technology can power:
- 🏥 Mental health monitoring apps
- 📞 Customer service quality analysis
- 🎮 Emotion-responsive games
- 📚 Interactive learning platforms
- 💼 HR interview analysis tools
 for detailed business strategy and improvement roadmap.

## Tech Stack

- **Deep Learning:** TensorFlow/Keras LSTM
- **Audio Processing:** Librosa
- **Features:** MFCC, Chroma, Spectral Contrast, Tonnetz




