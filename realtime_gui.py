import threading
import queue
import os
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import font as tkfont
from PIL import Image, ImageTk

import sounddevice as sd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from predict_voice import (
    SAMPLE_RATE,
    DURATION,
    load_model_and_tools,
    extract_features_from_audio,
    prepare_features_for_model,
)


class RealTimeGUI:
    """Realtime GUI with live waveform and logo.

    - Shows a live waveform updated ~20Hz.
    - Highlights high-amplitude spikes.
    - Runs model predictions periodically and shows class probabilities.
    - Loads `emodio.png` from the same folder if present.
    """

    def __init__(self, model, scaler, label_encoder):
        self.model = model
        self.scaler = scaler
        self.le = label_encoder

        self.root = tk.Tk()
        self.root.title("Emodio - Live")

        # Top: logo + title
        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, padx=6, pady=6)

        # Logo (optional) - try several filenames (use the elephant image you shared)
        self.logo_label = ttk.Label(top)
        self.logo_label.pack(side=tk.LEFT, padx=(0,8))
        logo_candidates = ['emodio.png', 'emodio.jpg', 'elephant.png', 'elephant.jpg', 'elephant.jpeg']
        logo_path = None
        for fname in logo_candidates:
            if os.path.exists(fname):
                logo_path = fname
                break
        if logo_path:
            try:
                img = Image.open(logo_path)
                img.thumbnail((100, 100), Image.LANCZOS)
                self.logo_img = ImageTk.PhotoImage(img)
                self.logo_label.config(image=self.logo_img)
            except Exception:
                pass

        ttk.Label(top, text="Emodio — Live Emotion", font=(None, 16, 'bold')).pack(side=tk.LEFT, padx=8)

        # Canvas area: waveform + probability bars (slightly smaller)
        plots = ttk.Frame(self.root)
        plots.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.fig, (self.ax_wave, self.ax_bar) = plt.subplots(2, 1, figsize=(6, 3.2))
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plots)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bottom: controls and status
        bottom = ttk.Frame(self.root)
        bottom.pack(fill=tk.X, padx=6, pady=6)
        self.pred_var = tk.StringVar(value="Waiting for audio...")
        self.conf_var = tk.StringVar(value="")

        # Larger, more aesthetic fonts
        self.title_font = tkfont.Font(family='Segoe UI', size=18, weight='bold')
        self.small_font = tkfont.Font(family='Segoe UI', size=12)
        self.btn_font = tkfont.Font(family='Segoe UI', size=12, weight='bold')

        ttk.Label(bottom, textvariable=self.pred_var, font=self.title_font).pack(side=tk.LEFT)
        ttk.Label(bottom, textvariable=self.conf_var, font=self.small_font).pack(side=tk.LEFT, padx=(8, 0))

        btns = ttk.Frame(bottom)
        btns.pack(side=tk.RIGHT)
        # use tk.Button for direct font control
        self.start_btn = tk.Button(btns, text="Start", command=self.start, font=self.btn_font, bg='#4CAF50', fg='white')
        self.start_btn.pack(side=tk.LEFT, padx=6)
        self.stop_btn = tk.Button(btns, text="Stop", command=self.stop, state=tk.DISABLED, font=self.btn_font, bg='#F44336', fg='white')
        self.stop_btn.pack(side=tk.LEFT, padx=6)

        ttk.Label(bottom, text="Interval (s):", font=self.small_font).pack(side=tk.RIGHT, padx=(0,6))
        self.interval_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(bottom, from_=0.2, to=5.0, increment=0.2, textvariable=self.interval_var, width=5).pack(side=tk.RIGHT)

        # Internal state
        self.running = False
        self.buffer_size = int(SAMPLE_RATE * DURATION)
        self.buffer = np.zeros(self.buffer_size, dtype='float32')
        self.stream = None
        self.worker_q = queue.Queue()

        # Waveform initial plot
        t = np.linspace(-DURATION, 0, self.buffer_size)
        self.wave_line, = self.ax_wave.plot(t, np.zeros_like(t), lw=0.9, color='#1f77b4')
        self.spike_scatter = None
        self.ax_wave.set_ylim(-1.1, 1.1)
        self.ax_wave.set_xlabel('Time (s)', fontsize=9)
        self.ax_wave.tick_params(axis='both', which='major', labelsize=8)
        # aesthetic tweaks
        self.ax_wave.set_facecolor('#fbfbfb')
        for spine in ['top', 'right']:
            self.ax_wave.spines[spine].set_visible(False)

        # Probability bar placeholder
        self.classes = list(getattr(self.le, 'classes_', []))
        if self.classes:
            inds = np.arange(len(self.classes))
            self.bar_container = self.ax_bar.bar(inds, np.zeros(len(self.classes)), color='#2ca02c')
            self.ax_bar.set_xticks(inds)
            self.ax_bar.set_xticklabels(self.classes, rotation=45, ha='right', fontsize=8)
        else:
            self.bar_container = []
        self.ax_bar.set_ylim(0, 1)
        self.ax_bar.tick_params(axis='y', labelsize=8)
        self.ax_bar.set_facecolor('#fbfbfb')
        for spine in ['top', 'right']:
            self.ax_bar.spines[spine].set_visible(False)

        # Schedule rates
        self._wave_update_ms = 50

    def audio_callback(self, indata, frames, time_info, status):
        # indata shape: (frames, channels)
        mono = indata[:, 0]
        n = len(mono)
        # circular write
        self.buffer = np.roll(self.buffer, -n)
        self.buffer[-n:] = mono

    def start(self):
        if self.running:
            return
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        self.stream = sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=self.audio_callback)
        self.stream.start()

        # start visual updates and prediction loop
        self._schedule_wave_update()
        self.root.after(int(self.interval_var.get() * 1000), self._schedule_prediction)

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
        except Exception:
            pass

    def _schedule_wave_update(self):
        if not self.running:
            return
        try:
            self.update_waveform()
        finally:
            self.root.after(self._wave_update_ms, self._schedule_wave_update)

    def update_waveform(self):
        # copy buffer for thread-safety
        y = self.buffer.copy()
        # update line
        self.wave_line.set_ydata(y)
        # auto scale y-limit by observed max (keep at least 0.01)
        m = max(0.01, np.max(np.abs(y)))
        self.ax_wave.set_ylim(-m * 1.2, m * 1.2)

        # detect spikes: points above threshold (relative)
        thresh = 0.6 * m
        spike_idx = np.where(np.abs(y) >= thresh)[0]
        # remove previous scatter
        if self.spike_scatter is not None:
            try:
                self.spike_scatter.remove()
            except Exception:
                pass
            self.spike_scatter = None

        if len(spike_idx):
            times = np.linspace(-DURATION, 0, self.buffer_size)[spike_idx]
            vals = y[spike_idx]
            self.spike_scatter = self.ax_wave.scatter(times, vals, c='#e74c3c', s=36, edgecolors='none', alpha=0.9)

        self.canvas.draw_idle()

    def _schedule_prediction(self):
        if not self.running:
            return
        # run prediction in a background thread
        threading.Thread(target=self._predict_worker, daemon=True).start()
        # schedule next
        self.root.after(int(self.interval_var.get() * 1000), self._schedule_prediction)

    def _predict_worker(self):
        # copy buffer and run prediction
        try:
            audio = self.buffer.copy()
            features = extract_features_from_audio(audio, SAMPLE_RATE)
            feat = prepare_features_for_model(features, self.model, self.scaler)
            probs = self.model.predict(feat, verbose=0)[0]
            top = int(np.argmax(probs))
            label = self.le.inverse_transform([top])[0]
            conf = float(probs[top])
            # update UI in main thread
            self.root.after(0, lambda: self._update_prediction_ui(label, conf, probs))
        except Exception as e:
            self.root.after(0, lambda: self.pred_var.set(f"Error: {e}"))

    def _update_prediction_ui(self, label, conf, probs):
        self.pred_var.set(label)
        self.conf_var.set(f"Confidence: {conf:.2f}")
        # update bar chart
        if self.bar_container:
            for rect, p in zip(self.bar_container, probs):
                rect.set_height(float(p))
            self.ax_bar.set_ylim(0, max(1.0, float(max(probs))))
            self.canvas.draw_idle()

    def run(self):
        try:
            self.root.mainloop()
        finally:
            self.stop()


def main():
    print('Loading model and tools...')
    model, scaler, le = load_model_and_tools()
    gui = RealTimeGUI(model, scaler, le)
    gui.run()


if __name__ == '__main__':
    main()
