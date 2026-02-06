import sounddevice as sd
import numpy as np
import time

SAMPLE_RATE = 48000

def callback(indata, frames, time_info, status):
    if status:
        print(status)
    rms = np.sqrt(np.mean(indata**2))
    print(f"RMS: {rms:.6f}")

print("🎤 Testing microphone...")
print("Speak now. Press Ctrl+C to stop.")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    callback=callback
):
    while True:
        time.sleep(0.1)
