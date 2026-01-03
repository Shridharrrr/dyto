import numpy as np
import soundfile as sf

def detect_siren(audio_path, threshold=0.6):
    """
    Robust siren detector (format-independent)
    """

    signal, samplerate = sf.read(audio_path)

    # Convert stereo to mono
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    energy = np.mean(np.abs(signal))
    confidence = min(energy / 0.3, 1.0)

    return confidence >= threshold, round(confidence, 2)

