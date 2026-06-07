import numpy as np
import pandas as pd
import librosa

TARGET_SR = 16000
MAX_DURATION = 10.0
MAX_SAMPLES = int(TARGET_SR * MAX_DURATION)  # 160000


def resample_and_normalize(path: str, target_sr: int = TARGET_SR):
    """Load audio file, resample to target_sr, peak-normalize to [-1, 1]."""
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32), target_sr


def trim_and_pad(audio: np.ndarray, sr: int = TARGET_SR, max_duration: float = MAX_DURATION) -> np.ndarray:
    """Trim edge silence, clip to max_duration seconds, zero-pad if shorter."""
    max_samples = int(sr * max_duration)
    audio, _ = librosa.effects.trim(audio, top_db=30)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    elif len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)))
    return audio.astype(np.float32)


def write_manifest(rows: list, out_path: str) -> None:
    """Write list of row dicts to a CSV manifest file."""
    df = pd.DataFrame(rows, columns=["path", "acr", "ccr", "language", "system", "split", "source"])
    df["source"] = df["source"].fillna("unknown")
    df.to_csv(out_path, index=False)
