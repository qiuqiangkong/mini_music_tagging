from typing import Union
import librosa
import torch
import torchaudio
import numpy as np


def load(
    path: str, 
    sr: int, 
    mono: bool = True,
    offset: float = 0., 
    duration: Union[float, None] = None
) -> np.ndarray:
    r"""Load audio.

    Returns:
       audio: (channels, audio_samples) 

    Examples:
        >>> audio = load_audio(path="xx/yy.wav", sr=16000)
    """

    # Prepare arguments
    orig_sr = librosa.get_samplerate(path)

    seg_start_sample = round(offset * orig_sr)

    if duration is None:
        seg_samples = -1
    else:
        seg_samples = round(duration * orig_sr)

    # Load audio
    audio, fs = torchaudio.load(
        path, 
        frame_offset=seg_start_sample, 
        num_frames=seg_samples
    )
    # (channels, audio_samples)

    # Resample. Faster than librosa
    audio = torchaudio.functional.resample(
        waveform=audio, 
        orig_freq=orig_sr, 
        new_freq=sr
    )
    # shape: (channels, audio_samples,)

    if mono:
        audio = torch.mean(audio, dim=0, keepdim=True)

    audio = audio.numpy()
    # shape: (channels, audio_samples,)

    return audio
