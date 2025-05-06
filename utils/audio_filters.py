import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import librosa
import os
import soundfile as sf
from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file

def dfn_download_model(name: str = "DeepFilterNet3") -> str:
    """Download a DeepFilterNet model.

    Args:
        - name (str): Model name. Currently needs to one of `[DeepFilterNet, DeepFilterNet2]`.

    Returns:
        - base_dir: Return the model base directory as string.
    """
    if name.endswith(".zip"):
        name = name.removesuffix(".zip")
    model_dir = os.path.join("Models")
    os.makedirs(os.path.join(model_dir, name), exist_ok=True)
    if os.path.isfile(os.path.join(model_dir, name,"config.ini")) or os.path.isdir(
        os.path.join(model_dir, name,"checkpoints")
    ):
        return os.path.join(model_dir, name)
    url = f"https://github.com/Rikorose/DeepFilterNet/raw/main/models/{name}"
    download_file(url + ".zip", model_dir, extract=True)
    return os.path.join(model_dir, name)

def snr_filter(audio_path: str, 
               output_path: str = None, 
               n_fft: int = 2048, 
               hop_length: int = 512, 
               snr_threshold: float = 0.1) -> None:

    """
    Apply SNR-based filtering to an audio file.
    Args:
        audio_path (str): Path to the input audio file.
        output_path (str): Path to save the filtered audio file. If None, saves to the same path as input.
        n_fft (int): Number of FFT points. Default is 2048.
        hop_length (int): Number of samples between frames. Default is 512.
        snr_threshold (float): SNR threshold for filtering. Default is 0.1.
    """ 
    if output_path is None:
        output_path = audio_path

    audio_data , sr = librosa.load(audio_path, sr=None)
    
    stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(stft), np.angle(stft)

    noise_magnitude = np.mean(magnitude[:, :int(sr * 0.5 / hop_length)], axis=1, keepdims=True)

    noise_power = noise_magnitude ** 2
    signal_power = magnitude ** 2
    snr = signal_power / (noise_power + 1e-10)
    snr_mask = snr > snr_threshold
    filtered_magnitude = magnitude * snr_mask
    filtered_stft = filtered_magnitude * np.exp(1j * phase)
    filtered_audio = librosa.istft(filtered_stft, hop_length=hop_length)

    sf.write(output_path, filtered_audio*1.2, sr)

model_dir = dfn_download_model()

df_model, df_state, _ = init_df(model_base_dir=model_dir)

def dfn_filter(audio_path: str, output_path: str = None) -> None:
    """
    Apply DeepFilterNet denoising to an audio file.
    Args:
        audio_path (str): Path to the input audio file.
        output_path (str): Path to save the denoised audio file. If None, saves to the same path as input.
    """
    # Check if the model is initialized
    if output_path is None:
        output_path = audio_path

    audio, _ = load_audio(audio_path, sr=48000)
    # Denoise the audio
    enhanced = enhance(df_model, df_state, audio)    
    save_audio(output_path, enhanced, 48000)
    return output_path
