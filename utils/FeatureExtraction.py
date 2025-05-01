import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import multiprocessing as mp
from functools import partial 


def extract_mfcc(file_name: str) -> np.ndarray:

    """
    Extracts MFCC features from an audio file.
    Args:
        file_name (str): Path to the audio file.
    Returns:
        np.ndarray: MFCC features.
    """
    max_len = 100
    X, sample_rate = librosa.core.load(file_name , sr=48000)

    X_trimmed, _ = librosa.effects.trim(X, top_db=20)
    mfccs = librosa.feature.mfcc(y=X_trimmed, sr=sample_rate, n_mfcc=40)
    
    if mfccs.shape[1] > max_len:
        mfccs = mfccs[:, :max_len]
    else:
        pad_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    return mfccs

def normalize_mfcc(mfcc: np.ndarray) -> np.ndarray:

    """
    Normalizes MFCC features.
    Args:
        mfcc (np.ndarray): MFCC features.
    Returns:
        np.ndarray: Normalized MFCC features.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("error")  # Raise warnings as exceptions
        try:
            mfccs_norm = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / np.std(mfcc, axis=1, keepdims=True)
        except Warning as e:
            mfccs_norm = mfcc

    return mfccs_norm

def process_file(row_data: any, output_path: str) -> None:

    """
    Processes a single audio file to extract and save MFCC features.
    Args:
        row_data (any): Tuple containing index and row data.
        output_path (str): Path to save the MFCC features.
    Returns:
        str: Index of the processed file or None if already exists.
    """
    
    i, (index, row) = row_data
    i = i - 35000 if i >= 35000 else i

    gender = row['gender']
    save_path = os.path.join(output_path, gender, f'{gender}_{i}.npy')

    if os.path.exists(save_path):
        return None
    
    try:
        mfccs = normalize_mfcc(extract_mfcc(row['path']))
        np.save(save_path, mfccs)
        return index
    except Exception as e:
        return f"Error processing {index}: {str(e)}"

def run_feature_extraction(dataframe: pd.DataFrame, output_path: str, num_processes=None) -> None:
    """
    Extracts MFCC features from audio files in a DataFrame and saves them as .npy files.
    Args:
        dataframe (DataFrame): Pandas DataFrame containing audio file paths and labels.
        output_path (str): Path to save the MFCC features.
        num_processes (int, optional): Number of processes to use for parallel processing. Defaults to None. If left default will use 1 less than the number of CPU cores (workers) available.
    """
    dataframe.replace(to_replace='mp3',value='wav', inplace=True , regex=True)
    os.makedirs(os.path.join(output_path, 'Female'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'Male'), exist_ok=True)

    if num_processes is None:
        num_processes = mp.cpu_count() - 1  # Leave one CPU free
    
    # Create a pool of workers
    pool = mp.Pool(processes=num_processes)
    
    # Prepare data for parallel processing
    data = list(enumerate(dataframe.iterrows()))
    
    # Create a partial function with the fixed argument
    process_func = partial(process_file, output_path=output_path)
    
    # Process files in parallel with progress bar
    results = list(tqdm(pool.imap(process_func, data), total=len(data)))
    
    # Close the pool
    pool.close()
    pool.join()
    
    
