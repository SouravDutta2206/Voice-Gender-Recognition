import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
from utils.audio_filters import dfn_filter, snr_filter
import multiprocessing 
from functools import partial 


#----------------------------------MP3 to WAV Converter----------------------------------#

def process_file(row, denoise_audio=False, denoise_method=None):
    input_file_path = row['path']
    
    # Skip if file doesn't exist
    if not os.path.exists(input_file_path):
        print(f"File not found: {input_file_path}")
        return
        
    # Define output path
    output_file_path = input_file_path.replace('.mp3', '.wav')
    
    try:
        # Convert mp3 to wav
        audio = AudioSegment.from_mp3(input_file_path)
        audio.export(output_file_path, format="wav")
        
        # Optional: Remove the original .mp3 file
        os.remove(input_file_path)
        
        if denoise_audio:
            if denoise_method == 'snr':
                snr_filter(audio_path=output_file_path)
            elif denoise_method != 'dfn':  # Any other method
                # Apply other denoising methods
                pass
    
    except Exception as e:
        print(f"Error processing {input_file_path}: {str(e)}")

def ConvertMP3toWAV(dataframe, denoise_audio=False, denoise_method=None):
    dataframe.replace(to_replace='wav', value='mp3', inplace=True, regex=True)
    
    # For 'dfn' method, keep the original sequential processing
    if denoise_audio and denoise_method == 'dfn':
        pbar = tqdm(total=len(dataframe), unit='file')
        
        for _, row in dataframe.iterrows():
            input_file_path = row['path']
            
            # Skip if file doesn't exist
            if not os.path.exists(input_file_path):
                pbar.update(1)
                print(f"File not found: {input_file_path}")
                continue
                
            # Define output path
            output_file_path = input_file_path.replace('.mp3', '.wav')
            
            try:
                # Convert mp3 to wav
                audio = AudioSegment.from_mp3(input_file_path)
                audio.export(output_file_path, format="wav")
                
                # Optional: Remove the original .mp3 file
                os.remove(input_file_path)
                
                if denoise_audio:
                    dfn_filter(audio_path=output_file_path)
            
            except Exception as e:
                print(f"Error processing {input_file_path}: {str(e)}")
            
            pbar.update(1)
        
        pbar.close()
    
    # For other methods, use parallel processing
    else:
        # Determine number of CPU cores to use (leave one free for system)
        num_cores = max(1, multiprocessing.cpu_count() - 1)
        
        # Create a partial function with fixed parameters
        process_func = partial(
            process_file, 
            denoise_audio=denoise_audio, 
            denoise_method=denoise_method
        )
        
        # Process files in parallel with progress bar
        with multiprocessing.Pool(num_cores) as pool:
            list(tqdm(
                pool.imap(process_func, [row for _, row in dataframe.iterrows()]),
                total=len(dataframe),
                unit='file'
            ))

