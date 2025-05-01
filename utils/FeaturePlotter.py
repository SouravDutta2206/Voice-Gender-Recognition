import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import time
from tqdm import tqdm
from utils.utils import mfcc_to_image

def _npyToImage(file_info: str) -> None:

    """
    Process a single file to convert MFCC data to an image.
    Args:
        file_info (Any): File path and output path.
    """
    file_path, output_path = file_info
    
    try:
        # Load the data
        data = np.load(file_path)
        
        _ = mfcc_to_image(mfccs=data, output_path=output_path)
        
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False
    
def _find_files_to_process(data_directory: str, output_directory: str, max_to_process: int) -> list:
    """
    Find all files in the data directory that need processing.
    Args:
        data_directory (str): Directory containing the MFCC files.
        output_directory (str): Directory to save the output images.
        max_to_process (int): Maximum number of files to process.
    Returns:
        list: List of file paths and output paths.
    """
    file_list = []
    
    # Create output directories if they don't exist
    os.makedirs(os.path.join(output_directory, 'Female'), exist_ok=True)
    os.makedirs(os.path.join(output_directory, 'Male'), exist_ok=True)
    
    for root, _, files in os.walk(data_directory):
        for filename in tqdm(files):
            if not filename.endswith('.npy'):
                continue
                
            if filename.startswith('female'):
                output_path = os.path.join(output_directory, 'Female', filename.replace('.npy', '.jpg'))
            elif filename.startswith('male'):
                output_path = os.path.join(output_directory, 'Male', filename.replace('.npy', '.jpg'))
            else:
                continue
                
            # Skip if output already exists
            if os.path.exists(output_path):
                continue
                
            file_list.append((os.path.join(root, filename), output_path))
            
            if len(file_list) >= max_to_process:
                break
                
        if len(file_list) >= max_to_process:
            break
    
    return file_list

def ConvertMFCCToImage(data_directory: str, output_directory: str, max_to_process: int) -> None:
    """
    Convert MFCC data to images from npy files.
    Args:
        data_directory (str): Directory containing the MFCC files.
        output_directory (str): Directory to save the output images.
        max_to_process (int): Maximum number of files to process.
    """
    os.makedirs(output_directory, exist_ok=True)

    start_time = time.time()
    
    # Find all files that need processing
    print("Finding files to process...")
    file_list = _find_files_to_process(data_directory, output_directory, max_to_process)
    print(f"Found {len(file_list)} files to process")
    
    # Determine optimal number of workers based on CPU cores
    num_workers = os.cpu_count() - 1 or 1  # Leave one core free
    
    # Process files in parallel
    print(f"Processing with {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(_npyToImage, file_list),
            total=len(file_list),
            desc="Processing files"
        ))
    
    # Count successful conversions
    successful = results.count(True)
    
    end_time = time.time()
    print(f"Completed processing {successful} files")
    print(f"Total time: {end_time - start_time:.2f} seconds")

