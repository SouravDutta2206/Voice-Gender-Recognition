import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import glob
import os

from utils.utils import process_csv, delete_files
from utils.FeatureExtraction import run_feature_extraction
from utils.FeaturePlotter import ConvertMFCCToImage
from utils.wavconverter import ConvertMP3toWAV

dataset_path = r'CommonVoice'

if __name__ == "__main__":

    # Convert mp3 to wav

    print("Processing csv files...")
    csv_files = glob.glob(os.path.join(dataset_path, '*.csv'))
    df = process_csv(csv_files)

    delete_files(df, dataset_path)

    choice = input("Do you want to convert mp3 to wav? (y/n): ")
    
    # df = dataset_path
    if choice.lower() == 'y':
        denoiser = input("Do you want to denoise the audio?\nPress 1 for DeepFilterNet\nPress 2 for SNR based filtering\nPress 3 for no denoising\n")
        if denoiser == '1':
            print("Converting mp3 to wav with DeepFilterNet denoising...")
            ConvertMP3toWAV(df, denoise_audio=True, denoise_method='dfn')
        elif denoiser == '2':
            print("Converting mp3 to wav with SNR based filtering...")
            ConvertMP3toWAV(df, denoise_audio=True, denoise_method='snr')
        else:
            print("Converting mp3 to wav without denoising...")
            ConvertMP3toWAV(df)
        print("Conversion completed.")
    else:
        print("Skipping mp3 to wav conversion.")
    
    # Extract features

    choice = input("Extract Features (MFCCs), this will extract and save all features as .npy files for further processing? (y/n): ")

    if choice.lower() == 'y':

        features_output_path = os.path.join(dataset_path, 'mfcc', 'npy_data')
        os.makedirs(features_output_path, exist_ok=True)
        run_feature_extraction(df, features_output_path)
        print("Processing completed.")
    else:
        print("Skipping feature extraction.")

    # Convert MFCCs to images
    
    choice = input("Do you want convert MFCCs to images? (y/n): ")

    if choice.lower() == 'y':

        print("Converting MFCCs to images...")
        mfcc_data_path = os.path.join(dataset_path, 'mfcc', 'npy_data')
        image_output_path = os.path.join(dataset_path, 'mfcc', 'mfcc_images')
        os.makedirs(image_output_path, exist_ok=True)
        ConvertMFCCToImage(data_directory=mfcc_data_path, output_directory=image_output_path, max_to_process=70000)
        print("Conversion completed.")
    else:
        print("Skipping MFCC to image conversion.")
    
    print("Exiting....")

    

    



