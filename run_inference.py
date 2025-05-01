#!/usr/bin/env python3
import argparse
import sys
from utils.utils import load_model, audio_infer, detect_human_speech


def main():
    # Define available models
    available_models = ['CNN', 'inceptionV3', 'Xception', 'ResNet50', 'MobileNetV2', 'VGG16']
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Audio speech detection and classification')
    parser.add_argument('--input_file', '-i', type=str, required=True, 
                        help='Path to input WAV audio file')
    parser.add_argument('--model_name', '-m', type=str, required=True, choices=available_models,
                        help=f'Model name to use for inference. Must be one of: {", ".join(available_models)}')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input file has .wav extension
    if not args.input_file.lower().endswith('.wav'):
        parser.error("Input file must be a WAV file")
    
    # Detect human speech
    speech_type = detect_human_speech(args.input_file)
    if speech_type == 'Non-Human':
        print('No human speech detected')
        sys.exit(1)
    
    # Load model
    try:
        model, image_size = load_model(args.model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Run inference
    try:
        label, _, inference_time = audio_infer(
            file=args.input_file, 
            test_model=model, 
            image_size=image_size
        )
        
        # Print results
        print(f"Result: {label}")
        print(f"Inference time: {inference_time:.4f} seconds")
    
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()