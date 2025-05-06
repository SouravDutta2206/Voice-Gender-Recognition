import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import time
import gc
import librosa
import concurrent
import concurrent.futures
import pandas as pd
import numpy as np
from tqdm import tqdm
import speech_recognition as sr
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from utils.audio_filters import dfn_filter
from utils.FeatureExtraction import extract_mfcc, normalize_mfcc


def delete_files(dataframe: pd.DataFrame, input_directory: str) -> None:

    """
    Deletes files not in the dataframe.\n
    Args:
        dataframe (pd.DataFrame): DataFrame containing paths to files.
    """
    dataframe.replace(to_replace='wav',value='mp3', inplace=True , regex=True)

    valid_paths = dataframe['path'].values.tolist()

    dir_paths = []
    for root, _, files in os.walk(input_directory):
        for file in tqdm(files):
            file_path = os.path.join(root, file)
            if not file_path.endswith('.mp3'):
                continue
            else:
                dir_paths.append(file_path)

    set_valid_paths = set(valid_paths)
    set_files_to_delete = set(dir_paths)
    files_to_delete = list(set_files_to_delete - set_valid_paths)
    
    # Filter out paths that exist in valid_paths    
    print(f"Deleting {len(files_to_delete)} files...")
    
    def delete_file(file_path):
        try:
            os.remove(file_path)
            return True
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
            return False
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
        # Submit all deletion tasks
        future_to_file = {executor.submit(delete_file, file_path): file_path 
                          for file_path in files_to_delete}
        
        # Process as they complete with progress bar
        success_count = 0
        with tqdm(total=len(files_to_delete), desc="Deleting files") as pbar:
            for future in concurrent.futures.as_completed(future_to_file):
                if future.result():
                    success_count += 1
                pbar.update(1)

#--------------------------------------Process CSVs--------------------------------------#

def process_csv(csv_files: list) -> pd.DataFrame:

    """
    Process CSV files to extract paths and labels for audio files.\n
    Args:
        csv_files (list): List of CSV file paths.
    Returns:
        DataFrame: Pandas DataFrame containing paths and labels.
    """

    paths , labels = [] , []

    for j, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        new_df = df[['filename','gender']]
        new_df = new_df[np.logical_or(new_df['gender'] == 'female', new_df['gender'] == 'male')]
        for i , (index , rows) in tqdm(enumerate(new_df.iterrows()), total=len(new_df)):
            folder_name = rows['filename'].split('/')[0]
            paths.append(os.path.join('Commonvoice',folder_name,rows['filename'].replace('\\','/')))
            labels.append(rows['gender'])

    df = pd.DataFrame({'path':paths , 'dataset':'Commonvoice' , 'gender':labels}).sort_values(by=["gender"])
    df_female = df.drop(df[df['gender'] == 'male'].index)
    df_female = df_female.iloc[:-(len(df_female)-35000)]
    df_male = df.drop(df[df['gender'] == 'female'].index)
    df_male = df_male.iloc[:-(len(df_male)-35000)]
    df_CV = pd.concat([df_female,df_male])
    df_CV.replace(to_replace='mp3',value='wav', inplace=True , regex=True)
    return df_CV

#-------------------------------Plot Metrics after Training-------------------------------#

def plot_metrics(model_title: str, test_set: any, model: any, history: any = None) -> None:

    """
    Plot training and validation loss and accuracy and other metrics.\n
    Args:
        model_title (str): Title of the model. This will be used in the plot title.
        history (tf.keras.src.callbacks.history.History): Training history object.
        test_set (tf.keras.utils.image_dataset_from_directory): Test dataset.
        model (tf.keras.models.Sequential): Trained model.
    """
    if history is not None:

        fig, axs = plt.subplots(1, 2, figsize=(20, 5))
        plt.suptitle(f'{model_title} with CommonVoice MFCCs', size=15)
        results = pd.DataFrame(history.history)
        results[["loss", "val_loss"]].plot(ax=axs[0])
        axs[0].set_title("Validation loss {:.3f} (mean last 3)".format(np.mean(history.history["val_loss"][-3:])))
        results[["accuracy", "val_accuracy"]].plot(ax=axs[1])
        axs[1].set_title("Validation accuracy {:.3f} (mean last 3)".format(np.mean(history.history["val_accuracy"][-3:])))
        plt.show()

    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()

    predicted_labels , true_labels = [] , []

    for batch in test_set.as_numpy_iterator(): 
        X, y = batch
        yhat = model.predict(X,verbose=0)
        
        try:
            yhat = yhat.get('output_0')
        except:
            yhat = yhat
        
        for i in range(len(y)):
            # Get the individual true label
            true_label = y[i]
            
            # Get the predicted value and convert to binary label
            pred_value = yhat[i][0]
            pred_label = 1 if pred_value > 0.5 else 0
            
            # Append individual labels
            true_labels.append(true_label)
            predicted_labels.append(pred_label) 

        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)

    print(f"Model precision: {pre.result().numpy() * 100:.2f}%")
    print(f"Model recall: {re.result().numpy() * 100:.2f}%")
    print(f"Model accuracy: {acc.result().numpy() * 100:.2f}%")
    
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Female','Male'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_title} MFCC Test Set")
    plt.show()

#---------------------------------Human Speech Detection----------------------------------#


def detect_human_speech(audio_file: str) -> str:
    """
    Detect if the audio file contains human speech using Google Speech Recognition.
    Args:
        audio_file (str): Path to the audio file.
    Returns:
        str: "Human" if human speech is detected, "Non-Human" otherwise.
    """
   
    recognizer = sr.Recognizer()
   
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)  
    try:
        recognized_text = recognizer.recognize_google(audio)
        return "Human" if recognized_text else "Non-Human"
    except sr.UnknownValueError:
        return "Non-Human"  
    except sr.RequestError as e:
        print("Error:", e)
        return "Non-Human"  
    
#---------------------------------load data from directory----------------------------------#

def load_data(data_dir: str, batch_size: int = 32, image_size: tuple = (256, 256)):
    """
    Load audio data from a directory and return a TensorFlow dataset.
    
    Args:
        data_dir (str): Path to the directory containing audio files.
        batch_size (int): Batch size for the dataset.
        image_size (tuple): Size to which the images will be resized.
    
    Returns:
        tf.data.Dataset: A TensorFlow dataset containing the audio data.
    """
    # Load audio files and their labels
    dataset = tf.keras.utils.image_dataset_from_directory(data_dir, batch_size=batch_size, image_size=image_size)
    
    return dataset

#---------------------------------split data into train, val, test-------------------------------#

def split_data(data: any) -> dict:
    """
    Split the dataset into training, validation, and test sets.
    Args:
        data (any): The dataset to split.
    Returns:
        dict: A dictionary containing the training, validation, and test sets.
    """

    train_size = int(len(data)*.8)
    val_size = int(len(data)*.1)
    test_size = int(len(data)*.1)
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)

    return {
        "train": train,
        "val": val,
        "test": test,
    }

#---------------------------------load model from directory----------------------------------#

def load_model(model_name: str) -> tuple[tf.keras.models.Sequential, tuple]:
    """
    Load a pre-trained model from a specified path.
    
    Args:
        model_name (str): Name of the model to load.
    Returns:
        tuple: (model, image_size) where:
            - model: The loaded TensorFlow Sequential model
            - image_size: A tuple containing input image dimensions (height, width)
    """
    
    model = tf.keras.Sequential([tf.keras.layers.TFSMLayer(os.path.join(os.getcwd(), 'Models', model_name), call_endpoint='serving_default')])

    if model_name == 'VGG16':
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    else:    
        model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

    image_sizes = {
        'CNN': (256, 256),
        'inceptionV3': (299, 299),
        'Xception': (299, 299),
        'ResNet50': (224, 224),
        'MobileNetV2': (224, 224),
        'VGG16': (224, 224)
    }
    image_size = image_sizes.get(model_name)

    return model , image_size

#---------------------------------convert mfcc to image----------------------------------#

def mfcc_to_image(mfccs: np.ndarray, output_path: str = None) -> str:
    """
    Convert MFCC data to an image.
    
    Args:
        mfccs (numpy.ndarray): MFCC data to convert.
        output_path (str): Path to save the image. If None, a temporary path will be used.
    Returns:
        str: Path to the saved image.
    """
    if output_path:
        output_path = output_path
    else:
        output_path = 'temp.jpg'

    librosa.display.specshow(mfccs)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close('all')
    del mfccs
    gc.collect()

    return output_path

#---------------------------------make prediction on audio file-------------------------------#

def audio_infer(file: str, test_model: any, image_size: tuple = (256, 256)) -> tuple[str, str, float]:
    """
    Make a prediction on a single audio file.
    Args:
        file (str): Path to the audio file.
        test_model (any): Pre-trained model for prediction.
        image_size (tuple): Size to which the images will be resized.
    Returns:
        tuple: (label, gender, prediction_time) where:
            - label: Predicted label
            - gender: if being used with labeled data returns the actual label.
            - prediction_time: Time taken for prediction
    """
    # Extract gender from file path
    gender = file.split('/')[1]
    
    # Process audio and convert to image
    enhanced_audio = dfn_filter(file, 'enhanced.wav')
    mfccs = normalize_mfcc(extract_mfcc(enhanced_audio))
    mfcc_image_path = mfcc_to_image(mfccs)
    
    # Load and preprocess image in one step
    img = tf.io.decode_jpeg(tf.io.read_file(mfcc_image_path))
    img = tf.image.resize(img, image_size)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)
    
    # Time and make prediction
    start_time = time.time()
    prediction = test_model.predict(img, verbose=0)
    prediction_time = time.time() - start_time
    
    # Simplify label determination
    label = "Male" if list(prediction.values())[0] > 0.5 else "Female"
    
    # Clean up temporary files
    for temp_file in [enhanced_audio, mfcc_image_path]:
        os.remove(temp_file)
    
    return label, gender, prediction_time

#---------------------------------evaluate model on new data-------------------------------#

def evaluate_model(model_name: str, model: any, test_data_directory: str, image_size: tuple = (256, 256)) -> tuple[float, float]:
    """
    Evaluates model performance on test data.
    
    Args:
        model_name: Name of the model for display purposes
        model: Trained model to evaluate
        test_data_directory: Directory containing test audio files
        image_size: Input image size required by the model
        
    Returns:
        Accuracy and average prediction time
    """
    y_test = []
    y_pred = []
    pred_total_time = 0
    processed_files = 0
    
    # Walk through the test directory
    for root, dirs, files in os.walk(test_data_directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                try:
                    # Process and predict
                    label, gender, prediction_time = audio_infer(file_path, model, image_size)
                    
                    # Only add valid predictions
                    if label and gender:
                        y_test.append(gender)
                        y_pred.append(label)
                        pred_total_time += prediction_time
                        processed_files += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    # Ensure we have predictions before creating confusion matrix
    if not y_test or not y_pred:
        print("No valid predictions were made. Check your test data directory and prediction function.")
        return
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Female', 'Male'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name} MFCC External Data")
    plt.show()
    
    # Print metrics
    accuracy = accuracy_score(y_test, y_pred)
    avg_pred_time = pred_total_time / processed_files if processed_files > 0 else 0
    
    print(f"Model accuracy on new data: {accuracy * 100:.2f}%")
    print(f"Average prediction time: {avg_pred_time:.4f} seconds")
    print(f"Total files processed: {processed_files}")
    
    return accuracy, avg_pred_time




