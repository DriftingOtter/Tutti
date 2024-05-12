import os
import json
import librosa
import numpy as np
import logging
import sys
from tensorflow.keras.models import load_model


# Function to extract features from audio
def extract_features(audio, sample_rate):
    if audio.ndim > 1:  # If the audio has multiple channels
        audio = audio.mean(axis=1)  # Convert to mono by averaging channels
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=44)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features


# Function to predict emotion from audio file
def predict_emotion(audio_file):
    # Load the emotion detection model
    model = load_model("/home/daksh/Documents/tutti/src/emotion_model/song_emotion_detection_model.keras")

    # Extract features from the audio file
    audio, sample_rate = librosa.load(audio_file, sr=48000, mono=False)
    prediction_feature = extract_features(audio, sample_rate)
    prediction_feature = prediction_feature.reshape(1, -1)

    # Make predictions using the model
    predictions = model.predict(prediction_feature)[0]

    # Unpack predictions
    happy, sad, angry = predictions

    # Get the name of the audio file
    song_name = os.path.basename(audio_file)

    return song_name, happy, sad, angry

def load_cache(cache_file):
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {cache_file}: {e}")
            return {}
    else:
        return {}

def save_cache(cache_file, cache_data):
    with open(cache_file, "w") as f:
        json.dump(cache_data, f)


def check_song_in_cache(song_name, cache_data):
    return song_name in cache_data


def add_record_to_cache(song_name, happy, sad, angry, cache_data):
    cache_data[song_name] = {"happy": happy, "sad": sad, "angry": angry}


def cache_song_record(song_emotion_cluster, cache_file):
    song_name, happy, sad, angry = song_emotion_cluster

    happy = float(happy)
    sad = float(sad)
    angry = float(angry)

    # Check if the cache file already exists
    cache_data = load_cache(cache_file)

    # Check if the song already exists in the cache
    if check_song_in_cache(song_name, cache_data):
        logging.info(f"Song '{song_name}' already exists in the cache. Skipping...")
        return

    # Check if any emotion value is missing
    if None in [happy, sad, angry]:
        raise ValueError("Emotion data is incomplete.")

    # Add the record to the cache
    add_record_to_cache(song_name, happy, sad, angry, cache_data)

    # Write the updated cache to file
    save_cache(cache_file, cache_data)

    logging.info(
        f"New record cached for '{song_name}' with happy={happy}, sad={sad}, and angry={angry}"
    )


def cache_all_songs_in_directory(directory, cache_file):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp3") or file.endswith(".wav"):
                audio_file_path = os.path.join(root, file)
                cache_song_record(predict_emotion(audio_file_path), cache_file)


def cli_main():
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 2:
        logging.error("Usage: python script.py <audio_file_path>")
        sys.exit(1)

    audio_file_path = sys.argv[1]
    emotion_cluster = predict_emotion(audio_file_path)
    print(f"{emotion_cluster[0]}: Happy Val:{emotion_cluster[1]}, ")


def api():
    music_directory = os.path.expanduser("~/Music")
    cache_file = "cache.json"
    cache_all_songs_in_directory(music_directory, cache_file)


if __name__ == "__main__":
    cli_main()
