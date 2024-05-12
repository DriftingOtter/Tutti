import os
from pydub import AudioSegment
import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from tqdm import tqdm


# Function to extract features from audio
def extract_features(audio, sample_rate):
    if audio.ndim > 1:  # If the audio has multiple channels
        audio = audio.mean(axis=1)  # Convert to mono by averaging channels
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=44)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features


# Function to load audio from file
def load_audio(file, duration):
    audio = AudioSegment.from_file(file)
    duration_seconds = (len(audio) / 1000) - duration
    duration = max(0, duration_seconds)
    audio_array, sample_rate = librosa.load(
        file, sr=48000, mono=False, duration=duration
    )
    return audio_array, sample_rate


class AudioProcessor:
    def __init__(self):
        pass

    def process_audio_data(
        self,
        csv_path,
        dataset_path,
        duration_adjustment=0.5,
        test_size=0.3,
        random_state=0,
    ):
        class_labels = {}  # Dictionary to store numerical labels for each class
        extracted_features = []
        label_counter = 0  # Counter to assign numerical labels

        with ThreadPoolExecutor() as executor:
            futures = []
            for class_dir in os.listdir(dataset_path):
                class_path = os.path.join(dataset_path, class_dir)
                if not os.path.isdir(class_path):
                    continue  # Skip non-directory files
                class_labels[class_dir] = label_counter
                label_counter += 1  # Increment label counter for next class
                futures.append(
                    executor.submit(
                        self.process_class_directory,
                        class_path,
                        class_labels[class_dir],
                        duration_adjustment,
                    )
                )

            for future in tqdm(
                futures, total=len(futures), desc="Processing audio files"
            ):
                class_features = future.result()
                extracted_features.extend(class_features)

        extracted_features_df = pd.DataFrame(
            extracted_features, columns=["feature", "class"]
        )
        X = np.array(extracted_features_df["feature"].tolist())
        y = np.array(extracted_features_df["class"].tolist())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test

    def process_class_directory(self, class_path, class_label, duration_adjustment):
        class_features = []
        for audio_file in os.listdir(class_path):
            file_path = os.path.join(class_path, audio_file)
            audio, sample_rate = load_audio(
                file_path, duration_adjustment
            )  # Adjusting duration here
            features = extract_features(audio, sample_rate)
            class_features.append([features, class_label])
        return class_features


def train_model(
    X_train_features, Y_train_labels, X_test_features, Y_test_labels, num_classes
):
    # Define model architecture
    model = Sequential(
        [
            Dense(1024, input_shape=(X_train_features.shape[1],), activation="relu"),
            Dense(512, activation="relu"),
            Dense(256, activation="relu"),
            Dense(
                num_classes, activation="softmax"
            ),  # Output layer with number of classes
        ]
    )

    # Print model summary
    model.summary()

    # Compile model
    model.compile(
        loss="sparse_categorical_crossentropy",  # Loss function for multi-class classification
        metrics=["accuracy"],
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    )

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        "best_model.h5", monitor="val_loss", verbose=1, save_best_only=True, mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=10, mode="min", restore_best_weights=True
    )

    # Train model
    model.fit(
        X_train_features,
        Y_train_labels,
        validation_data=(X_test_features, Y_test_labels),
        epochs=100,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    # Evaluate model
    loss, accuracy = model.evaluate(X_test_features, Y_test_labels)
    print("---------------------------------------------")
    print(f"Test Loss: {round(loss)}, Test Accuracy: {round(accuracy*100)}%")
    print("---------------------------------------------")

    return model


if __name__ == "__main__":
    # Set the paths
    dataset_path = "/home/daksh/Music/dataset"
    csv_path = "/home/daksh/Documents/tutti/src/catagorized_song_dataset.csv"

    # Create an instance of AudioProcessor
    audio_processor = AudioProcessor()

    # Preprocess audio data and split into train/test sets
    (
        X_train_features,
        X_test_features,
        Y_train_labels,
        Y_test_labels,
    ) = audio_processor.process_audio_data(csv_path, dataset_path)

    # Determine the number of classes from the maximum label value
    num_classes = np.max(np.concatenate((Y_train_labels, Y_test_labels))) + 1

    # Train model
    trained_model = train_model(
        X_train_features, Y_train_labels, X_test_features, Y_test_labels, num_classes
    )

    # Save model
    trained_model.save("song_emotion_detection_model.keras")
