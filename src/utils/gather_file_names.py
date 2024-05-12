import os
import csv


# Function to get all audio files in a directory
def get_audio_files(directory):
    audio_files = []
    for file in os.listdir(directory):
        if (
            file.endswith(".mp3")
            or file.endswith(".wav")
            or file.endswith(".flac")
            or file.endswith(".ogg")
        ):
            audio_files.append(file)
    return audio_files


# Function to append audio file names to CSV
def append_to_csv(audio_files, csv_filename, class_label, existing_files):
    with open(csv_filename, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        for file in audio_files:
            if file not in existing_files:  # Check for duplicates
                csv_writer.writerow([file, class_label])
                existing_files.add(file)  # Add to set to track duplicates


def main():
    # Directory containing audio files
    directories = [
        "/home/daksh/Music/dataset/happy",
        "/home/daksh/Music/dataset/sad",
        "/home/daksh/Music/dataset/angry",
    ]

    # CSV filename
    csv_filename = "catagorized_song_dataset.csv"

    # Create CSV file with column headers if it doesn't exist
    if not os.path.exists(csv_filename):
        with open(csv_filename, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["song_name", "class"])

    # Set to store existing filenames in the CSV
    existing_files = set()

    # Process each directory
    for idx, directory in enumerate(directories):
        class_label = idx
        audio_files = get_audio_files(directory)
        append_to_csv(audio_files, csv_filename, class_label, existing_files)

    print("CSV file updated successfully with audio file names.")


if __name__ == "__main__":
    main()
