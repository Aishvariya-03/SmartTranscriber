import os
import json
import whisper
import ffmpeg
from pathlib import Path
from tqdm import tqdm

# Function to find all media files in the given directory
def find_media_files(directory):
    """
    Recursively searches for audio and video files in the specified directory.
    Returns a list of file paths.
    """
    media_extensions = {".mp3", ".wav", ".flac", ".m4a", ".mp4", ".avi", ".mov", ".mkv"}
    media_files = []

    # Walk through all directories and find media files
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).suffix in media_extensions:
                media_files.append(os.path.join(root, file))
    
    return media_files

# Function to transcribe a media file
def transcribe_file(file_path, model):
    """
    Uses Whisper AI to transcribe a given audio/video file.
    Returns the transcribed text.
    """
    print(f"\nProcessing: {file_path}")
    try:
        result = model.transcribe(file_path) 
        return result["text"]
    except Exception as e:
        print(f"Error transcribing {file_path}: {e}")
        return None

# Main function
def main():
    """
    Scans a folder for media files, transcribes them using Whisper, and saves output as JSON.
    """
    # Get directory input from user
    input_directory = input("Enter the directory containing media files: ").strip()

    if not os.path.isdir(input_directory):
        print("Invalid directory. Please enter a valid path.")
        return

    # Find all media files
    media_files = find_media_files(input_directory)
    
    if not media_files:
        print("No media files found in the given directory.")
        return

    print(f"\nFound {len(media_files)} media files. Starting transcription...\n")

    # Load Whisper model (smallest version for speed)
    model = whisper.load_model("base")

    transcriptions = {}

    # Process each file with a progress bar
    for file in tqdm(media_files, desc="Transcribing files"):
        transcription = transcribe_file(file, model)
        if transcription:
            transcriptions[file] = transcription

    # Save transcriptions as JSON
    output_file = os.path.join(input_directory, "transcriptions.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(transcriptions, f, indent=4)

    print(f"\n✅ Transcription complete! Results saved to: {output_file}")

# Run the script
if __name__ == "__main__":
    main()