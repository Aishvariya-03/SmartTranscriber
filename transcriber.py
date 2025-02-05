import streamlit as st
import whisper
import json
import os

# Clear any initial output
st.set_page_config(page_title="Transcription App", layout="centered")
st.markdown("<style>body{background-color: #f4f4f4;}</style>", unsafe_allow_html=True)

# Hide the code blocks from appearing in the UI
for key in list(st.session_state.keys()):
    del st.session_state[key]
  
def transcribe_audio(file_path, model):
    """Transcribes audio using Whisper and returns text."""
    result = model.transcribe(file_path)
    return result["text"]

def save_transcription(file_name, transcription):
    """Saves transcription to a JSON file."""
    output_file = file_name.replace(".mp3", "_transcription.json").replace(".mp4", "_transcription.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({file_name: transcription}, f, indent=4)
    return output_file

# Streamlit UI
st.title("🎙️ Media File Transcription App")
st.write("Upload an audio/video file and get an instant transcription!")

# File uploader
uploaded_file = st.file_uploader("Choose a media file (MP3/MP4)", type=["mp3", "mp4"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    
    # Save uploaded file locally
    file_path = os.path.join("temp_" + uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # Load Whisper model
    st.info("Loading Whisper model... This may take a moment.")
    model = whisper.load_model("tiny")  # Smallest model for fast processing
    
    # Transcribe file
    st.info("Transcribing... Please wait.")
    transcription = transcribe_audio(file_path, model)
    
    # Display transcription
    st.subheader("📝 Transcription Output")
    st.text_area("Transcribed Text", transcription, height=250)
    
    # Save transcription to JSON
    output_file = save_transcription(file_path, transcription)
    
    # Provide a download button for the JSON
    with open(output_file, "rb") as f:
        st.download_button("Download Transcription (JSON)", f, file_name=output_file, mime="application/json")
    
    st.success("✅ Transcription complete!")