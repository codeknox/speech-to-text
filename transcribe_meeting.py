from transformers import WhisperTokenizer, WhisperForConditionalGeneration
from pydub import AudioSegment
import torchaudio
import io
import torch
import sys

def transcribe_audio(audio_path):
    # Load the tokenizer and model with error handling
    try:
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
    except OSError as e:
        print("An error occurred while loading the Whisper tokenizer or model:")
        print(e)
        print("Please ensure you have an active internet connection and that the model name is correct.")
        print("If the problem persists, check for any local directories that might conflict with the model name.")
        sys.exit(1)

    print("Loading the WAV audio file...")
    speech, sample_rate = torchaudio.load(audio_path)
    print("Audio file loaded.")

    print("Generating transcription...")
    with torch.no_grad():
        logits = model(speech).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])
    print("Decoding completed.")

    return transcription

if __name__ == "__main__":
    if len(sys.argv) != 2:
        # print("Usage: python transcribe_meeting.py <path_to_wav_file>")
        audio_path = "/Users/sibagy/Downloads/Meeting1.wav"
    else:
        audio_path = sys.argv[1]

    print("Starting transcription process...")
    transcription = transcribe_audio(audio_path)
    print("Transcription process completed.")
    print("Transcription:")
    print(transcription)
