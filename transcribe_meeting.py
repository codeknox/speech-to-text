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

    print("Converting m4a file to WAV format...")
    audio_segment = AudioSegment.from_file(audio_path, format="m4a")
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)  # Set sample rate to 16kHz and mono channel
    buffer = io.BytesIO()
    audio_segment.export(buffer, format="wav")
    buffer.seek(0)
    print("Conversion to WAV completed.")

    print("Loading the WAV audio file...")
    speech, sample_rate = torchaudio.load(buffer, format="wav")
    print("Audio file loaded.")

    print("Tokenizing and generating transcription...")
    inputs = tokenizer(speech, return_tensors="pt", sampling_rate=sample_rate)
    with torch.no_grad():
        generated_ids = model.generate(**inputs)
    transcription = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("Decoding completed.")

    return transcription

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transcribe_meeting.py <path_to_wav_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    print("Starting transcription process...")
    transcription = transcribe_audio(audio_path)
    print("Transcription process completed.")
    print("Transcription:")
    print(transcription)
