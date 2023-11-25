from transformers import WhisperTokenizer, WhisperForConditionalGeneration
from pydub import AudioSegment
import torchaudio
import io
import torch
import sys

def transcribe_audio(audio_path):
    # Load the tokenizer and model
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

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

    print("Tokenizing the audio...")
    input_values = tokenizer(speech, return_tensors="pt").input_values
    print("Tokenization completed.")

    print("Generating transcription...")
    with torch.no_grad():
        logits = model(input_values).logits
    print("Transcription generation completed.")

    print("Decoding the transcription...")
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])
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
