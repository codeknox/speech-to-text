from transformers import WhisperTokenizer, WhisperForConditionalGeneration
from pydub import AudioSegment
import torchaudio
import io
import torch
import sys

def transcribe_audio(audio_path):
    # Load the tokenizer and model with error handling
    try:
        tokenizer = WhisperTokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        model = WhisperForConditionalGeneration.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    except OSError as e:
        print("An error occurred while loading the Whisper tokenizer or model:")
        print(e)
        print("Please ensure you have an active internet connection and that the model name is correct.")
        print("If the problem persists, check for any local directories that might conflict with the model name.")
        sys.exit(1)

    print("Loading the WAV audio file...")
    speech, sample_rate = torchaudio.load(audio_path)
    print("Audio file loaded.")

    print("Tokenizing and generating transcription...")
    inputs = tokenizer(speech.squeeze(), return_tensors="pt", padding="longest", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
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
