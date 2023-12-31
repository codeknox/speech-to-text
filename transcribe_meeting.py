from transformers import WhisperTokenizer, WhisperForConditionalGeneration, logging
logging.set_verbosity_error()
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
    # Convert mono or stereo audio to 128-channel audio
    if speech.shape[0] == 1:
        speech = speech.repeat(128, 1)
    elif speech.shape[0] == 2:
        speech = speech.repeat(64, 1)
    print("Audio file loaded.")

    print("Generating transcription...")
    with torch.no_grad():
        logits = model(speech).logits
    print("Generating transcription, logits done...")
    predicted_ids = torch.argmax(logits, dim=-1)
    print("Generating transcription, going to decode...")
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
