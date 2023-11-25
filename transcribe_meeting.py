from transformers import WhisperTokenizer, WhisperForConditionalGeneration
import torchaudio
import torch
import sys

def transcribe_audio(audio_path):
    # Load the tokenizer and model
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

    # Load the audio file
    speech, sample_rate = torchaudio.load(audio_path)

    # Tokenize the raw audio
    input_values = tokenizer(speech, return_tensors="pt").input_values

    # Generate the transcription
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the predicted ids
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])

    return transcription

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transcribe_meeting.py <path_to_wav_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    transcription = transcribe_audio(audio_path)
    print(transcription)
