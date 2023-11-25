import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Load the pre-trained model and tokenizer
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-ll60k")
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-ll60k")

# Load the WAV recording
wav_file = "path/to/meeting_recording.wav"

# Convert the WAV recording to text
input_values = tokenizer(wav_file, return_tensors="pt").input_values
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)[0]

# Print the transcription
print(transcription)
