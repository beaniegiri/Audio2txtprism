import torch
import soundfile as sf
import numpy as np
from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor
from datasets import load_dataset
import os

os.environ["PATH"] += os.pathsep + r"C:\Users\binis\OneDrive\Documents\PRISM\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin"

repo_id = "mistralai/Voxtral-Mini-4B-2602"

processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralRealtimeForConditionalGeneration.from_pretrained(repo_id, device_map="auto", torch_dtype=torch.float16)
print("model is running")
print(torch.cuda.is_available())

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
audio, sr = sf.read(r"C:\Users\binis\OneDrive\Documents\PRISM\Audio2txtprism\audio.wav")
print("audio")

#If stereo, convert to mono by averaging the channels
if audio.ndim ==2:
    audio = np.mean(audio, axis=1)
    print("averaged")

# Convert to float32 (VERY IMPORTANT)
audio = audio.astype(np.float32)

# Resample if needed
if sr != 16000:
    import librosa
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    print("resampled")

inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
inputs = inputs.to(model.device, dtype=model.dtype)

print("inputed")
outputs = model.generate(**inputs, max_new_tokens=200)
decoded_outputs = processor.batch_decode(outputs, skip_special_tokens=True)


print(decoded_outputs[0])