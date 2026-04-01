import whisper

model = whisper.load_model("turbo")
result = model.transcribe("audio2.wav")
print(result["text"])