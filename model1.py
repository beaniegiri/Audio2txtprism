<<<<<<< HEAD
import whispere
=======
import whisper
>>>>>>> TEST
import os

os.environ["PATH"] += os.pathsep + r"C:\Users\binis\OneDrive\Documents\PRISM\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin"


model = whisper.load_model("turbo")
result = model.transcribe("audio.wav")
print(result["text"])