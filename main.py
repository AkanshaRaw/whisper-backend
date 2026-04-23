from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import os

app = FastAPI()

# Load lightweight model
model = WhisperModel("tiny")

@app.get("/")
def home():
    return {"message": "Whisper API running"}

@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        file_location = f"temp_{file.filename}"

        with open(file_location, "wb") as f:
            f.write(await file.read())

        segments, _ = model.transcribe(file_location)

        transcription = " ".join([seg.text for seg in segments])

        os.remove(file_location)

        return {
            "transcription": transcription,
            "feedback": "Good clarity, keep practicing!"
        }

    except Exception as e:
        return {"error": str(e)}
