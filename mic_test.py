import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer

MODEL_PATH = r"C:\Personal_Folders\EchoAtlasAppV2\models\vosk-model-small-en-us-0.15"
model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, 16000)

q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

# Open microphone stream
with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                       channels=1, callback=callback):
    print("🎤 Speak into the microphone...")
    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            print("✅ Final:", result.get("text", ""))
        else:
            partial = json.loads(rec.PartialResult())
            if partial.get("partial"):
                print("...Partial:", partial["partial"])
