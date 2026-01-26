from gtts import gTTS
from playsound import playsound
import uuid
import os
import threading

class Speaker:
    def __init__(self):
        self.is_speaking = False

    def speak(self, text):
        if self.is_speaking:
            return  # لا تكرر الصوت

        def _play():
            try:
                self.is_speaking = True
                filename = f"temp_{uuid.uuid4()}.mp3"
                tts = gTTS(text=text, lang="ar")
                tts.save(filename)
                playsound(filename)
                os.remove(filename)
            finally:
                self.is_speaking = False

        threading.Thread(target=_play, daemon=True).start()
