import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2
import time
import threading
from gtts import gTTS
from playsound import playsound
import tempfile

from vision.detector import ObjectDetector

# =========================
# 🗣️ ترجمة أسماء الأجسام للعربي
# =========================
AR_LABELS = {
    "person": "شخص",
    "car": "سيارة",
    "bus": "حافلة",
    "truck": "شاحنة",
    "motorcycle": "دراجة نارية",
    "bicycle": "دراجة",
    "chair": "كرسي",
    "bench": "مقعد",
    "dog": "كلب",
    "cat": "قط",
    "cell phone": "هاتف",
    "laptop": "حاسوب محمول",
    "tv": "تلفاز",
    "door": "باب",
    "cup": "كوب",
}

def translate_label(label):
    return AR_LABELS.get(label, label)

# =========================
# 🔊 صوت غير معلق (Thread)
# =========================
def speak_async(text):
    def run():
        try:
            tts = gTTS(text=text, lang="ar")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                filename = fp.name
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
        except Exception as e:
            print("خطأ صوت:", e)

    threading.Thread(target=run, daemon=True).start()

# =========================
# 🧠 الاتجاه والمسافة
# =========================
def get_direction(bbox, w):
    x1, _, x2, _ = bbox
    cx = (x1 + x2) / 2
    if cx < w / 3:
        return "على اليسار"
    elif cx > 2 * w / 3:
        return "على اليمين"
    else:
        return "أمامك"

def get_distance(bbox, area):
    x1, y1, x2, y2 = bbox
    a = (x2 - x1) * (y2 - y1)
    r = a / area
    if r > 0.2:
        return "قريب"
    elif r > 0.08:
        return "متوسط"
    else:
        return "بعيد"

# =========================
# 🎥 الكاميرا
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ الكاميرا لا تعمل")
    exit()

detector = ObjectDetector(conf=0.2)

print("✅ النظام يعمل — اضغط Q للخروج")

LAST_SPEAK = 0
INTERVAL = 4  # ثواني

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    area = h * w

    frame, detections = detector.detect(frame)

    if detections:
        # اختيار أقرب جسم
        d = max(
            detections,
            key=lambda x: (x["bbox"][2]-x["bbox"][0]) * (x["bbox"][3]-x["bbox"][1])
        )

        bbox = d["bbox"]
        label_en = d["label"]
        label_ar = translate_label(label_en)

        direction = get_direction(bbox, w)
        distance = get_distance(bbox, area)

        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(
            frame,
            f"{label_ar} | {direction} | {distance}",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

        now = time.time()
        if now - LAST_SPEAK >= INTERVAL:
            speak_async(f"يوجد {label_ar} {direction} وهو {distance}")
            LAST_SPEAK = now

    cv2.imshow("AI Visual Assistant - Live", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
