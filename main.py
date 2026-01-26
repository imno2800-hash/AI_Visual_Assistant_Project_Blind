import cv2
from ultralytics import YOLO
import pyttsx3
import time
import threading
from queue import Queue

# =============================
# إعداد محرك الصوت
# =============================
engine = pyttsx3.init(driverName='sapi5')
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

# طابور الصوت
speech_queue = Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

# تشغيل Thread الصوت
threading.Thread(target=speech_worker, daemon=True).start()

# =============================
# تحميل YOLO
# =============================
model = YOLO("models/yolov8n.pt")

# =============================
# فتح الكاميرا
# =============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ لم يتم فتح الكاميرا")
    exit()

# =============================
# إعدادات منطق النطق
# =============================
last_spoken = {}
SPEAK_DELAY = 4  # ثواني

translations = {
    "person": "شخص",
    "chair": "كرسي",
    "bottle": "زجاجة",
    "cup": "كوب",
    "laptop": "حاسوب",
    "cell phone": "هاتف",
    "tv": "تلفاز",
    "book": "كتاب",
    "car": "سيارة"
}

print("✅ النظام يعمل... اضغط Q للخروج")

# =============================
# الحلقة الرئيسية
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, verbose=False)
    current_objects = set()

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            current_objects.add(label)

            # رسم المربع
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # منطق النطق
            now = time.time()
            if label not in last_spoken or now - last_spoken[label] > SPEAK_DELAY:
                arabic = translations.get(label, label)
                speech_queue.put(f"أمامك {arabic}")
                last_spoken[label] = now

    # تنظيف الكائنات المختفية
    for obj in list(last_spoken.keys()):
        if obj not in current_objects:
            del last_spoken[obj]

    cv2.imshow("AI Blind Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =============================
# إغلاق نظيف
# =============================
speech_queue.put(None)
cap.release()
cv2.destroyAllWindows()
engine.stop()
