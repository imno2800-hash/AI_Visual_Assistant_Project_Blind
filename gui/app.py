import streamlit as st
import requests
import numpy as np
import cv2
from PIL import Image
import pyttsx3
import threading
import time
import subprocess
import sys

API_URL = "http://127.0.0.1:8000/detect/image"

# =========================
# إعداد الصفحة
# =========================
st.set_page_config(
    page_title="AI Visual Assistant",
    layout="wide"
)

st.title("🦯 AI Visual Assistant للمكفوفين")
st.write("كشف الأجسام مع تحديد الاتجاه والمسافة والنطق العربي")

# =========================
# 🔊 دالة الصوت (لوضع الصورة فقط)
# =========================
def speak_ar(text):
    def run():
        engine = pyttsx3.init()
        engine.setProperty("rate", 140)

        for voice in engine.getProperty("voices"):
            if "arab" in voice.name.lower() or "ar" in voice.id.lower():
                engine.setProperty("voice", voice.id)
                break

        engine.say(text)
        engine.runAndWait()
        engine.stop()

    threading.Thread(target=run, daemon=True).start()

# =========================
# إعداد الـ Cooldown
# =========================
COOLDOWN = 3

if "last_speech_time" not in st.session_state:
    st.session_state.last_speech_time = 0.0

# =========================
# دوال الاتجاه والمسافة
# =========================
def get_direction(bbox, frame_width):
    x1, _, x2, _ = bbox
    center_x = (x1 + x2) / 2
    if center_x < frame_width / 3:
        return "على اليسار"
    elif center_x > 2 * frame_width / 3:
        return "على اليمين"
    else:
        return "في المنتصف"

def get_distance(bbox, frame_area):
    x1, y1, x2, y2 = bbox
    box_area = (x2 - x1) * (y2 - y1)
    ratio = box_area / frame_area
    if ratio > 0.20:
        return "قريب جدًا"
    elif ratio > 0.08:
        return "متوسط"
    else:
        return "بعيد"

# =========================
# 🎥 الوضع المباشر (كاميرا حقيقية)
# =========================
st.markdown("---")
st.subheader("🎥 الوضع المباشر (كاميرا حقيقية + صوت)")

col_live1, col_live2 = st.columns(2)

# زر تشغيل
with col_live1:
    if st.button("▶️ تشغيل الكاميرا الحقيقية"):
        if "live_process" not in st.session_state or st.session_state.live_process is None:
            st.session_state.live_process = subprocess.Popen(
                [sys.executable, "gui/live_camera.py"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            st.success("✅ تم تشغيل الكاميرا")
        else:
            st.warning("⚠️ الكاميرا تعمل بالفعل")

# زر إيقاف
with col_live2:
    if st.button("⏹️ إيقاف الكاميرا"):
        if "live_process" in st.session_state and st.session_state.live_process is not None:
            st.session_state.live_process.terminate()
            st.session_state.live_process = None
            st.success("🛑 تم إيقاف الكاميرا")
        else:
            st.warning("⚠️ الكاميرا غير مشغلة")

# =========================
# 📷 الوضع الأول: صورة واحدة
# =========================
st.markdown("---")
st.subheader("📷 كشف صورة واحدة")

camera_image = st.camera_input("التقط صورة")

if camera_image is not None:
    image = Image.open(camera_image).convert("RGB")
    image_np = np.array(image)

    h, w = image_np.shape[:2]
    frame_area = h * w

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("الصورة الأصلية")
        st.image(image_np, use_column_width=True)

    if st.button("🔍 كشف الأجسام"):
        with st.spinner("⏳ جاري التحليل..."):
            response = requests.post(
                API_URL,
                files={
                    "file": (
                        "frame.jpg",
                        camera_image.getvalue(),
                        "image/jpeg"
                    )
                },
                timeout=30
            )

        if response.status_code == 200:
            data = response.json()
            detections = data["detections"]

            spoken_sentences = []

            for d in detections:
                bbox = d["bbox"]
                label = d["label"]

                direction = get_direction(bbox, w)
                distance = get_distance(bbox, frame_area)

                spoken_sentences.append(
                    f"يوجد {label} {direction} وهو {distance}"
                )

                x1, y1, x2, y2 = bbox
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image_np,
                    f"{label} | {direction} | {distance}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            with col2:
                st.subheader("النتيجة")
                st.image(image_np, use_column_width=True)

            now = time.time()
            if spoken_sentences and (now - st.session_state.last_speech_time >= COOLDOWN):
                speak_ar("تنبيه، " + " . ".join(spoken_sentences))
                st.session_state.last_speech_time = now

            st.subheader("📊 بيانات الكشف")
            st.json(data)

        else:
            st.error("❌ فشل الاتصال بالـ API")
