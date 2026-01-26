from ultralytics import YOLO

# تحميل نموذج YOLOv8 خفيف
model = YOLO("yolov8n.pt")

# تدريب فعلي (Dataset رسمي صغير)
model.train(
    data="coco8.yaml",  # Dataset جاهز + Labels
    epochs=5,           # قليل عشان الوقت
    imgsz=640,
    batch=8
)

# تقييم النموذج + حفظ صور التنبؤ
model.val(save=True)
