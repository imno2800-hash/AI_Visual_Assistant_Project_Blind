import time

class DecisionMaker:
    def __init__(self, cooldown=4):
        self.last_spoken = {}
        self.cooldown = cooldown

        self.arabic = {
            "person": "شخص",
            "car": "سيارة",
            "chair": "كرسي",
            "bottle": "زجاجة"
        }

        self.priority = {
            "car": 1,
            "person": 2,
            "chair": 3
        }

    def _can_speak(self, key):
        now = time.time()
        last = self.last_spoken.get(key, 0)
        if now - last >= self.cooldown:
            self.last_spoken[key] = now
            return True
        return False

    def get_direction(self, bbox, frame_width):
        x1, _, x2, _ = bbox
        center_x = (x1 + x2) / 2

        if center_x < frame_width / 3:
            return "على يسارك"
        elif center_x > 2 * frame_width / 3:
            return "على يمينك"
        else:
            return "أمامك"

    def get_distance(self, bbox, frame_area):
        x1, y1, x2, y2 = bbox
        box_area = (x2 - x1) * (y2 - y1)
        ratio = box_area / frame_area

        if ratio > 0.20:
            return "قريب جدًا"
        elif ratio > 0.08:
            return "قريب"
        elif ratio > 0.03:
            return "متوسط"
        else:
            return "بعيد"

    def choose_message(self, detections, frame_width, frame_area):
        if not detections:
            return None

        detections = sorted(
            detections,
            key=lambda d: self.priority.get(d["label"], 99)
        )

        for d in detections:
            if d["confidence"] < 0.6:
                continue

            label = d["label"]
            if not self._can_speak(label):
                continue

            name = self.arabic.get(label, label)
            direction = self.get_direction(d["bbox"], frame_width)
            distance = self.get_distance(d["bbox"], frame_area)

            return f"{name} {distance} {direction}"

        return None
