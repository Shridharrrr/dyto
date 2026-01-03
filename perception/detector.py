from ultralytics import YOLO

# COCO vehicle classes we care about
VALID_CLASSES = {
    "motorcycle",
    "car",
    "bus",
    "truck"
}

class VehicleDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image):
        results = self.model(image, verbose=False)[0]
        counts = {cls: 0 for cls in VALID_CLASSES}

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]

            if cls_name in VALID_CLASSES:
                counts[cls_name] += 1

        return counts
