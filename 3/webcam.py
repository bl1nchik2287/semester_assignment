import cv2
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"): #По сути yolov8x.pt должна лучше обрабатывать
        self.model = YOLO(model_path)            #Но с yolov8n.pt программа банально работет быстрее (фпс стабильный)
        self.detected_objects = []

    def process_frame(self, frame):
        resized = cv2.resize(frame, (640, 480))
        results = self.model(resized, verbose=False)
        self._store_detections(results)
        return results[0].plot()

    def _store_detections(self, results):
        boxes = results[0].boxes
        frame_data = []

        for class_id, confidence in zip(boxes.cls.cpu().numpy().astype(int),
                                        boxes.conf.cpu().numpy()):
            frame_data.append({
                "class": self.model.names[class_id],
                "confidence": float(confidence),
                "timestamp": cv2.getTickCount() / cv2.getTickFrequency()
            })

        self.detected_objects.append(frame_data)
        print(f"Найдено: {[obj['class'] for obj in frame_data]}")


def main():
    detector = ObjectDetector()
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    video_capture.set(cv2.CAP_PROP_FPS, 30) #Попытался наладить фпс на веб-камере
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)

    if not video_capture.isOpened():
        print("Не удалось открыть камеру.")
        return

    print("Начало потока, чтобы выйти закройте окно с веб-камерой или нажмите Escape")

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                continue

            processed = detector.process_frame(frame)
            cv2.imshow("YOLOv8 Object Detection", processed)

            key = cv2.waitKey(1)
            if key == 27 or cv2.getWindowProperty("YOLOv8 Object Detection", cv2.WND_PROP_VISIBLE) < 1:
                break


    finally:
        video_capture.release()
        cv2.destroyAllWindows()
        print(f"Обработано кадров: {len(detector.detected_objects)}")
        print("Работа Завершена")


if __name__ == "__main__":
    main()
