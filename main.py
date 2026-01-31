from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from ultralytics import YOLO
import io
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(title="Coffee Cup Counter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Cups-Detected"],
)
model = YOLO("yolov8n.pt")
CUP_CLASSES = [41]

def detect_and_draw(image: np.ndarray):
    results = model(image, conf=0.4, iou=0.5, verbose=False)
    count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls in CUP_CLASSES:
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, "Cup", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, count

@app.post("/process/image")
async def process_image(file: UploadFile = File(...)):
    data = await file.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    processed, count = detect_and_draw(image)

    _, buffer = cv2.imencode(".jpg", processed)
    img_bytes = io.BytesIO(buffer.tobytes())

    headers = {
        "X-Cups-Detected": str(count)
    }

    return StreamingResponse(
        img_bytes,
        media_type="image/jpeg",
        headers=headers
    )
