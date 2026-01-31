import json
import os
from datetime import datetime

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph
from starlette.responses import FileResponse
from ultralytics import YOLO
import io
from fastapi.middleware.cors import CORSMiddleware

pdfmetrics.registerFont(TTFont("DejaVu", "DejaVuSans.ttf"))

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

HISTORY_FILE = "history.json"

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

def save_history(count: int, filename: str):
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    data.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "filename": filename,
        "cups_detected": count
    })

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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

    save_history(count, file.filename)

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

@app.get("/report/pdf")
def generate_pdf_report():
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    pdf_path = "report.pdf"
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="RussianTitle",
        fontName="DejaVu",
        fontSize=18,
        leading=22,
        spaceAfter=14
    ))

    styles.add(ParagraphStyle(
        name="RussianText",
        fontName="DejaVu",
        fontSize=12,
        leading=14
    ))
    doc = SimpleDocTemplate(pdf_path)

    story = [
        Paragraph("Отчет по подсчету стаканов", styles["RussianTitle"]),
        Paragraph(f"Всего запросов: {len(data)}", styles["RussianText"]),
    ]

    for i, item in enumerate(data, 1):
        text = (
            f"{i}. {item['timestamp']} — "
            f"файл: {item['filename']} — "
            f"стаканов: {item['cups_detected']}"
        )
        story.append(Paragraph(text, styles["RussianText"]))

    doc.build(story)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename="coffee_report.pdf"
    )

@app.get("/history")
def get_history():
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)