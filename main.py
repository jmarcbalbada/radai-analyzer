from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont
import requests
from fastapi.responses import JSONResponse
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],  
    allow_headers=["*"],
)

YOLO_API_URL = "https://predict.ultralytics.com"
YOLO_API_KEY = "3b5056ac3a9ea918ac838037d777446ba97e9ad3fc"
MODEL_URL = "https://hub.ultralytics.com/models/SMt917G5PhT5W142f1Iq"

@app.get("/test")
def test_endpoint():
    return JSONResponse(content={"message": "Hello, this is a test endpoint!"})

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    print("Hit endpoint")
    try:
        image_bytes = await file.read()

        # Send to YOLOv8 API
        response = requests.post(
            YOLO_API_URL,
            headers={"x-api-key": YOLO_API_KEY},
            data={
                "model": MODEL_URL,
                "imgsz": 640,
                "conf": 0.25,
                "iou": 0.45,
            },
            files={"file": (file.filename, image_bytes)},
        )

        result = response.json()
        detections = result["images"][0]["results"]

        # Load image using PIL
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(image)

        # Load font
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()

        # Draw boxes
        for det in detections:
            box = det["box"]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            label = f"{det['name']} ({det['confidence']*100:.1f}%)"

            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1 - 10), label, fill="red", font=font)

        # Return image
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return Response(content=img_bytes.getvalue(), media_type="image/png")

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)