import cv2
import json
import time
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.engine.risk_scoring import DynamicCognitiveRiskEngine
from app.engine.inference import ai_inference

app = FastAPI(title=settings.PROJECT_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared single-thread worker for processing heavy frames outside the event loop
executor = ThreadPoolExecutor(max_workers=1)

# Initialize Risk Scoring Engine matching settings specifications
risk_engine = DynamicCognitiveRiskEngine(
    calibration_window_frames=settings.CALIBRATION_FRAMES
)

latest_frame = None
last_flush = 0.0

# Safe, pre-populated default schema state matching frontend data structures
latest_telemetry = {
    "fatigue_probability": 0.0,
    "attention_score": 1.0,
    "micro_sleep_incidents": 0,
    "distraction_events": 0,
    "yawn_events": 0,
    "driver_state": "CALIBRATING",
    "average_attention": 1.0,
    "average_fatigue": 0.0,
    "longest_distraction_seconds": 0.0,
    "session_duration_minutes": 0.0,
    "calibrating": 1.0,
    "calibration_progress": 0.0,
    "xai_explanation": "Initializing system telemetry channels..."
}

async def video_capture_loop():
    """
    Asynchronous ingestion loop that uses custom buffer flushing 
    to completely eliminate video stream latency.
    """
    global latest_frame, latest_telemetry, last_flush

    source = settings.VIDEO_SOURCE
    # Handle both string IP cameras (DroidCam) and integer webcam indexes seamlessly
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    
    # Force OS/OpenCV level single-frame buffer depth to prevent processing backlogs
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"[NETRA] Multi-threaded video pipeline streaming operational from source: {source}")

    loop = asyncio.get_running_loop()

    try:
        while True:
            # ANTI-LAG MECHANISM: Explicitly read multiple frames out of the buffer 
            # to discard older cached items and target the absolute newest frame.
            for _ in range(2):
                cap.grab()

            ret, frame = cap.retrieve()
            if not ret or frame is None:
                await asyncio.sleep(0.01)
                continue

            # Run blocking MediaPipe FaceMesh extraction inside the worker thread pool
            metrics, annotated = await loop.run_in_executor(
                executor, ai_inference.process_frame, frame
            )
            
            latest_frame = annotated

            # COMPATIBILITY ALIGNMENT: Calculate risk scoring telemetry from your engine instance
            telemetry = risk_engine.process_kinematics(
                ear=metrics["ear"],
                mar=metrics["mar"],
                yaw=metrics["yaw"],
                pitch=metrics["pitch"],
                model_eyes_closed=metrics.get("model_eyes_closed", 0),
                model_yawning=metrics.get("model_yawning", 0),
                model_fatigued=metrics.get("model_fatigued", 0),
                face_detected=metrics.get("face_detected", True)
            )

            now = time.time()
            # Feed continuous frame updates through during calibration; throttle output after
            should_flush = (now - last_flush > 1.0) or (not risk_engine.is_calibrated)

            if should_flush:
                telemetry.update({
                    "raw_ear": metrics["ear"],
                    "raw_mar": metrics["mar"],
                    "yaw": metrics["yaw"],
                    "pitch": metrics["pitch"],
                    "gaze": metrics.get("gaze", 0.0),
                    "is_calibrated": risk_engine.is_calibrated,
                    "calibrating": 1.0 if not risk_engine.is_calibrated else 0.0,
                    "calibration_progress": min(
                        1.0,
                        len(risk_engine.history_ear) / risk_engine.calibration_window
                    )
                })
                latest_telemetry = telemetry
                last_flush = now

            # Tiny yield step allows context-switching to handle incoming API and WebSocket requests
            await asyncio.sleep(0.001)

    except Exception as e:
        print(f"[PIPELINE ERROR CRITICAL]: {str(e)}")
    finally:
        cap.release()
        executor.shutdown(wait=False)

@app.on_event("startup")
async def startup():
    asyncio.create_task(video_capture_loop())

def generate_stream():
    """Encodes active images from memory and streams frames at ~30 FPS."""
    global latest_frame
    while True:
        if latest_frame is None:
            time.sleep(0.03)
            continue

        ok, buffer = cv2.imencode(
            ".jpg",
            latest_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 75]  # Optimized image quality to save network bandwidth
        )

        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                buffer.tobytes() +
                b"\r\n"
            )
        time.sleep(0.033)

@app.get("/api/v1/video_feed")
def video_feed():
    return StreamingResponse(
        generate_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/api/v1/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            # Broadcast metrics back at a uniform frequency rate (~12.5 Hz)
            await ws.send_text(json.dumps(latest_telemetry))
            await asyncio.sleep(0.08)
    except WebSocketDisconnect:
        pass