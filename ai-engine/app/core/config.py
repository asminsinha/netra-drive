import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "NetraDrive AI Engine"
    API_V1_STR: str = "/api/v1"
    WS_V1_STR: str = "/api/v1/stream"
    
    # Video Source Configuration (Dynamic Override for DroidCam)
    # If using DroidCam over Wi-Fi, change this to your IP URL, e.g., "http://192.168.1.50:4747/video"
    # If using DroidCam via USB (Virtual Device Client), set it to an integer index like 0, 1, or 2.
    VIDEO_SOURCE: str = os.getenv("VIDEO_SOURCE", "http://192.168.0.104:4747/video")
    
    # Bayesian Context Baselines
    CALIBRATION_FRAMES: int = 1800  # Initial 60 seconds at 30 FPS to adapt to the driver's face
    
    class Config:
        case_sensitive = True

settings = Settings()