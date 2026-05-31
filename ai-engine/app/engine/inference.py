import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, List

class AIInferencePipeline:
    """Processes frame geography and maps structural facial landmarks."""
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        # Index lists for specific facial components
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.MOUTH = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

    def _calculate_ear_metric(self, landmarks, indices: List[int], width: int, height: int) -> float:
        """Calculates the Eye Aspect Ratio (EAR) based on vertical and horizontal distances."""
        points = [np.array([landmarks[i].x * width, landmarks[i].y * height]) for i in indices]
        # Vertical distances
        v1 = np.linalg.norm(points[3] - points[13])
        v2 = np.linalg.norm(points[5] - points[11])
        # Horizontal distance
        h = np.linalg.norm(points[0] - points[8])
        return float((v1 + v2) / (2.0 * h + 1e-6))

    def _calculate_mar_metric(self, landmarks, indices: List[int], width: int, height: int) -> float:
        """Calculates the Mouth Aspect Ratio (MAR) to monitor yawning behavior."""
        points = [np.array([landmarks[i].x * width, landmarks[i].y * height]) for i in indices]
        v = np.linalg.norm(points[5] - points[15])  # Vertical mouth opening
        h = np.linalg.norm(points[0] - points[10])  # Horizontal mouth width
        return float(v / (h + 1e-6))

    def _estimate_head_pose(self, landmarks, width: int, height: int) -> Tuple[float, float]:
        """Maps 3D land-mesh nodes to estimate approximate Yaw and Pitch angles."""
        nose_tip = landmarks[1]
        chin = landmarks[152]
        left_eye_corner = landmarks[33]
        right_eye_corner = landmarks[263]
        
        # Compute horizontal vs vertical landmark vectors
        eye_center_x = (left_eye_corner.x + right_eye_corner.x) * 0.5 * width
        nose_x = nose_tip.x * width
        
        # Geometric tracking approximation for Yaw and Pitch
        yaw = float((nose_x - eye_center_x) * 1.8)
        pitch = float(((nose_tip.y - (left_eye_corner.y + right_eye_corner.y) * 0.5) * height) * 1.4)
        
        return np.clip(yaw, -90.0, 90.0), np.clip(pitch, -90.0, 90.0)

    def process_frame(self, frame: np.ndarray) -> Tuple[Dict, np.ndarray]:
        """Runs the image frame through the facial landmark pipeline."""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        # Default fallback metrics payload if no face is detected
        metrics = {"ear": 0.30, "mar": 0.15, "yaw": 0.0, "pitch": 0.0, "face_detected": False}
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            metrics["face_detected"] = True
            
            ear_l = self._calculate_ear_metric(landmarks, self.LEFT_EYE, w, h)
            ear_r = self._calculate_ear_metric(landmarks, self.RIGHT_EYE, w, h)
            metrics["ear"] = float((ear_l + ear_r) * 0.5)
            metrics["mar"] = self._calculate_mar_metric(landmarks, self.MOUTH, w, h)
            
            yaw, pitch = self._estimate_head_pose(landmarks, w, h)
            metrics["yaw"] = yaw
            metrics["pitch"] = pitch
            
            # Simulated model classifications (replace with explicit weights if needed)
            metrics["model_eyes_closed"] = 1 if metrics["ear"] < 0.18 else 0
            metrics["model_yawning"] = 1 if metrics["mar"] > 0.58 else 0
            metrics["model_fatigued"] = 1 if metrics["ear"] < 0.22 and metrics["mar"] > 0.40 else 0
            
            # Render basic diagnostic tracking visual overlays onto the video output
            for idx in [1, 33, 263, 152]:  # Draw primary anchor markers
                pt = landmarks[idx]
                cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 2, (0, 255, 0), -1)
                
        return metrics, frame

ai_inference = AIInferencePipeline()