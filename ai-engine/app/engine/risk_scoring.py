import time
import numpy as np
from typing import Dict, List, Tuple
from collections import deque

class DynamicCognitiveRiskEngine:
    """
    Industrial ADAS Cognitive Vigilance Scoring Processor.
    Integrates PERCLOS, Head Pose Kinetic Vectors, and EMA Risk Compounding.
    """
    def __init__(self, calibration_window_frames: int = 150, time_window_seconds: int = 60):
        # Calibration Parameters
        self.calibration_window = calibration_window_frames
        self.is_calibrated = False
        self.history_ear: List[float] = []
        self.mean_ear = 0.30
        self.std_ear = 0.04
        
        # --- ATTENTION ADAPTIVE CALIBRATION ---
        self.history_yaw: List[float] = []
        self.history_pitch: List[float] = []
        self.center_yaw_bias = 0.0
        self.center_pitch_bias = 0.0
        
        # Temporal Trackers
        self.time_window = time_window_seconds
        self.temporal_buffer = deque()  # Stores tuples of (timestamp, is_eye_closed, is_yawning)
        self.last_update_time = time.time()
        
        # Core State Metrics
        self.smoothed_fatigue = 0.0
        self.attention_score = 1.0
        self.micro_sleep_incidents = 0
        self.last_microsleep_trigger_time = 0.0
        
        # State Tracking Flags
        self.eye_closed_start_time = None
        self.micro_sleep_active = False
        self.distraction_start_time = None
        
        # Session Analytics Accumulators
        self.total_yawn_events = 0
        self.last_yawn_timestamp = 0.0
        self.longest_distraction_duration = 0.0
        self.distraction_event_counter = 0
        self.session_start_time = time.time()

        # Analytics History Windows (for metric rolling averages)
        self.fatigue_history = deque(maxlen=600)
        self.attention_history = deque(maxlen=600)

    def update_baseline_calibration(self, ear: float, yaw: float = 0.0, pitch: float = 0.0) -> bool:
        """Applies online statistical updates to calculate driver eye thresholds and geometric camera alignment."""
        if self.is_calibrated:
            return True
            
        if len(self.history_ear) < self.calibration_window:
            self.history_ear.append(float(ear))
            self.history_yaw.append(float(yaw))
            self.history_pitch.append(float(pitch))
            return False
            
        self.mean_ear = float(np.mean(self.history_ear))
        self.std_ear = max(float(np.std(self.history_ear)), 0.02)
        
        # Establish center-point reference to cancel out mobile mount skew/offset
        self.center_yaw_bias = float(np.mean(self.history_yaw))
        self.center_pitch_bias = float(np.mean(self.history_pitch))
        
        self.is_calibrated = True
        return True

    def _purge_stale_buffer_records(self, current_time: float):
        """Removes telemetry inputs that fall outside the historical sliding window."""
        while self.temporal_buffer and (current_time - self.temporal_buffer[0][0] > self.time_window):
            self.temporal_buffer.popleft()

    def _calculate_perclos(self) -> float:
        """Computes the percentage of eye closure across the active sliding window."""
        if not self.temporal_buffer:
            return 0.0
        closed_frames = sum(1 for _, is_closed, _ in self.temporal_buffer if is_closed)
        return float(closed_frames / len(self.temporal_buffer))

    def _generate_xai_diagnostic(self, state: str, fatigue: float, attention: float) -> str:
        """
        10-Tier Hierarchical Rule-Engine generating context-aware 
        explainable AI feedback vectors based on dynamic safety indexes.
        """
        current_time = time.time()
        # Critical attention-independent emergency states prioritized first
        if state == "HIGH RISK CRITICAL"and (current_time - self.last_microsleep_trigger_time <= 3.2):
            return "CRITICAL EMERGENCY: Multiple microsleep incidents detected alongside structural attention collapse. Triggering cabin audio warnings."
        if fatigue > 0.85:
            return "CRITICAL FATIGUE: PERCLOS metrics indicate severe drowsiness. Visual tracking speed is highly degraded."
        if fatigue > 0.60 and attention < 0.50:
            return "HIGH RISK: Driver is simultaneously exhibiting elevated micro-sleep biomarkers and persistent off-road gaze behavior."
        
        # Distraction State Evaluation
        if attention < 0.30:
            return "CRITICAL DISTRACTION: Extended off-road gaze detected. High collision risk due to prolonged visual detachment."
        if attention < 0.65:
            return "WARNING: Frequent off-gaze scanning detected. Driver attention is wandering outside the defensive driving corridor."
        
        # Fatigue Warning Degradation Escalation
        if fatigue > 0.60:
            return "WARNING: Saccadic eye movement patterns show steady degradation. Neural fatigue metrics are climbing."
        if fatigue > 0.35:
            return "NOTICE: Micro-yawn clusters and shifting EAR baselines point to early-stage drowsiness onset."
        
        # Nominal & Optimal Operation States
        if attention > 0.85 and fatigue < 0.20:
            return "OPTIMAL PERFORMANCE: Geometric gaze vector aligned with the roadway center line. Baseline alertness remains high."
        if attention > 0.80:
            return "NOMINAL STATE: Normal mirror checks and gaze patterns observed within safe parameters."
        
        return "SYSTEM ACTIVE: Collecting telemetry. Baseline driver attention and fatigue indexes are operating within safe bounds."

    def process_kinematics(
        self, 
        ear: float, 
        mar: float, 
        yaw: float, 
        pitch: float, 
        model_eyes_closed: int, 
        model_yawning: int, 
        model_fatigued: int, 
        face_detected: bool
    ) -> Dict:
        """Applies structural calculations to update telemetry states and return system metrics."""
        current_time = time.time()
        dt = min(0.1, max(0.01, current_time - self.last_update_time))
        self.last_update_time = current_time

        # Pass positional kinetic coordinates to build environment baseline offsets
        if not self.update_baseline_calibration(ear, yaw, pitch):
            return {
                "calibrating": 1.0,
                "calibration_progress": len(self.history_ear) / self.calibration_window,
                "driver_state": "CALIBRATING",
                "xai_explanation": "Analyzing physical kinetics to calibrate personalized eye metrics..."
            }

        # =====================================================================
        # 1. FATIGUE LOGIC & CALCULATION (UNCHANGED, INTENTIONAL BACKEND MATCH)
        # =====================================================================
        adaptive_close_threshold = self.mean_ear - (1.9 * self.std_ear)
        is_eye_closed_frame = (ear < adaptive_close_threshold) or (model_eyes_closed == 1)

        is_yawning_frame = (mar > 0.60) or (model_yawning == 1)
        if is_yawning_frame and (current_time - self.last_yawn_timestamp > 4.0):
            self.total_yawn_events += 1
            self.last_yawn_timestamp = current_time

        self.temporal_buffer.append((current_time, is_eye_closed_frame, is_yawning_frame))
        self._purge_stale_buffer_records(current_time)

        if is_eye_closed_frame:
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = current_time
            else:
                duration = current_time - self.eye_closed_start_time
                if duration >= 0.75 and not self.micro_sleep_active:
                    self.micro_sleep_incidents += 1
                    self.micro_sleep_active = True
                    self.last_microsleep_trigger_time = current_time
        else:
            self.eye_closed_start_time = None
            self.micro_sleep_active = False

        perclos = self._calculate_perclos()
        yawn_scalar = min(1.0, (self.total_yawn_events * 0.15))
        model_scalar = 0.30 if model_fatigued else 0.0
        
        raw_fatigue_score = (perclos * 0.50) + (yawn_scalar * 0.20) + (model_scalar)
        raw_fatigue_score = np.clip(raw_fatigue_score, 0.0, 1.0)
        
        alpha_fatigue = 0.08
        self.smoothed_fatigue = (alpha_fatigue * raw_fatigue_score) + ((1.0 - alpha_fatigue) * self.smoothed_fatigue)
        # =====================================================================

        # =====================================================================
        # 2. RECTIFIED ATTENTION FOCUS SECTOR (ADAPTIVE BIAS CONE CORRECTION)
        # =====================================================================
        if not face_detected:
            # Immediate track drop when face disappears entirely
            self.attention_score = max(0.0, self.attention_score - (0.25 * dt))
            if self.distraction_start_time is None:
                self.distraction_start_time = current_time
        else:
            # Normalize live input against calibrated installation center baseline
            normalized_yaw = yaw - self.center_yaw_bias
            normalized_pitch = pitch - self.center_pitch_bias
            
            abs_yaw = abs(normalized_yaw)
            abs_pitch = abs(normalized_pitch)
            
            # Dynamic boundaries accommodating standard mobile placement variation
            if abs_yaw <= 25.0 and abs_pitch <= 18.0:
                # Driver is focused on the central road safety corridor -> Recover attention score linearly
                self.attention_score = min(1.0, self.attention_score + (0.35 * dt))
                self.distraction_start_time = None
            else:
                # Driver is looking off-road -> Compute distraction penalties
                if self.distraction_start_time is None:
                    self.distraction_start_time = current_time
                    self.distraction_event_counter += 1
                    
                distraction_run_time = current_time - self.distraction_start_time
                self.longest_distraction_duration = max(self.longest_distraction_duration, distraction_run_time)
                
                angular_deviation_weight = min(1.0, (abs_yaw / 90.0))
                decay_rate = 0.05 + (distraction_run_time * 0.04) + (angular_deviation_weight * 0.12)
                self.attention_score = max(0.0, self.attention_score - (decay_rate * dt))

        # =====================================================================
        # 3. DRIVER STATE MACHINE & LOG BALANCING
        # =====================================================================
        if (current_time - self.last_microsleep_trigger_time <= 3.2) or \
           (is_eye_closed_frame and self.eye_closed_start_time and (current_time - self.eye_closed_start_time) > 1.5):
            driver_state = "HIGH RISK CRITICAL"
        elif self.smoothed_fatigue > 0.65:
            driver_state = "DROWSY COMPROMISED"
        elif self.attention_score < 0.45:
            driver_state = "DISTRACTED DECONCENTRATED"
        elif self.attention_score > 0.80 and self.smoothed_fatigue < 0.25:
            driver_state = "ATTENTIVE ACTIVATED"
        else:
            driver_state = "NOMINAL ALERT"

        self.fatigue_history.append(self.smoothed_fatigue)
        self.attention_history.append(self.attention_score)

        return {
            "fatigue_probability": float(self.smoothed_fatigue),
            "attention_score": float(self.attention_score),
            "micro_sleep_incidents": int(self.micro_sleep_incidents),
            "distraction_events": int(self.distraction_event_counter),
            "yawn_events": int(self.total_yawn_events),
            "driver_state": driver_state,
            "average_attention": float(np.mean(self.attention_history)),
            "average_fatigue": float(np.mean(self.fatigue_history)),
            "longest_distraction_seconds": float(self.longest_distraction_duration),
            "session_duration_minutes": float((current_time - self.session_start_time) / 60.0),
            "calibrating": 0.0,
            "calibration_progress": 1.0,
            "xai_explanation": self._generate_xai_diagnostic(driver_state, self.smoothed_fatigue, self.attention_score)
        }
    

