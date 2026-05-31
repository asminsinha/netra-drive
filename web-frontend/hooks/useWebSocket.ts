import { useEffect, useState, useRef } from 'react';

export interface TelemetryData {
  fatigue_probability: number;
  attention_score: number;
  calibrating: number;
  raw_ear: number;
  raw_mar: number;
  is_calibrated: boolean;
  calibration_progress: number;
}

export const useWebSocket = (url: string) => {
  const [data, setData] = useState<TelemetryData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const socket = new WebSocket(url);
    socketRef.current = socket;

    socket.onopen = () => {
      setIsConnected(true);
      console.log("NetraDrive Telemetry Socket Stream Connected.");
    };

    socket.onmessage = (event) => {
      try {
        const telemetry: TelemetryData = JSON.parse(event.data);
        setData(telemetry);
      } catch (err) {
        console.error("Error decoding inbound telemetry packet:", err);
      }
    };

    socket.onclose = () => {
      setIsConnected(false);
      console.log("NetraDrive Telemetry Socket Stream Disconnected.");
    };

    return () => {
      socket.close();
    };
  }, [url]);

  return { data, isConnected };
};