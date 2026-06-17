'use client';

import React from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import { 
  Eye, ShieldAlert, Activity, Wifi, WifiOff, 
  Video, EyeOff, FileText, Brain, Gauge
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts';
import { jsPDF } from 'jspdf';

interface TelemetryData {
  fatigue_probability: number;
  attention_score: number;
  micro_sleep_incidents: number;
  distraction_events: number;
  yawn_events: number;
  driver_state: string;
  average_attention: number;
  average_fatigue: number;
  longest_distraction_seconds: number;
  session_duration_minutes: number;
  calibrating: number;
  calibration_progress: number;
  xai_explanation: string;
}

//-------------------------------------------------------------------
/*export default function DashboardPage() {
  const WS_ENDPOINT = process.env.NEXT_PUBLIC_WS_URL || 'wss://asminsinha2005-netra-drive-backend.hf.space/ws';
  const { data, isConnected } = useWebSocket(WS_ENDPOINT) as { 
    data: TelemetryData | null; 
    isConnected: boolean; 
  };*/

  export default function DashboardPage() {
  // RECTIFICATION: Uses local development socket string as fallback if env variables aren't active
  const WS_ENDPOINT = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws';
  
  const { data, isConnected } = useWebSocket(WS_ENDPOINT) as { 
    data: TelemetryData | null; 
    isConnected: boolean; 
  };

//---------------------------------------------------------------

  const [history, setHistory] = React.useState<any[]>([]);
  const [telemetryLedger, setTelemetryLedger] = React.useState<any[]>([]);
  const lastProcessedTime = React.useRef<number>(0);
  const videoRef = React.useRef<HTMLVideoElement | null>(null);
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const [usingWebcam, setUsingWebcam] = React.useState<boolean>(false);

  // Compute uniform, safe layout metrics to shield against backend scale variance
  const fatigueValue = data?.fatigue_probability ?? 0.0;
  const attentionValue = data?.attention_score ?? 1.0;

  const displayFatiguePercent = typeof fatigueValue === 'number' 
    ? (fatigueValue <= 1.0 ? fatigueValue * 100 : fatigueValue) 
    : 0.0;

  const displayAttentionPercent = typeof attentionValue === 'number' 
    ? (attentionValue <= 1.0 ? attentionValue * 100 : attentionValue) 
    : 100.0;

  React.useEffect(() => {
    if (!data) return;

    const now = Date.now();
    // 3-second update throttle window to maintain high render efficiency
    if (now - lastProcessedTime.current < 3000) return;
    lastProcessedTime.current = now;

    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    // Window scaling normalization
    const currentFatigue = data.fatigue_probability <= 1.0 ? data.fatigue_probability * 100 : data.fatigue_probability;
    const currentAttention = data.attention_score <= 1.0 ? data.attention_score * 100 : data.attention_score;
    const microSleeps = data.micro_sleep_incidents ?? 0;

    // 1. Update Tracking Charts
    setHistory((prev) => {
      const updated = [...prev, { 
        time: timestamp, 
        fatigue: currentFatigue,
        attention: currentAttention
      }];
      return updated.slice(-20);
    });

    // 2. Append directly to Verification Ledger
    setTelemetryLedger((prev) => {
      const newEntry = {
        time: timestamp,
        sleeps: microSleeps,
        fatigue: currentFatigue,
        attention: currentAttention,
        xai: data.xai_explanation ? data.xai_explanation.split('|')[0].trim() : "Optimal neural state recorded."
      };
      return [...prev, newEntry].slice(-50);
    });
  }, [data]);

    //-----------------------
  React.useEffect(() => {
    let stream: MediaStream | null = null;
    let frameInterval: NodeJS.Timeout | null = null;
    let isProcessingFrame = false; // Guard to stop simultaneous overlapping uploads

    async function initBrowserWebcam() {
      if (!isConnected) return;
      
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: 480, height: 360 } // Downsampled resolution to minimize data size
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setUsingWebcam(true);
        }

        const apiEndpoint = WS_ENDPOINT.replace('wss://', 'https://').replace('ws://', 'http://').replace('/ws', '/api/v1/process_webcam_frame');

        // Lower frequency to 250ms (~4 FPS) giving the Hugging Face space time to finish computation loops
        frameInterval = setInterval(() => {
          if (!canvasRef.current || !videoRef.current || videoRef.current.readyState !== 4) return;
          if (isProcessingFrame) return; // Skip frame drop if the server is currently crunching data
          
          isProcessingFrame = true;
          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d');
          if (!ctx) { isProcessingFrame = false; return; }

          canvas.width = videoRef.current.videoWidth;
          canvas.height = videoRef.current.videoHeight;
          ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

          canvas.toBlob(async (blob) => {
            if (!blob) { isProcessingFrame = false; return; }
            const formData = new FormData();
            formData.append('file', blob, 'frame.jpg');

            try {
              const response = await fetch(apiEndpoint, { 
                method: 'POST', 
                body: formData,
                headers: { 'Accept': 'application/json' }
              });
              if (!response.ok) {
                console.log("Server buffer full, skipping packet frame dropping sequence.");
              }
            } catch (err) {
              console.error("Webcam packet delivery failed:", err);
            } finally {
              isProcessingFrame = false; // Unlock frame loop execution safety guard
            }
          }, 'image/jpeg', 0.4); // Dropped quality compression matrix to 0.4 for instantaneous payload delivery
        }, 250);

      } catch (err) {
        console.log("Local device webcam not active or locked, utilizing default stream pipeline fallback.", err);
        setUsingWebcam(false);
      }
    }

    initBrowserWebcam();

    return () => {
      if (stream) stream.getTracks().forEach(track => track.stop());
      if (frameInterval) clearInterval(frameInterval);
    };
  }, [isConnected, WS_ENDPOINT]);


    //-------------------------------------
  const calculateMean = (arr: number[]) => arr.length ? (arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(1) : "0.0";
  const calculateMedian = (arr: number[]) => {
    if (!arr.length) return "0.0";
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return (sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2).toFixed(1);
  };
  const calculateMode = (arr: number[]) => {
    if (!arr.length) return "0.0";
    const mapping: Record<number, number> = {};
    let maxCount = 0;
    let modeVal = arr[0];
    arr.forEach((val) => {
      const rounded = Math.round(val);
      mapping[rounded] = (mapping[rounded] || 0) + 1;
      if (mapping[rounded] > maxCount) {
        maxCount = mapping[rounded];
        modeVal = rounded;
      }
    });
    return modeVal.toFixed(1);
  };

  const generateFleetReportPDF = () => {
    const doc = new jsPDF();
    const fatigueList = telemetryLedger.map(l => l.fatigue);
    const attentionList = telemetryLedger.map(l => l.attention);
    const totalMicroSleeps = telemetryLedger.length > 0 ? telemetryLedger[telemetryLedger.length - 1].sleeps : 0;

    // --- NEW: DYNAMIC OVERALL SESSION PERFORMANCE DIAGNOSTIC ANALYZER ---
    // --- REVISED: MULTI-VARIABLE TELEMETRY VALIDATION ENGINE ---
    const generateSessionTrendInsight = (): string => {
      if (!telemetryLedger.length) return "DATA UNAVAILABLE: Insufficient session telemetry records compiled to compute baseline risk analytics.";

      const meanFatigue = telemetryLedger.reduce((sum, item) => sum + item.fatigue, 0) / telemetryLedger.length;
      const meanAttention = telemetryLedger.reduce((sum, item) => sum + item.attention, 0) / telemetryLedger.length;

      // Extract current tail telemetry to evaluate immediate real-time directional trends
      const finalRecord = telemetryLedger[telemetryLedger.length - 1];
      const currentFatigue = finalRecord.fatigue;
      const currentAttention = finalRecord.attention;

      // 1. CRITICAL TRUE FATIGUE (High Micro-Sleeps AND High/Moderate Average Fatigue)
      if (totalMicroSleeps >= 5 && meanFatigue >= 30.0) {
        return `CRITICAL SYSTEM ALARM: Severe behavioral risk. The driver triggered ${totalMicroSleeps} microsleep incidents, matching an elevated mean fatigue index of ${meanFatigue.toFixed(1)}%. True compounding neurological fatigue is confirmed across this log block. Immediate dispatch safety intervention required.`;
      }

      // 2. HIGH TRANSIENT LAPSE OR SENSOR FLICKER (High Micro-Sleeps BUT Low Average Fatigue & High Attention)
      if (totalMicroSleeps >= 5 && meanFatigue < 30.0 && meanAttention >= 85.0) {
        return `NOMINAL STATE WITH TRANSIENT ANOMALIES: Log tracking indicates ${totalMicroSleeps} isolated micro-sleep flags; however, the session baseline shows exceptional focus (${meanAttention.toFixed(1)}%) and low overall fatigue (${meanFatigue.toFixed(1)}%). Incidents are categorized as momentary blinks or potential camera tracking fluctuations rather than systemic exhaustion. Continuous monitoring active.`;
      }

      // 3. COMPOUNDING MIXED RISK (Moderate Fatigue AND Distraction with Micro-Sleeps present)
      if (meanFatigue >= 25.0 && meanAttention < 75.0) {
        return `COMPLIANCE WARNING (MIXED PROFILE DEGRADATION): Intersecting risk markers detected. Driver attention is drifting below baseline standards (Mean: ${meanAttention.toFixed(1)}%) while rolling fatigue metrics have settled at an elevated ${meanFatigue.toFixed(1)}% with ${totalMicroSleeps} micro-lapses recorded. Systemic drowsiness is beginning to impair active roadway focus.`;
      }

      // 4. PURE SUSTAINED INATTENTION / DISTRACTION (Low Attention, regardless of micro-sleeps)
      if (meanAttention < 70.0) {
        return `COMPLIANCE WARNING (DISTRACTION): Persistent off-road gaze vectors detected. Driver operated with an average attentiveness focus rating of ${meanAttention.toFixed(1)}% over the compiled timeline. Visual detachment violates fleet safety guidelines, despite nominal baseline fatigue metrics (${meanFatigue.toFixed(1)}%).`;
      }

      // 5. PURE DROWSINESS ONSET (High rolling fatigue index, low micro-sleep spikes)
      if (meanFatigue > 40.0) {
        return `MANAGEMENT NOTICE (DROWSINESS): Heavy systemic fatigue patterns observed. Mean session fatigue is tracking at ${meanFatigue.toFixed(1)}%. Saccadic behavior rhythms indicate steady stamina degradation. Scheduling a mandatory vehicle rest break is strongly advised.`;
      }

      // === NEW INTEL RULES ADDED BELOW (NON-REDUNDANT PREDICTIVE TRENDS) ===

      // 6. HYPER-VIGILANCE COGNITIVE TUNNELING (Fixated gaze masking progressive exhaustion)
      if (meanFatigue >= 22.0 && meanAttention >= 95.0) {
        return `OPERATIONAL NOTICE (COGNITIVE TUNNELING): Driver exhibits an unusually high, rigid attention vector (${meanAttention.toFixed(1)}%) alongside an ascending baseline fatigue index (${meanFatigue.toFixed(1)}%). This behavioral pattern typically correlates with exhausted 'staring' or cognitive fixation, where normal mirror-scanning frequencies drop as the operator fights fatigue.`;
      }

      // 7. REAL-TIME RETAIL STABILIZATION TREND (Session was messy, but driver is sharpening focus right now)
      if (meanFatigue > 20.0 && currentFatigue < meanFatigue && currentAttention > meanAttention) {
        return `ANALYTICAL UPDATE (ALERTNESS STABILIZATION): While historical session averages show baseline degradation tokens (Mean Fatigue: ${meanFatigue.toFixed(1)}%), current real-time telemetry indicates an active recovery trend. Immediate metrics have stabilized to ${currentFatigue.toFixed(1)}% fatigue and ${currentAttention.toFixed(0)}% attention. No manual break required yet; tracking trajectory closely.`;
      }

      // 8. PROGRESSIVE DEGRADATION SLIP (Session averages look fine, but real-time tail data is plummeting)
      if (meanFatigue <= 20.0 && currentFatigue > (meanFatigue + 10.0)) {
        return `ANALYTICAL ALERT (PROGRESSIVE ONSET): Overall session averages present a clean compliance profile, but immediate real-time telemetry shows a sharp downward trajectory. Fatigue has rapidly spiked to ${currentFatigue.toFixed(1)}% over the last few intervals. Early-stage exhaustion or distraction is actively materializing.`;
      }

      // 9. IDEAL COMPLIANCE RATING (Everything within green bounds)
      return `EXCELLENT COMPLIANCE RATING: Driver demonstrated exceptional performance over this 50-frame validation block. Maintained a steady defensive road focus rating of ${meanAttention.toFixed(1)}% with negligible baseline fatigue markers (${meanFatigue.toFixed(1)}%). No active risk tracking violations recorded.`;
    };

    const sessionXaiInsight = generateSessionTrendInsight();
    // --------------------------------------------------------------------

    // Document Header Frame Setup
    doc.setFillColor(15, 23, 42); 
    doc.rect(0, 0, 210, 38, 'F');
    
    doc.setTextColor(52, 211, 153);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(18);
    doc.text("NETRA-DRIVE DISPATCH COMPLIANCE LOG", 14, 22);
    
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(9);
    doc.setFont("helvetica", "normal");
    doc.text(`Generated Session Stamp: ${new Date().toLocaleString()} | Active Records Compiled: ${telemetryLedger.length}/50`, 14, 31);

    // Section 1 - Profile Statistics Text Data Rows
    doc.setTextColor(15, 23, 42);
    doc.setFontSize(12);
    doc.setFont("helvetica", "bold");
    doc.text("1. Driver Risk Profile Summary", 14, 52);
    
    doc.setFontSize(9.5);
    doc.setFont("helvetica", "normal");
    doc.text(`• Fatigue Probability Index -> Current: ${displayFatiguePercent.toFixed(1)}% | Mean: ${calculateMean(fatigueList)}% | Median: ${calculateMedian(fatigueList)}% | Mode: ${calculateMode(fatigueList)}%`, 16, 62);
    doc.text(`• Attentiveness Focus Rating -> Current: ${displayAttentionPercent.toFixed(0)}% | Mean: ${calculateMean(attentionList)}% | Median: ${calculateMedian(attentionList)}% | Mode: ${calculateMode(attentionList)}%`, 16, 69);
    doc.text(`• Total Accounted Micro-Sleep Flags: ${totalMicroSleeps} cumulative loop violations recorded`, 16, 76);

    // Section 2 - Model XAI Neural Diagnosis (Optimized with Summary Evaluation Block)
    doc.setFontSize(12);
    doc.setFont("helvetica", "bold");
    doc.text("2. Model XAI Neural Diagnosis", 14, 92);
    
    doc.setFillColor(248, 250, 252);
    doc.rect(14, 97, 182, 22, 'F');
    doc.setFontSize(9.5);
    doc.setFont("helvetica", "italic");
    doc.setTextColor(71, 85, 105);
    
    // Split and inject our newly generated multi-record session trend evaluation block
    const wrappedText = doc.splitTextToSize(sessionXaiInsight, 174);
    doc.text(wrappedText, 18, 103);

    // Section 3 - Ledger Compilation Window Setup
    doc.setTextColor(15, 23, 42);
    doc.setFontSize(12);
    doc.setFont("helvetica", "bold");
    doc.text("3. Structural Compliance Verification Ledger", 14, 134);

    let currentY = 144;
    doc.setFontSize(9.5);
    doc.setFont("helvetica", "bold");
    doc.text("Timestamp", 14, currentY);
    doc.text("Micro-Sleep Counter", 50, currentY);
    doc.text("Core Telemetry Metrics", 95, currentY);
    doc.text("Model Diagnostic Note", 142, currentY);
    
    doc.setDrawColor(226, 232, 240);
    doc.line(14, currentY + 2, 196, currentY + 2);
    currentY += 8;

    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);

    if (telemetryLedger.length === 0) {
      doc.text("No metrics buffered yet. Waiting for WebSocket injection streaming...", 14, currentY);
    } else {
      telemetryLedger.forEach((log, index) => {
        if (currentY > 275) {
          doc.setFontSize(8);
          doc.setTextColor(148, 163, 184);
          doc.text("NETRA-DRIVE DISPATCH COMPLIANCE LOG (CONTINUED)", 14, 288);
          doc.addPage();
          doc.setTextColor(15, 23, 42);
          doc.setFont("helvetica", "bold");
          doc.setFontSize(9.5);
          
          currentY = 20;
          doc.text("Timestamp", 14, currentY);
          doc.text("Micro-Sleep Counter", 50, currentY);
          doc.text("Core Telemetry Metrics", 95, currentY);
          doc.text("Model Diagnostic Note", 142, currentY);
          doc.line(14, currentY + 2, 196, currentY + 2);
          
          doc.setFont("helvetica", "normal");
          doc.setFontSize(9);
          currentY += 8;
        }

        if (index % 2 === 0) {
          doc.setFillColor(248, 250, 252);
          doc.rect(14, currentY - 4, 182, 6, 'F');
        }

        doc.text(log.time, 14, currentY);
        doc.text(`${log.sleeps} events`, 50, currentY);
        doc.text(`Fatigue: ${log.fatigue.toFixed(1)}% | Attn: ${log.attention.toFixed(0)}%`, 95, currentY);
        doc.text(log.xai.substring(0, 32), 142, currentY);
        currentY += 6;
      });
    }

    doc.setFontSize(8);
    doc.setTextColor(148, 163, 184);
    doc.text("CONFIDENTIAL DRIVER TELEMATICS ANALYTICS • SECURED NETRA-DRIVE CORE SYSTEM DATA PLATFORM", 14, 288);
    doc.save(`NetraDrive_SessionReport_${Date.now()}.pdf`);
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-50 p-5 font-sans relative pb-24">
      <header className="flex justify-between items-center border-b border-slate-800 pb-3 mb-5">
        <div>
          <h1 className="text-xl font-black tracking-wider text-emerald-400">NETRA-DRIVE HUB</h1>
          <p className="text-[11px] text-slate-400">Cognitive Vigilance Neural Pipeline Telematics Terminal</p>
        </div>
        <div className="flex items-center gap-3">
          <button 
            onClick={generateFleetReportPDF}
            className="flex items-center gap-1.5 bg-emerald-600 hover:bg-emerald-700 active:scale-95 transition-all text-white px-3 py-1.5 rounded-lg text-xs font-bold shadow-md"
          >
            <FileText className="w-3.5 h-3.5" /> DOWNLOAD LOG (.PDF)
          </button>
          <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px] font-bold tracking-wide ${isConnected ? 'bg-emerald-950/40 text-emerald-400 border border-emerald-500/20' : 'bg-rose-950/40 text-rose-400 border border-rose-500/20'}`}>
            {isConnected ? <Wifi className="w-3.5 h-3.5 animate-pulse" /> : <WifiOff className="w-3.5 h-3.5" />}
            {isConnected ? 'ONLINE STREAM INJECTOR' : 'DISCONNECTED'}
          </div>
        </div>
      </header>

      {/* Dynamic Driver Kinetics Calibration Loading Module Overlay */}
      {data?.calibrating === 1.0 && (
        <div className="w-full bg-slate-900/80 border border-emerald-500/30 rounded-xl p-5 backdrop-blur-md mb-5 animate-pulse">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-bold text-emerald-400 tracking-wider uppercase flex items-center gap-2">
              <Gauge className="w-4 h-4 animate-spin text-emerald-400" /> Initializing Driver Kinetics Baseline...
            </span>
            <span className="text-xs font-black text-emerald-400 font-mono">
              {Math.round((data?.calibration_progress ?? 0) * 100)}%
            </span>
          </div>
          <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
            <div 
              className="bg-gradient-to-r from-emerald-500 to-teal-400 h-full transition-all duration-300 ease-out"
              style={{ width: `${(data?.calibration_progress ?? 0) * 100}%` }}
            />
          </div>
          <p className="text-[11px] text-slate-400 mt-2 italic tracking-wide">
            Please watch the roadway vectors ahead naturally. Calibrating geometric mesh nodes and historical dynamic EAR baseline distributions...
          </p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Metric 1 - Fatigue Indicator */}
        <div className="bg-slate-900 border border-slate-800/80 rounded-xl p-4 shadow-md">
          <div className="flex justify-between items-start mb-2">
            <div>
              <p className="text-[10px] text-slate-400 uppercase tracking-widest font-extrabold">Fatigue Probability Index</p>
              <h3 className="text-3xl font-black mt-0.5 text-slate-100">
                {`${displayFatiguePercent.toFixed(1)}%`}
              </h3>
            </div>
            <ShieldAlert className={`w-6 h-6 ${displayFatiguePercent > 50 ? 'text-rose-500 animate-bounce' : 'text-emerald-500'}`} />
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
            <div 
              className={`h-full transition-all duration-300 ${displayFatiguePercent > 50 ? 'bg-rose-500' : 'bg-emerald-500'}`}
              style={{ width: `${displayFatiguePercent}%` }}
            />
          </div>
        </div>

        {/* Metric 2 - Attention Indicator */}
        <div className="bg-slate-900 border border-slate-800/80 rounded-xl p-4 shadow-md">
          <div className="flex justify-between items-start mb-2">
            <div>
              <p className="text-[10px] text-slate-400 uppercase tracking-widest font-extrabold">Driver Attention Rating</p>
              <h3 className="text-3xl font-black mt-0.5 text-slate-100">
                {`${displayAttentionPercent.toFixed(0)}%`}
              </h3>
            </div>
            <Eye className={`w-6 h-6 ${displayAttentionPercent < 55 ? 'text-rose-500 animate-pulse' : 'text-emerald-400'}`} />
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
            <div 
              className={`h-full transition-all duration-300 ${displayAttentionPercent < 55 ? 'bg-rose-500' : 'bg-emerald-500'}`}
              style={{ width: `${displayAttentionPercent}%` }}
            />
          </div>
        </div>

        {/* Micro Sleep Incident Box */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-md flex items-center gap-4 lg:col-span-2">
          <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800/30">
            <EyeOff className="w-6 h-6 text-rose-400" />
          </div>
          <div>
            <p className="text-[10px] uppercase tracking-wider text-slate-400 font-extrabold">Micro-Sleep Incidents</p>
            <h4 className="text-2xl font-black text-slate-100 mt-0.5">{data?.micro_sleep_incidents ?? 0} <span className="text-xs text-slate-500 font-medium">events detected</span></h4>
          </div>
        </div>

        {/* Live XAI Output Feed */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-md lg:col-span-2">
          <h4 className="text-[10px] font-bold uppercase tracking-widest text-slate-400 mb-2 flex items-center gap-1.5">
            <Brain className="w-3.5 h-3.5 text-purple-400" /> Live Neural Diagnostic Explanation Output Feed
          </h4>
          <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800/60 font-mono text-xs text-emerald-400 min-h-8 flex items-center tracking-wide">
            {data?.xai_explanation || "Parsing real-time facial node telemetrics..."}
          </div>
        </div>

        {/* Linear Chart Tracking View */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-md lg:col-span-2">
          <h4 className="text-[10px] font-bold uppercase tracking-widest text-slate-400 mb-2 flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5 text-emerald-400" /> Cognitive Trend Array Tracker
          </h4>
          <div className="h-40 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history}>
                <XAxis dataKey="time" stroke="#475569" fontSize={8} tickLine={false} />
                <YAxis domain={[0, 100]} stroke="#475569" fontSize={8} tickLine={false} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc', fontSize: '11px' }} />
                <Line type="monotone" dataKey="fatigue" stroke="#f43f5e" strokeWidth={2} dot={false} name="Fatigue %" />
                <Line type="monotone" dataKey="attention" stroke="#10b981" strokeWidth={2} dot={false} name="Attention Score %" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Video Streaming Window Frame */}
      <div className="fixed bottom-5 right-5 w-60 bg-slate-900/90 border border-slate-800 p-2 rounded-xl shadow-2xl z-50">
        <div className="flex items-center justify-between w-full mb-1.5">
          <span className="text-[9px] font-bold text-slate-400 tracking-widest uppercase flex items-center gap-1">
            <Video className="w-3 h-3 text-emerald-400" /> {usingWebcam ? "LOCAL DEVICE WEBCAM" : "LIVE AI ENGINE STREAM"}
          </span>
          <span className="flex h-1.5 w-1.5 relative">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500"></span>
          </span>
        </div>
        <div className="relative overflow-hidden rounded-lg border border-slate-800 bg-black w-full aspect-video">
          {/* Hidden components for capturing frame matrices from the user side */}
          <canvas ref={canvasRef} className="hidden" />
          <video ref={videoRef} autoPlay playsInline muted className="hidden" />

          {/* Conditional toggle: displays browser webcam if running live, or backend MJPEG if testing locally */}
          {usingWebcam ? (
            <video
              autoPlay
              playsInline
              muted
              ref={(el) => { if (el) el.srcObject = videoRef.current?.srcObject || null; }}
              className="w-full h-full object-cover transform scale-x-[-1]"
            />
          ) : (
            // eslint-disable-next-line @next/next/no-img-element
            <img 
              src={WS_ENDPOINT.includes('localhost') ? "http://localhost:8000/api/v1/video_feed" : "https://asminsinha2005-netra-drive-backend.hf.space/api/v1/video_feed"}
              alt="NetraDrive Feed"
              className="w-full h-full object-cover transform scale-x-[-1]"
              onError={(e) => {
                e.currentTarget.src = "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=600";
              }}
            />
          )}
        </div>
      </div>
    </main>
  );
}
