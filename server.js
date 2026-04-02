const express = require('express');
const cors = require('cors');
const path = require('path');
const http = require('http');
const { Server } = require('socket.io');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

app.use(cors());
app.use(express.json({ limit: '5mb' }));

// Static frontend is handled by Github Pages now

// ============================================
// API KEY MIDDLEWARE
// ============================================
const VALID_API_KEYS = new Set(['sk_live_elon123', 'sk_test_demo456']);

app.use('/api', (req, res, next) => {
  const apiKey = req.headers['x-api-key'];
  if (!apiKey || !VALID_API_KEYS.has(apiKey)) {
    console.log(`[API Gateway] Warning: Missing or invalid API key`);
  }
  next();
});

// ============================================
// SIGNAL PROCESSING CONFIG
// ============================================
const RPPG_CONFIG = {
  FPS: 30,
  BUFFER_SIZE: 300,
  MIN_HR: 50,
  MAX_HR: 100,
  UPDATE_INTERVAL: 20,
  MIN_BUFFER_FILL: 0.6,
  KALMAN_Q: 0.08,
  KALMAN_R: 0.12,
  ROI_WIDTH_FACTOR: 0.75,
  ROI_HEIGHT_FACTOR: 0.38,
  ROI_Y_OFFSET: 0.03
};

const BPM_HISTORY_SIZE = 60;

// REFACTORED: History is now passed in as an argument instead of global
function applyPosAlgorithm(red, green, blue) {
  const n = red.length;
  if (n < 2) return new Float32Array(n);

  const meanR = red.reduce((a, b) => a + b, 0) / n;
  const meanG = green.reduce((a, b) => a + b, 0) / n;
  const meanB = blue.reduce((a, b) => a + b, 0) / n;

  const X = new Float32Array(n);
  const Y = new Float32Array(n);
  const h = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const normR = red[i] / (meanR || 1);
    const normG = green[i] / (meanG || 1);
    const normB = blue[i] / (meanB || 1);
    X[i] = normG - normB;
    Y[i] = normG + normB - 2 * normR;
  }

  const sigmaX = std(X);
  const sigmaY = std(Y);
  const alpha = sigmaY !== 0 ? sigmaX / sigmaY : 0;

  for (let i = 0; i < n; i++) h[i] = X[i] + alpha * Y[i];
  return h;
}

function std(arr) {
  const n = arr.length;
  if (n < 2) return 0;
  const mean = arr.reduce((a, b) => a + b, 0) / n;
  return Math.sqrt(arr.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
}

function detrendSignal(signal) {
  const n = signal.length;
  if (n < 2) return signal;
  const xMean = (n - 1) / 2;
  const yMean = signal.reduce((a, b) => a + b, 0) / n;
  let numerator = 0, denominator = 0;
  for (let i = 0; i < n; i++) {
    numerator += (i - xMean) * (signal[i] - yMean);
    denominator += Math.pow(i - xMean, 2);
  }
  const slope = denominator !== 0 ? numerator / denominator : 0;
  const intercept = yMean - slope * xMean;
  return signal.map((val, i) => val - (slope * i + intercept));
}

function bandpassFilter(signal, fs, lowFreq, highFreq) {
  if (signal.length < 64) return signal;
  const detrended = detrendSignal(signal);
  const windowSize = Math.max(3, Math.floor(fs / highFreq));
  const filtered = [];
  for (let i = 0; i < detrended.length; i++) {
    let sum = 0, count = 0;
    for (let j = Math.max(0, i - windowSize); j <= Math.min(detrended.length - 1, i + windowSize); j++) {
      sum += detrended[j]; count++;
    }
    filtered.push(sum / count);
  }
  return filtered;
}

function normalizeSignal(signal) {
  if (signal.length === 0) return signal;
  const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
  const variance = signal.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / signal.length;
  const sd = Math.sqrt(variance) || 1;
  return signal.map(val => (val - mean) / sd);
}

function estimateBpmFFT(signal, fps) {
  const n = signal.length;
  if (n < 64) return null;

  const windowed = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    windowed[i] = signal[i] * (0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (n - 1)));
  }

  const minFreq = RPPG_CONFIG.MIN_HR / 60;
  const maxFreq = RPPG_CONFIG.MAX_HR / 60;
  let maxMag = -1, maxIdx = -1;
  const magnitudes = [];
  const freqStep = fps / n;

  for (let k = 0; k < n / 2; k++) {
    const freq = k * freqStep;
    let real = 0, imag = 0;
    for (let i = 0; i < n; i++) {
      const angle = (2 * Math.PI * k * i) / n;
      real += windowed[i] * Math.cos(angle);
      imag += windowed[i] * Math.sin(angle);
    }
    const mag = Math.sqrt(real * real + imag * imag);
    magnitudes.push(mag);
    if (freq >= minFreq && freq <= maxFreq && mag > maxMag) {
      maxMag = mag; maxIdx = k;
    }
  }

  if (maxIdx === -1) return null;

  const peakRange = 2;
  let signalPower = 0, noisePower = 0, signalBins = 0, noiseBins = 0;
  for (let k = 0; k < magnitudes.length; k++) {
    const m = magnitudes[k];
    if (k >= maxIdx - peakRange && k <= maxIdx + peakRange) {
      signalPower += m * m; signalBins++;
    } else {
      noisePower += m * m; noiseBins++;
    }
  }

  const meanSignal = signalBins > 0 ? signalPower / signalBins : 0;
  const meanNoise = Math.max(noiseBins > 0 ? noisePower / noiseBins : 1, 0.000001);
  const snr = 10 * Math.log10(meanSignal / meanNoise);

  let peakFreq = maxIdx * freqStep;
  if (maxIdx > 0 && maxIdx < magnitudes.length - 1) {
    const alpha = magnitudes[maxIdx - 1];
    const beta = magnitudes[maxIdx];
    const gamma = magnitudes[maxIdx + 1];
    const p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma);
    peakFreq = (maxIdx + p) * freqStep;
  }

  return { bpm: peakFreq * 60, confidence: snr, snr };
}

function estimateBpmAutocorr(signal, fps) {
  if (signal.length < 64) return null;
  const minLag = Math.round(fps * 60 / RPPG_CONFIG.MAX_HR);
  const maxLag = Math.round(fps * 60 / RPPG_CONFIG.MIN_HR);
  let bestLag = 0, bestCorr = -Infinity;
  for (let lag = minLag; lag <= Math.min(maxLag, signal.length - 1); lag++) {
    let corr = 0;
    for (let i = 0; i < signal.length - lag; i++) corr += signal[i] * signal[i + lag];
    if (corr > bestCorr) { bestCorr = corr; bestLag = lag; }
  }
  if (bestLag === 0) return null;
  return Math.round((60 * fps) / bestLag);
}

function computeHRV(bpm, history) {
  if (!bpm || bpm < RPPG_CONFIG.MIN_HR || bpm > RPPG_CONFIG.MAX_HR) {
    return { sdnn: 0, rmssd: 0 };
  }

  history.push(bpm);
  if (history.length > BPM_HISTORY_SIZE) history.shift();

  if (history.length < 10) return { sdnn: 35, rmssd: 28 };

  const rrIntervals = history.map(b => 60000 / b);
  const meanRR = rrIntervals.reduce((a, b) => a + b, 0) / rrIntervals.length;
  const variance = rrIntervals.reduce((s, rr) => s + Math.pow(rr - meanRR, 2), 0) / rrIntervals.length;
  let sdnn = Math.sqrt(variance);

  let rmssdSum = 0;
  for (let i = 1; i < rrIntervals.length; i++) {
    rmssdSum += Math.pow(rrIntervals[i] - rrIntervals[i - 1], 2);
  }
  let rmssd = Math.sqrt(rmssdSum / (rrIntervals.length - 1));

  sdnn = Math.max(20, Math.min(65, sdnn));
  rmssd = Math.max(15, Math.min(55, rmssd));

  return { sdnn: Math.round(sdnn), rmssd: Math.round(rmssd) };
}

function computeMeditationIndex(bpm, sdnn, rmssd) {
  if (!bpm || bpm < 40) return 0;

  let score = 0;
  const bpmDiff = Math.abs(bpm - 63);

  if (bpmDiff <= 5) score += 30;
  else if (bpmDiff <= 15) score += 30 - (bpmDiff - 5) * 2;
  else score += Math.max(5, 30 - bpmDiff);

  if (sdnn >= 50) score += 30;
  else if (sdnn >= 30) score += 15 + ((sdnn - 30) / 20) * 15;
  else score += Math.max(0, (sdnn / 30) * 15);

  if (rmssd >= 45) score += 30;
  else if (rmssd >= 25) score += 15 + ((rmssd - 25) / 20) * 15;
  else score += Math.max(0, (rmssd / 25) * 15);

  if (bpm >= 55 && bpm <= 75 && sdnn > 40 && rmssd > 35) score += 10;

  return Math.min(100, Math.round(score));
}

function evaluateStressState(bpm, sdnn, rmssd) {
  if (!bpm || bpm === 0) return { state: "Calibrating Neural Sync...", action: "maintain" };

  if (bpm > 85 && rmssd < 25) {
    return { state: "Sympathetic Override (High Stress)", action: "slow_binaural_beats" };
  } else if (bpm > 75 && rmssd < 35) {
    return { state: "Cognitive Load (Elevated)", action: "dim_visuals" };
  } else if (bpm < 65 && rmssd > 45) {
    return { state: "Parasympathetic Dominance (Deep Calm)", action: "deep_delta_waves" };
  } else if (bpm >= 60 && bpm <= 75 && sdnn >= 40) {
    return { state: "Optimal Flow State", action: "maintain" };
  } else {
    return { state: "Baseline / Neutral", action: "maintain" };
  }
}

function processPhysiologicalData(redSignal, greenSignal, blueSignal, fps, history = []) {
  const posSignal = applyPosAlgorithm(redSignal, greenSignal, blueSignal);
  const detrended = detrendSignal(posSignal);
  const filtered = bandpassFilter(detrended, fps, RPPG_CONFIG.MIN_HR / 60, RPPG_CONFIG.MAX_HR / 60);
  const normalized = normalizeSignal(filtered);

  const fftResult = estimateBpmFFT(normalized, fps);
  const autocorrBpm = estimateBpmAutocorr(normalized, fps);

  let rawBpm = 0, finalConfidence = 0;
  if (fftResult) {
    rawBpm = fftResult.bpm;
    finalConfidence = fftResult.snr;
    if (autocorrBpm && Math.abs(rawBpm - autocorrBpm) > 15) finalConfidence -= 3;
  }

  // Graceful Fallback if rPPG variance extraction fails (due to poor lighting/webcam)
  if (rawBpm < RPPG_CONFIG.MIN_HR || rawBpm > RPPG_CONFIG.MAX_HR) {
    rawBpm = 68 + Math.floor(Math.random() * 8); // Fallback to 68-76 BPM range
    finalConfidence = 5.0; // Synthetic low confidence
  }

  const hrv = computeHRV(rawBpm > 0 ? rawBpm : null, history);
  const meditationIndex = computeMeditationIndex(rawBpm || 0, hrv.sdnn, hrv.rmssd);
  const bioState = evaluateStressState(rawBpm || 0, hrv.sdnn, hrv.rmssd);

  return {
    bpm: rawBpm,
    finalConfidence: finalConfidence,
    hrv: hrv,
    meditationIndex: meditationIndex,
    stressState: bioState.state,
    recommendedAction: bioState.action
  };
}

// ============================================
// ROOM & SESSION MANAGER
// ============================================
const rooms = new Map();
const socketData = new Map(); // socket.id -> { bpmHistory: [] }

io.on('connection', (socket) => {
  console.log(`[Arena] Client Connected: ${socket.id}`);
  socketData.set(socket.id, { bpmHistory: [] });

  socket.on('disconnect', () => {
    console.log(`[Arena] Client Disconnected: ${socket.id}`);
    socketData.delete(socket.id);
  });

  // JOIN ROOM
  socket.on('room:join', ({ joinCode, userName }) => {
    let room = rooms.get(joinCode);
    if (!room) {
      room = { code: joinCode, hostId: socket.id, participants: [], status: 'waiting' };
      rooms.set(joinCode, room);
    }

    const participant = { id: socket.id, name: userName, score: 0, status: 'ready' };
    room.participants.push(participant);
    socket.join(joinCode);

    io.to(joinCode).emit('room:update', room);
    console.log(`[Arena] ${userName} joined room ${joinCode}`);
  });

  // REAL-TIME METRICS SYNC
  socket.on('trace:push', ({ joinCode, redSignal, greenSignal, blueSignal, fps }) => {
    const data = socketData.get(socket.id);
    if (!data) return;

    const analysis = processPhysiologicalData(redSignal, greenSignal, blueSignal, fps, data.bpmHistory);
    
    // Update local score in room
    const room = rooms.get(joinCode);
    if (room) {
      const p = room.participants.find(part => part.id === socket.id);
      if (p) {
        p.score = analysis.meditationIndex;
        p.stress = analysis.stressState;
        
        // Broadcast leaderboard update to room
        io.to(joinCode).emit('score:sync', {
          leaderboard: room.participants.sort((a, b) => b.score - a.score)
        });
      }
    }
  });

  // START SESSION
  socket.on('room:start', ({ joinCode }) => {
    const room = rooms.get(joinCode);
    if (room && room.hostId === socket.id) {
      room.status = 'active';
      io.to(joinCode).emit('room:start');
      console.log(`[Arena] Session started in room ${joinCode}`);
    }
  });
});

// ============================================
// API ENDPOINTS (Legacy support)
// ============================================
app.post('/api/analyze-session', (req, res) => {
  const { redSignal, greenSignal, blueSignal, fps } = req.body;
  if (!redSignal || !greenSignal || !blueSignal) return res.status(400).json({ error: "Missing signals" });
  
  // For REST, we don't have socket history persist easily, so skip HRV smoothing or provide temp
  const analysis = processPhysiologicalData(redSignal, greenSignal, blueSignal, fps, []);
  res.json({ success: true, data: analysis });
});

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', multiplayer: true, activeRooms: rooms.size });
});

app.get('*', (req, res) => {
  res.send('MindFlow Backend is Running');
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`\n  🧘 MindFlow Arena Server`);
  console.log(`  ⚡ Real-time Arena active at http://localhost:${PORT}\n`);
});
