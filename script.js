const video = document.querySelector("#camera");
const overlay = document.querySelector("#overlay");
const frame = document.querySelector("#frame");
const statusEl = document.querySelector("#status");
const shapeIcon = document.querySelector("#shapeIcon");
const shapeName = document.querySelector("#shapeName");
const predictionCard = document.querySelector("#predictionCard");
const startButton = document.querySelector("#startButton");
const soundButton = document.querySelector("#soundButton");
const clearButton = document.querySelector("#clearButton");
const sequenceEl = document.querySelector("#sequence");
const sequenceCount = document.querySelector("#sequenceCount");
const circularityMeter = document.querySelector("#circularityMeter");
const aspectMeter = document.querySelector("#aspectMeter");
const extentMeter = document.querySelector("#extentMeter");

const ctx = frame.getContext("2d", { willReadFrequently: true });
const overlayCtx = overlay.getContext("2d");

const SHAPES = {
  circle: { name: "Círculo", icon: "○", note: "C4", frequency: 261.63, color: "#56e39f" },
  square: { name: "Quadrado", icon: "□", note: "E4", frequency: 329.63, color: "#ffca3a" },
  triangle: { name: "Triângulo", icon: "△", note: "G4", frequency: 392.0, color: "#8ecae6" },
};

const SHAPE_PROFILES = {
  circle: { center: [0.86, 1.0, 0.76], tolerance: [0.24, 0.36, 0.22] },
  square: { center: [0.66, 1.0, 0.91], tolerance: [0.22, 0.34, 0.16] },
  triangle: { center: [0.5, 1.0, 0.56], tolerance: [0.2, 0.34, 0.2] },
};

let tree;
let audioCtx;
let soundEnabled = true;
let lastPlayedShape = "";
let lastPlayedAt = 0;
let sequence = [];
let running = false;

class DecisionTree {
  constructor(maxDepth = 5, minSize = 8) {
    this.maxDepth = maxDepth;
    this.minSize = minSize;
    this.root = null;
  }

  fit(rows) {
    this.root = this.split(rows, 1);
  }

  predict(features) {
    let node = this.root;
    while (node.label === undefined) {
      node = features[node.feature] < node.value ? node.left : node.right;
    }
    return node.label;
  }

  split(rows, depth) {
    const labels = rows.map((row) => row.label);
    const uniqueLabels = [...new Set(labels)];

    if (uniqueLabels.length === 1 || depth >= this.maxDepth || rows.length <= this.minSize) {
      return { label: this.majority(labels) };
    }

    const candidate = this.bestSplit(rows);
    if (!candidate) {
      return { label: this.majority(labels) };
    }

    return {
      feature: candidate.feature,
      value: candidate.value,
      left: this.split(candidate.left, depth + 1),
      right: this.split(candidate.right, depth + 1),
    };
  }

  bestSplit(rows) {
    let best = null;
    const featureCount = rows[0].features.length;

    for (let feature = 0; feature < featureCount; feature += 1) {
      const values = [...new Set(rows.map((row) => row.features[feature]))].sort((a, b) => a - b);
      for (let index = 1; index < values.length; index += 1) {
        const value = (values[index - 1] + values[index]) / 2;
        const left = rows.filter((row) => row.features[feature] < value);
        const right = rows.filter((row) => row.features[feature] >= value);
        if (left.length === 0 || right.length === 0) continue;

        const score = this.gini(left, right);
        if (!best || score < best.score) {
          best = { feature, value, left, right, score };
        }
      }
    }

    return best;
  }

  gini(left, right) {
    const total = left.length + right.length;
    return [left, right].reduce((sum, group) => {
      const labels = group.map((row) => row.label);
      const counts = labels.reduce((acc, label) => {
        acc[label] = (acc[label] || 0) + 1;
        return acc;
      }, {});
      const impurity = 1 - Object.values(counts).reduce((part, count) => {
        const probability = count / group.length;
        return part + probability * probability;
      }, 0);
      return sum + impurity * (group.length / total);
    }, 0);
  }

  majority(labels) {
    const counts = labels.reduce((acc, label) => {
      acc[label] = (acc[label] || 0) + 1;
      return acc;
    }, {});
    return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
  }
}

function jitter(value, amount) {
  return value + (Math.random() * 2 - 1) * amount;
}

function buildTrainingSet() {
  const rows = [];
  const add = (label, base, spread, count = 90) => {
    for (let i = 0; i < count; i += 1) {
      rows.push({
        label,
        features: base.map((value, index) => Math.max(0, jitter(value, spread[index]))),
      });
    }
  };

  add("circle", [0.86, 1.0, 0.76], [0.08, 0.16, 0.09]);
  add("square", [0.66, 1.0, 0.91], [0.08, 0.14, 0.07]);
  add("triangle", [0.50, 1.0, 0.56], [0.09, 0.18, 0.09]);
  return rows;
}

function setupModel() {
  tree = new DecisionTree(5, 6);
  tree.fit(buildTrainingSet());
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    startButton.textContent = "Câmera ativa";
    startButton.disabled = true;
    statusEl.textContent = "Mostre uma forma escura desenhada em papel claro";
    running = true;
    requestAnimationFrame(processFrame);
  } catch (error) {
    statusEl.textContent = "Não foi possível acessar a câmera";
    console.error(error);
  }
}

function processFrame() {
  if (!running || video.readyState < 2) {
    requestAnimationFrame(processFrame);
    return;
  }

  const width = 320;
  const height = Math.round((video.videoHeight / video.videoWidth) * width) || 240;
  frame.width = width;
  frame.height = height;
  overlay.width = overlay.clientWidth;
  overlay.height = overlay.clientHeight;

  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(video, -width, 0, width, height);
  ctx.restore();

  const image = ctx.getImageData(0, 0, width, height);
  const component = findLargestComponent(image, width, height);

  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  if (component && component.area > 450) {
    const features = extractFeatures(component, width, height);
    const prediction = classifyGeometry(features);

    if (prediction) {
      updatePrediction(prediction.label, prediction.confidence, features, component, width, height);
    } else {
      updateRejected(features, component, width, height);
    }
  } else {
    updateIdle();
  }

  requestAnimationFrame(processFrame);
}

function findLargestComponent(image, width, height) {
  const data = image.data;
  const mask = new Uint8Array(width * height);
  let sum = 0;

  for (let i = 0; i < data.length; i += 4) {
    sum += (data[i] + data[i + 1] + data[i + 2]) / 3;
  }
  const average = sum / (width * height);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      const red = data[offset];
      const green = data[offset + 1];
      const blue = data[offset + 2];
      const luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue;
      const darkThreshold = Math.min(145, average - 34);

      if (luma < darkThreshold) {
        mask[y * width + x] = 1;
      }
    }
  }

  return largestComponent(mask, width, height);
}

function largestComponent(mask, width, height) {
  const visited = new Uint8Array(mask.length);
  const queue = [];
  let best = null;

  for (let start = 0; start < mask.length; start += 1) {
    if (!mask[start] || visited[start]) continue;

    let head = 0;
    let area = 0;
    let perimeter = 0;
    let minX = width;
    let minY = height;
    let maxX = 0;
    let maxY = 0;
    queue.length = 0;
    queue.push(start);
    visited[start] = 1;

    while (head < queue.length) {
      const index = queue[head];
      head += 1;
      const x = index % width;
      const y = Math.floor(index / width);
      area += 1;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);

      const neighbors = [index - 1, index + 1, index - width, index + width];
      for (const next of neighbors) {
        const nx = next % width;
        const invalidHorizontal = Math.abs(nx - x) > 1;
        if (next < 0 || next >= mask.length || invalidHorizontal || !mask[next]) {
          perimeter += 1;
        } else if (!visited[next]) {
          visited[next] = 1;
          queue.push(next);
        }
      }
    }

    if (!best || area > best.area) {
      best = { area, perimeter, minX, minY, maxX, maxY };
    }
  }

  return best;
}

function extractFeatures(component, sourceWidth, sourceHeight) {
  const boxWidth = component.maxX - component.minX + 1;
  const boxHeight = component.maxY - component.minY + 1;
  const circularity =
    component.perimeter === 0 ? 0 : (4 * Math.PI * component.area) / (component.perimeter ** 2);
  const aspectRatio = Math.min(boxWidth, boxHeight) / Math.max(boxWidth, boxHeight);
  const extent = component.area / (boxWidth * boxHeight);
  const areaRatio = component.area / (sourceWidth * sourceHeight);
  const boxAreaRatio = (boxWidth * boxHeight) / (sourceWidth * sourceHeight);

  return {
    circularity: clamp(circularity, 0, 1),
    aspectRatio: clamp(aspectRatio, 0, 1.5),
    extent: clamp(extent, 0, 1),
    areaRatio,
    boxAreaRatio,
    boxWidth,
    boxHeight,
  };
}

function classifyGeometry(features) {
  if (
    features.areaRatio < 0.006 ||
    features.areaRatio > 0.24 ||
    features.boxAreaRatio > 0.36 ||
    features.aspectRatio < 0.48 ||
    features.boxWidth < 24 ||
    features.boxHeight < 24
  ) {
    return null;
  }

  const treeLabel = tree.predict([features.circularity, features.aspectRatio, features.extent]);
  const profile = SHAPE_PROFILES[treeLabel];
  const values = [features.circularity, features.aspectRatio, features.extent];
  const distance = values.reduce((total, value, index) => {
    const normalized = (value - profile.center[index]) / profile.tolerance[index];
    return total + normalized * normalized;
  }, 0);
  const confidence = clamp(1 - Math.sqrt(distance / values.length), 0, 1);

  return confidence >= 0.42 ? { label: treeLabel, confidence } : null;
}

function updatePrediction(label, confidence, features, component, sourceWidth, sourceHeight) {
  const shape = SHAPES[label];
  const now = performance.now();
  shapeIcon.textContent = shape.icon;
  shapeIcon.style.color = shape.color;
  shapeName.textContent = `${shape.name} (${shape.note})`;
  predictionCard.style.borderColor = shape.color;
  statusEl.textContent = `Forma detectada pela Árvore de Decisão: ${Math.round(confidence * 100)}%`;

  circularityMeter.value = features.circularity;
  aspectMeter.value = features.aspectRatio;
  extentMeter.value = features.extent;
  drawDetectionBox(component, sourceWidth, sourceHeight, shape.color);

  if (label !== lastPlayedShape || now - lastPlayedAt > 900) {
    playShape(shape);
    addToSequence(label);
    lastPlayedShape = label;
    lastPlayedAt = now;
  }
}

function updateRejected(features, component, sourceWidth, sourceHeight) {
  shapeIcon.textContent = "×";
  shapeIcon.style.color = "#ff6b6b";
  shapeName.textContent = "Objeto ignorado";
  predictionCard.style.borderColor = "var(--danger)";
  statusEl.textContent = "Aponte para uma forma escura simples em papel claro";
  circularityMeter.value = features.circularity;
  aspectMeter.value = features.aspectRatio;
  extentMeter.value = features.extent;
  drawDetectionBox(component, sourceWidth, sourceHeight, "#ff6b6b");
  lastPlayedShape = "";
}

function updateIdle() {
  shapeIcon.textContent = "?";
  shapeIcon.style.color = "#ffca3a";
  shapeName.textContent = "Nenhuma forma detectada";
  predictionCard.style.borderColor = "var(--line)";
  circularityMeter.value = 0;
  aspectMeter.value = 0;
  extentMeter.value = 0;
}

function drawDetectionBox(component, sourceWidth, sourceHeight, color) {
  const scaleX = overlay.width / sourceWidth;
  const scaleY = overlay.height / sourceHeight;
  const x = component.minX * scaleX;
  const y = component.minY * scaleY;
  const width = (component.maxX - component.minX) * scaleX;
  const height = (component.maxY - component.minY) * scaleY;

  overlayCtx.strokeStyle = color;
  overlayCtx.lineWidth = 4;
  overlayCtx.shadowColor = color;
  overlayCtx.shadowBlur = 18;
  overlayCtx.strokeRect(x, y, width, height);
  overlayCtx.shadowBlur = 0;
}

function playShape(shape) {
  if (!soundEnabled) return;
  const BrowserAudioContext = window.AudioContext || window.webkitAudioContext;
  audioCtx = audioCtx || new BrowserAudioContext();

  const oscillator = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  oscillator.type = "sine";
  oscillator.frequency.value = shape.frequency;
  gain.gain.setValueAtTime(0.001, audioCtx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.28, audioCtx.currentTime + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.38);
  oscillator.connect(gain);
  gain.connect(audioCtx.destination);
  oscillator.start();
  oscillator.stop(audioCtx.currentTime + 0.42);
}

function addToSequence(label) {
  sequence.push(label);
  sequence = sequence.slice(-24);
  sequenceEl.innerHTML = "";
  for (const item of sequence) {
    const shape = SHAPES[item];
    const note = document.createElement("span");
    note.className = `note ${item}`;
    note.textContent = shape.icon;
    note.title = `${shape.name} - ${shape.note}`;
    sequenceEl.appendChild(note);
  }
  sequenceCount.textContent = `${sequence.length} ${sequence.length === 1 ? "nota" : "notas"}`;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

startButton.addEventListener("click", startCamera);

soundButton.addEventListener("click", async () => {
  soundEnabled = !soundEnabled;
  soundButton.textContent = soundEnabled ? "Som ligado" : "Som desligado";
  if (soundEnabled && audioCtx?.state === "suspended") {
    await audioCtx.resume();
  }
});

clearButton.addEventListener("click", () => {
  sequence = [];
  sequenceEl.innerHTML = "";
  sequenceCount.textContent = "0 notas";
});

setupModel();
