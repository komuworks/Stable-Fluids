const canvas = document.getElementById('fluid');
let ctx;

const SIM = {
  dt: 0.1,
  diffusion: 0.00008,
  viscosity: 0.00003,
  iterations: 18,
  dyeScale: 20,
  forceScale: 5,
  forceVelocityResponse: 1,
  dyeVelocityResponse: 0,
  dyeBrushRadius: 1,
  impulseSpacing: 2,
  fade: 0.992,
  gridScale: 0.22,
  resolutionDivisor: 2,
  effectiveUpscaleRatio: 1,
  aaMode: 'fxaa',
  aaStrengthBase: 0.55,
  aaMaxStrength: 1.85,
  aaUpscaleThreshold: 2,
  aaStrength: 0,
  impulseInterpolationMode: 'bezier',
  clickBurstForceScale: 25,
  clickBurstDyeScale: 12,
  clickBurstRadius: 1,
};

let gridWidth = 0;
let gridHeight = 0;
let size = 0;
let u;
let v;
let uPrev;
let vPrev;
let dens;
let densPrev;
let imageData;
let offscreen;
let offCtx;
let blurPassA;
let blurPassACtx;
let blurPassB;
let blurPassBCtx;
let blurCompositeData;
let lastFrameTime = 0;
let fpsAccumulatedTime = 0;
let fpsFrameCount = 0;

const fpsOutput = document.getElementById('fpsValue');

const pointer = {
  down: false,
  x: 0,
  y: 0,
  px: 0,
  py: 0,
  hasPrev: false,
  prevT: 0,
  prev2Dx: 0,
  prev2Dy: 0,
  prevDx: 0,
  prevDy: 0,
  hue: 210,
};



function applyFinalAaComposite(source, options = {}) {
  const { allowAaAtLowUpscale = false } = options;
  const lowUpscale = isLowUpscale(SIM.effectiveUpscaleRatio);
  const aaDisabled = SIM.aaMode === 'off';
  ctx.filter = 'none';

  if (lowUpscale && !allowAaAtLowUpscale && aaDisabled) {
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
    return;
  }

  const aaStrength = SIM.aaStrength;
  if (aaDisabled) {
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
    return;
  }

  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);

  if (SIM.aaMode === 'linear') {
    return;
  }

  if (SIM.aaMode === 'blur') {
    ctx.save();
    ctx.globalAlpha = clamp(0.08 + aaStrength * 0.12, 0.08, 0.35);
    ctx.filter = `blur(${(0.35 + aaStrength * 0.45).toFixed(2)}px)`;
    ctx.drawImage(canvas, 0, 0);
    ctx.restore();
    return;
  }

  // FXAA 相当: ごく軽いブラー + サブピクセルの再サンプルでジャギーを緩和。
  ctx.save();
  ctx.globalAlpha = clamp(0.08 + aaStrength * 0.09, 0.08, 0.3);
  ctx.filter = `blur(${(0.25 + aaStrength * 0.35).toFixed(2)}px)`;
  ctx.drawImage(canvas, 0, 0);
  ctx.restore();

  const jitter = clamp(0.2 + aaStrength * 0.22, 0.2, 0.8);
  ctx.save();
  ctx.globalAlpha = clamp(0.04 + aaStrength * 0.04, 0.04, 0.14);
  ctx.drawImage(canvas, -jitter, 0);
  ctx.drawImage(canvas, jitter, 0);
  ctx.drawImage(canvas, 0, -jitter);
  ctx.drawImage(canvas, 0, jitter);
  ctx.restore();
  ctx.filter = 'none';
}


function allocateGrid(width, height) {
  gridWidth = width;
  gridHeight = height;
  size = (gridWidth + 2) * (gridHeight + 2);

  u = new Float32Array(size);
  v = new Float32Array(size);
  uPrev = new Float32Array(size);
  vPrev = new Float32Array(size);
  dens = new Float32Array(size);
  densPrev = new Float32Array(size);

  imageData = new ImageData(gridWidth, gridHeight);
  offscreen = document.createElement('canvas');
  offscreen.width = gridWidth;
  offscreen.height = gridHeight;
  offCtx = offscreen.getContext('2d');

  blurPassA = document.createElement('canvas');
  blurPassA.width = gridWidth;
  blurPassA.height = gridHeight;
  blurPassACtx = blurPassA.getContext('2d');

  blurPassB = document.createElement('canvas');
  blurPassB.width = gridWidth;
  blurPassB.height = gridHeight;
  blurPassBCtx = blurPassB.getContext('2d');

  blurCompositeData = new ImageData(gridWidth, gridHeight);

}

function reset() {
  u.fill(0);
  v.fill(0);
  uPrev.fill(0);
  vPrev.fill(0);
  dens.fill(0);
  densPrev.fill(0);
}

function addSource(x, s, dt) {
  for (let i = 0; i < size; i += 1) {
    x[i] += dt * s[i];
  }
}

function setBoundary(b, x) {
  for (let y = 1; y <= gridHeight; y += 1) {
    x[idx(0, y)] = b === 1 ? -x[idx(1, y)] : x[idx(1, y)];
    x[idx(gridWidth + 1, y)] = b === 1 ? -x[idx(gridWidth, y)] : x[idx(gridWidth, y)];
  }

  for (let xPos = 1; xPos <= gridWidth; xPos += 1) {
    x[idx(xPos, 0)] = b === 2 ? -x[idx(xPos, 1)] : x[idx(xPos, 1)];
    x[idx(xPos, gridHeight + 1)] = b === 2 ? -x[idx(xPos, gridHeight)] : x[idx(xPos, gridHeight)];
  }

  x[idx(0, 0)] = 0.5 * (x[idx(1, 0)] + x[idx(0, 1)]);
  x[idx(0, gridHeight + 1)] = 0.5 * (x[idx(1, gridHeight + 1)] + x[idx(0, gridHeight)]);
  x[idx(gridWidth + 1, 0)] = 0.5 * (x[idx(gridWidth, 0)] + x[idx(gridWidth + 1, 1)]);
  x[idx(gridWidth + 1, gridHeight + 1)] =
    0.5 * (x[idx(gridWidth, gridHeight + 1)] + x[idx(gridWidth + 1, gridHeight)]);
}

function linSolve(b, x, x0, a, c) {
  for (let k = 0; k < SIM.iterations; k += 1) {
    for (let j = 1; j <= gridHeight; j += 1) {
      for (let i = 1; i <= gridWidth; i += 1) {
        x[idx(i, j)] = (
          x0[idx(i, j)] +
          a *
            (x[idx(i - 1, j)] + x[idx(i + 1, j)] + x[idx(i, j - 1)] + x[idx(i, j + 1)])
        ) / c;
      }
    }
    setBoundary(b, x);
  }
}

function diffuse(b, x, x0, diff, dt) {
  const gridScale = Math.max(gridWidth, gridHeight);
  const a = dt * diff * gridScale * gridScale;
  linSolve(b, x, x0, a, 1 + 4 * a);
}

function advect(b, d, d0, velocX, velocY, dt) {
  const dtX = dt * gridWidth;
  const dtY = dt * gridHeight;

  for (let j = 1; j <= gridHeight; j += 1) {
    for (let i = 1; i <= gridWidth; i += 1) {
      let x = i - dtX * velocX[idx(i, j)];
      let y = j - dtY * velocY[idx(i, j)];

      x = Math.max(0.5, Math.min(gridWidth + 0.5, x));
      y = Math.max(0.5, Math.min(gridHeight + 0.5, y));

      const i0 = Math.floor(x);
      const i1 = i0 + 1;
      const j0 = Math.floor(y);
      const j1 = j0 + 1;
      const s1 = x - i0;
      const s0 = 1 - s1;
      const t1 = y - j0;
      const t0 = 1 - t1;

      d[idx(i, j)] =
        s0 * (t0 * d0[idx(i0, j0)] + t1 * d0[idx(i0, j1)]) +
        s1 * (t0 * d0[idx(i1, j0)] + t1 * d0[idx(i1, j1)]);
    }
  }

  setBoundary(b, d);
}

function project(velocX, velocY, p, div) {
  for (let j = 1; j <= gridHeight; j += 1) {
    for (let i = 1; i <= gridWidth; i += 1) {
      div[idx(i, j)] =
        -0.5 *
        (velocX[idx(i + 1, j)] - velocX[idx(i - 1, j)] + velocY[idx(i, j + 1)] - velocY[idx(i, j - 1)]) /
        Math.max(gridWidth, gridHeight);
      p[idx(i, j)] = 0;
    }
  }

  setBoundary(0, div);
  setBoundary(0, p);
  linSolve(0, p, div, 1, 4);

  for (let j = 1; j <= gridHeight; j += 1) {
    for (let i = 1; i <= gridWidth; i += 1) {
      velocX[idx(i, j)] -= 0.5 * gridWidth * (p[idx(i + 1, j)] - p[idx(i - 1, j)]);
      velocY[idx(i, j)] -= 0.5 * gridHeight * (p[idx(i, j + 1)] - p[idx(i, j - 1)]);
    }
  }

  setBoundary(1, velocX);
  setBoundary(2, velocY);
}

function velocityStep() {
  addSource(u, uPrev, SIM.dt);
  addSource(v, vPrev, SIM.dt);

  [uPrev, u] = [u, uPrev];
  diffuse(1, u, uPrev, SIM.viscosity, SIM.dt);

  [vPrev, v] = [v, vPrev];
  diffuse(2, v, vPrev, SIM.viscosity, SIM.dt);

  project(u, v, uPrev, vPrev);

  [uPrev, u] = [u, uPrev];
  [vPrev, v] = [v, vPrev];
  advect(1, u, uPrev, uPrev, vPrev, SIM.dt);
  advect(2, v, vPrev, uPrev, vPrev, SIM.dt);

  project(u, v, uPrev, vPrev);

  uPrev.fill(0);
  vPrev.fill(0);
}

function densityStep() {
  addSource(dens, densPrev, SIM.dt);

  [densPrev, dens] = [dens, densPrev];
  diffuse(0, dens, densPrev, SIM.diffusion, SIM.dt);

  [densPrev, dens] = [dens, densPrev];
  advect(0, dens, densPrev, u, v, SIM.dt);

  for (let i = 0; i < size; i += 1) {
    dens[i] *= SIM.fade;
  }

  densPrev.fill(0);
}

function toGrid(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const x = (clientX - rect.left) / rect.width;
  const y = (clientY - rect.top) / rect.height;
  return {
    i: Math.max(1, Math.min(gridWidth, Math.floor(x * gridWidth) + 1)),
    j: Math.max(1, Math.min(gridHeight, Math.floor(y * gridHeight) + 1)),
  };
}

function addImpulse(fromX, fromY, toX, toY, elapsedMs, isDrag) {
  const start = toGrid(fromX, fromY);
  const end = toGrid(toX, toY);
  const di = end.i - start.i;
  const dj = end.j - start.j;
  const distance = Math.hypot(toX - fromX, toY - fromY);
  const safeElapsedMs = Math.max(1, elapsedMs);
  const velocity = distance / safeElapsedMs;

  const forceVelocityFactor = Math.max(0, 1 + SIM.forceVelocityResponse * (velocity - 1));
  const dyeVelocityFactor = Math.max(0, 1 + SIM.dyeVelocityResponse * (velocity - 1));
  const forceAmount = SIM.forceScale * forceVelocityFactor;

  const forceK = idx(end.i, end.j);
  uPrev[forceK] += di * forceAmount;
  vPrev[forceK] += dj * forceAmount;

  const color = pointer.hue;
  const dyeFactor = (isDrag ? 1.5 : 1) * dyeVelocityFactor;
  const dyeRadius = Math.max(1, Math.round(SIM.dyeBrushRadius));
  const dyeRadiusSq = dyeRadius * dyeRadius;
  for (let oy = -dyeRadius; oy <= dyeRadius; oy += 1) {
    for (let ox = -dyeRadius; ox <= dyeRadius; ox += 1) {
      if (ox * ox + oy * oy > dyeRadiusSq) {
        continue;
      }

      const dyeI = Math.max(1, Math.min(gridWidth, end.i + ox));
      const dyeJ = Math.max(1, Math.min(gridHeight, end.j + oy));
      const dyeK = idx(dyeI, dyeJ);

      densPrev[dyeK] += (SIM.dyeScale + color * 0.25) * dyeFactor;
    }
  }

  pointer.hue = (pointer.hue + 2.4) % 360;
}

function emitRadialClickBurst(clientX, clientY) {
  const center = toGrid(clientX, clientY);
  const radius = Math.max(1, Math.round(SIM.clickBurstRadius));
  const radiusSq = radius * radius;
  const color = pointer.hue;

  for (let oy = -radius; oy <= radius; oy += 1) {
    for (let ox = -radius; ox <= radius; ox += 1) {
      const distSq = ox * ox + oy * oy;
      if (distSq > radiusSq) {
        continue;
      }

      const i = Math.max(1, Math.min(gridWidth, center.i + ox));
      const j = Math.max(1, Math.min(gridHeight, center.j + oy));
      const k = idx(i, j);
      const dist = Math.sqrt(distSq);
      const nx = dist > 0 ? ox / dist : 0;
      const ny = dist > 0 ? oy / dist : 0;
      const falloff = 1 - dist / radius;
      const force = SIM.forceScale * SIM.clickBurstForceScale * falloff;

      uPrev[k] += nx * force;
      vPrev[k] += ny * force;
      densPrev[k] += (SIM.dyeScale + color * 0.25) * SIM.clickBurstDyeScale * falloff;
    }
  }

  pointer.hue = (pointer.hue + 8) % 360;
}

function cubicBezierPoint(p0, p1, p2, p3, t) {
  const mt = 1 - t;
  const mt2 = mt * mt;
  const t2 = t * t;
  return (
    mt2 * mt * p0 +
    3 * mt2 * t * p1 +
    3 * mt * t2 * p2 +
    t2 * t * p3
  );
}

function bezierArcLength(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, samples = 6) {
  let length = 0;
  let lastX = p0x;
  let lastY = p0y;

  for (let i = 1; i <= samples; i += 1) {
    const t = i / samples;
    const x = cubicBezierPoint(p0x, p1x, p2x, p3x, t);
    const y = cubicBezierPoint(p0y, p1y, p2y, p3y, t);
    length += Math.hypot(x - lastX, y - lastY);
    lastX = x;
    lastY = y;
  }

  return length;
}

function addLinearInterpolatedImpulses(fromX, fromY, toX, toY, elapsedMs, isDrag) {
  const spacing = Math.max(1, SIM.impulseSpacing);
  const lineLength = Math.hypot(toX - fromX, toY - fromY);
  const steps = Math.max(1, Math.ceil(lineLength / spacing));
  const stepElapsedMs = elapsedMs / steps;

  for (let step = 1; step <= steps; step += 1) {
    const t0 = (step - 1) / steps;
    const t1 = step / steps;
    const sx = fromX + (toX - fromX) * t0;
    const sy = fromY + (toY - fromY) * t0;
    const ex = fromX + (toX - fromX) * t1;
    const ey = fromY + (toY - fromY) * t1;
    addImpulse(sx, sy, ex, ey, stepElapsedMs, isDrag);
  }
}

function clampVectorLength(x, y, maxLength) {
  const length = Math.hypot(x, y);
  if (length <= maxLength || length === 0) {
    return { x, y };
  }

  const scale = maxLength / length;
  return {
    x: x * scale,
    y: y * scale,
  };
}

function addBezierInterpolatedImpulses(
  fromX,
  fromY,
  toX,
  toY,
  elapsedMs,
  isDrag,
  prev2Dx,
  prev2Dy,
  prevDx,
  prevDy,
  currDx,
  currDy,
) {
  const spacing = Math.max(1, SIM.impulseSpacing);

  // Weighted history tangents converted to Bezier handles for C1 continuity.
  const tangentSmoothing = 0.7;
  let m0x = (prev2Dx * 0.2 + prevDx * 0.5 + currDx * 0.3) * tangentSmoothing;
  let m0y = (prev2Dy * 0.2 + prevDy * 0.5 + currDy * 0.3) * tangentSmoothing;
  let m1x = (prev2Dx * 0.1 + prevDx * 0.3 + currDx * 0.6) * tangentSmoothing;
  let m1y = (prev2Dy * 0.1 + prevDy * 0.3 + currDy * 0.6) * tangentSmoothing;

  const segmentLength = Math.hypot(toX - fromX, toY - fromY);
  const maxTangent = segmentLength * 1.5;
  const clampedM0 = clampVectorLength(m0x, m0y, maxTangent);
  const clampedM1 = clampVectorLength(m1x, m1y, maxTangent);
  m0x = clampedM0.x;
  m0y = clampedM0.y;
  m1x = clampedM1.x;
  m1y = clampedM1.y;

  const c1x = fromX + m0x / 3;
  const c1y = fromY + m0y / 3;
  const c2x = toX - m1x / 3;
  const c2y = toY - m1y / 3;

  const curveLength = bezierArcLength(fromX, fromY, c1x, c1y, c2x, c2y, toX, toY);
  const steps = Math.max(1, Math.ceil(curveLength / spacing));
  const stepElapsedMs = elapsedMs / steps;

  for (let step = 1; step <= steps; step += 1) {
    const t0 = (step - 1) / steps;
    const t1 = step / steps;
    const sx = cubicBezierPoint(fromX, c1x, c2x, toX, t0);
    const sy = cubicBezierPoint(fromY, c1y, c2y, toY, t0);
    const ex = cubicBezierPoint(fromX, c1x, c2x, toX, t1);
    const ey = cubicBezierPoint(fromY, c1y, c2y, toY, t1);
    addImpulse(sx, sy, ex, ey, stepElapsedMs, isDrag);
  }
}

function addInterpolatedImpulses(
  fromX,
  fromY,
  toX,
  toY,
  elapsedMs,
  isDrag,
  prev2Dx,
  prev2Dy,
  prevDx,
  prevDy,
  currDx,
  currDy,
) {
  if (SIM.impulseInterpolationMode === 'linear') {
    addLinearInterpolatedImpulses(fromX, fromY, toX, toY, elapsedMs, isDrag);
    return;
  }

  addBezierInterpolatedImpulses(
    fromX,
    fromY,
    toX,
    toY,
    elapsedMs,
    isDrag,
    prev2Dx,
    prev2Dy,
    prevDx,
    prevDy,
    currDx,
    currDy,
  );
}

function renderDensityCpu() {
  const pixels = imageData.data;
  let p = 0;

  for (let j = 1; j <= gridHeight; j += 1) {
    for (let i = 1; i <= gridWidth; i += 1) {
      const d = Math.min(255, dens[idx(i, j)]);
      const glow = Math.min(255, d * 1.35);
      pixels[p] = Math.min(255, glow * 0.4);
      pixels[p + 1] = Math.min(255, glow * 0.75);
      pixels[p + 2] = Math.min(255, glow * 1.15);
      pixels[p + 3] = 255;
      p += 4;
    }
  }

  offCtx.putImageData(imageData, 0, 0);

  const blurRadius = clamp(Math.log2(Math.max(1, SIM.resolutionDivisor)), 0, 2);
  const blurPasses = blurRadius > 1.2 ? 2 : blurRadius > 0.15 ? 1 : 0;

  if (blurPasses > 0) {
    blurPassACtx.clearRect(0, 0, gridWidth, gridHeight);
    blurPassACtx.filter = `blur(${blurRadius.toFixed(2)}px)`;
    blurPassACtx.drawImage(offscreen, 0, 0);
    blurPassACtx.filter = 'none';

    let blurredCtx = blurPassACtx;

    if (blurPasses === 2) {
      blurPassBCtx.clearRect(0, 0, gridWidth, gridHeight);
      blurPassBCtx.filter = `blur(${(blurRadius * 0.75).toFixed(2)}px)`;
      blurPassBCtx.drawImage(blurPassA, 0, 0);
      blurPassBCtx.filter = 'none';
      blurredCtx = blurPassBCtx;
    }

    const srcPixels = imageData.data;
    const blurredPixels = blurredCtx.getImageData(0, 0, gridWidth, gridHeight).data;
    const outPixels = blurCompositeData.data;
    const blurBlendMax = clamp(0.28 + blurRadius * 0.14, 0.15, 0.55);

    let pIdx = 0;
    for (let j = 1; j <= gridHeight; j += 1) {
      for (let i = 1; i <= gridWidth; i += 1) {
        const dx = Math.abs(dens[idx(i + 1, j)] - dens[idx(i - 1, j)]);
        const dy = Math.abs(dens[idx(i, j + 1)] - dens[idx(i, j - 1)]);
        const grad = dx + dy;
        const edgeWeight = clamp(grad / 160, 0, 1);
        const smoothWeight = (1 - edgeWeight) ** 2;
        const blend = smoothWeight * blurBlendMax;
        const invBlend = 1 - blend;

        outPixels[pIdx] = srcPixels[pIdx] * invBlend + blurredPixels[pIdx] * blend;
        outPixels[pIdx + 1] = srcPixels[pIdx + 1] * invBlend + blurredPixels[pIdx + 1] * blend;
        outPixels[pIdx + 2] = srcPixels[pIdx + 2] * invBlend + blurredPixels[pIdx + 2] * blend;
        outPixels[pIdx + 3] = 255;
        pIdx += 4;
      }
    }

    offCtx.putImageData(blurCompositeData, 0, 0);
  }

  applyFinalAaComposite(offscreen);
}



function renderDensity() {
  renderDensityCpu();
}

const controls = [
  { id: 'forceAmount', key: 'forceScale', format: (v) => Math.round(v).toString() },
  { id: 'dyeAmount', key: 'dyeScale', format: (v) => Math.round(v).toString() },
  { id: 'dyeBrushSize', key: 'dyeBrushRadius', format: (v) => `${Math.round(Number(v))}px` },
  { id: 'forceVelocityResponse', key: 'forceVelocityResponse', format: (v) => Number(v).toFixed(2) },
  { id: 'dyeVelocityResponse', key: 'dyeVelocityResponse', format: (v) => Number(v).toFixed(2) },
  { id: 'impulseSpacing', key: 'impulseSpacing', format: (v) => `${Number(v).toFixed(1)}px` },
  { id: 'clickForceAmount', key: 'clickBurstForceScale', format: (v) => Number(v).toFixed(1) },
  { id: 'clickDyeAmount', key: 'clickBurstDyeScale', format: (v) => Number(v).toFixed(1) },
  { id: 'clickRadius', key: 'clickBurstRadius', format: (v) => `${Math.round(Number(v))}px` },
];

const resolutionOptions = new Set([1, 2, 4, 8, 16]);
const interpolationModes = new Set(['bezier', 'linear']);
const aaModes = new Set(['off', 'linear', 'blur', 'fxaa']);

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}


function isLowUpscale(upscaleRatio) {
  return upscaleRatio <= SIM.aaUpscaleThreshold;
}

function computeRecommendedAaStrength(upscaleRatio) {
  if (isLowUpscale(upscaleRatio)) {
    return 0;
  }

  const upscaleSteps = Math.log2(upscaleRatio / SIM.aaUpscaleThreshold);
  const strength = SIM.aaStrengthBase + upscaleSteps * 0.55;
  return clamp(strength, 0, SIM.aaMaxStrength);
}

function getAaModeLabel(mode) {
  const aaModeLabels = {
    off: 'オフ',
    linear: '線形',
    blur: 'ブラー',
    fxaa: 'FXAA相当',
  };

  return aaModeLabels[mode] ?? aaModeLabels.fxaa;
}

function getAaStrengthMessage() {
  const recommended = SIM.aaStrength.toFixed(2);

  if (SIM.aaMode === 'off') {
    return `${recommended} (モード: オフ)`;
  }

  if (SIM.aaStrength === 0) {
    return `${recommended} (推奨0だが ${getAaModeLabel(SIM.aaMode)} は適用中)`;
  }

  return `${recommended} (${getAaModeLabel(SIM.aaMode)} 適用中)`;
}

function updateAaModeDisplay() {
  const aaModeOutput = document.getElementById('aaModeValue');
  if (!aaModeOutput) {
    return;
  }

  const modeLabel = getAaModeLabel(SIM.aaMode);
  if (SIM.aaMode === 'off') {
    aaModeOutput.textContent = `${modeLabel} (AAなし)`;
    return;
  }

  if (isLowUpscale(SIM.effectiveUpscaleRatio)) {
    aaModeOutput.textContent = `${modeLabel} (低upscaleでも適用)`;
    return;
  }

  aaModeOutput.textContent = `${modeLabel} (推奨)`;
}

function updateAaStrengthDisplay() {
  SIM.aaStrength = computeRecommendedAaStrength(SIM.effectiveUpscaleRatio);
  const aaStrengthOutput = document.getElementById('aaStrengthValue');
  if (aaStrengthOutput) {
    aaStrengthOutput.textContent = getAaStrengthMessage();
  }

  updateAaModeDisplay();
}

function bindControls() {
  for (const control of controls) {
    const slider = document.getElementById(control.id);
    const numberInput = document.getElementById(`${control.id}Number`);
    const output = document.getElementById(`${control.id}Value`);
    if (!slider || !numberInput || !output) continue;

    const min = Number(slider.min);
    const max = Number(slider.max);

    const applyValue = (rawValue) => {
      const parsed = Number(rawValue);
      const baseValue = Number.isFinite(parsed) ? parsed : SIM[control.key];
      const value = clamp(baseValue, min, max);
      slider.value = String(value);
      numberInput.value = String(value);
      SIM[control.key] = value;
      output.textContent = control.format(value);
    };

    slider.addEventListener('input', () => applyValue(slider.value));
    numberInput.addEventListener('input', () => applyValue(numberInput.value));
    applyValue(slider.value);
  }

  const resolutionSelect = document.getElementById('simulationResolution');
  const resolutionOutput = document.getElementById('simulationResolutionValue');
  if (resolutionSelect && resolutionOutput) {
    const applyResolution = (rawValue) => {
      const parsed = Number(rawValue);
      const value = resolutionOptions.has(parsed) ? parsed : 1;
      resolutionSelect.value = String(value);
      SIM.resolutionDivisor = value;
      resolutionOutput.textContent = `1/${value}`;
      resize();
    };

    resolutionSelect.addEventListener('change', () => applyResolution(resolutionSelect.value));
    applyResolution(resolutionSelect.value);
  }


  const aaModeSelect = document.getElementById('aaMode');
  if (aaModeSelect) {
    const applyAaMode = (rawValue) => {
      const value = aaModes.has(rawValue) ? rawValue : 'fxaa';
      aaModeSelect.value = value;
      SIM.aaMode = value;
      updateAaStrengthDisplay();
    };

    aaModeSelect.addEventListener('change', () => applyAaMode(aaModeSelect.value));
    applyAaMode(aaModeSelect.value);
  }

  updateAaStrengthDisplay();

  const interpolationModeSelect = document.getElementById('impulseInterpolationMode');
  const interpolationModeOutput = document.getElementById('impulseInterpolationModeValue');
  if (interpolationModeSelect && interpolationModeOutput) {
    const modeLabels = {
      bezier: 'ベジェ',
      linear: '直線',
    };

    const applyInterpolationMode = (rawValue) => {
      const value = interpolationModes.has(rawValue) ? rawValue : 'bezier';
      interpolationModeSelect.value = value;
      SIM.impulseInterpolationMode = value;
      interpolationModeOutput.textContent = modeLabels[value];
    };

    interpolationModeSelect.addEventListener('change', () => applyInterpolationMode(interpolationModeSelect.value));
    applyInterpolationMode(interpolationModeSelect.value);
  }
}

function loop() {
  const now = performance.now();
  if (lastFrameTime > 0) {
    const frameTime = now - lastFrameTime;
    fpsAccumulatedTime += frameTime;
    fpsFrameCount += 1;

    if (fpsOutput && fpsAccumulatedTime >= 250) {
      const averageFrameTime = fpsAccumulatedTime / fpsFrameCount;
      const fps = 1000 / averageFrameTime;
      fpsOutput.textContent = fps.toFixed(1);
      fpsAccumulatedTime = 0;
      fpsFrameCount = 0;
    }
  }
  lastFrameTime = now;

  velocityStep();
  densityStep();
  renderDensity();
  requestAnimationFrame(loop);
}

function resize() {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.floor(window.innerWidth * dpr);
  const h = Math.floor(window.innerHeight * dpr);
  canvas.width = w;
  canvas.height = h;

  const longSide = Math.max(w, h);
  const defaultResolution = Math.max(64, Math.floor(longSide * SIM.gridScale));
  const baseResolution = Math.max(16, Math.floor(defaultResolution / SIM.resolutionDivisor));
  const aspect = w / h;
  const nextGridWidth = Math.max(16, Math.floor(aspect >= 1 ? baseResolution : baseResolution * aspect));
  const nextGridHeight = Math.max(16, Math.floor(aspect >= 1 ? baseResolution / aspect : baseResolution));

  allocateGrid(nextGridWidth, nextGridHeight);
  const totalPixels = canvas.width * canvas.height;
  const totalGridCells = gridWidth * gridHeight;
  SIM.effectiveUpscaleRatio = totalGridCells > 0
    ? Math.sqrt(totalPixels / totalGridCells)
    : 1;
  updateAaStrengthDisplay();


  reset();
}

function onPointerDown(e) {
  pointer.down = true;
  pointer.x = e.clientX;
  pointer.y = e.clientY;
  pointer.px = e.clientX;
  pointer.py = e.clientY;
  pointer.hasPrev = true;
  pointer.prevT = e.timeStamp;
  pointer.prev2Dx = 0;
  pointer.prev2Dy = 0;
  pointer.prevDx = 0;
  pointer.prevDy = 0;
  emitRadialClickBurst(e.clientX, e.clientY);
}

function onPointerMove(e) {
  if (!pointer.hasPrev) {
    pointer.px = e.clientX;
    pointer.py = e.clientY;
    pointer.prevT = e.timeStamp;
    pointer.hasPrev = true;
  }

  pointer.x = e.clientX;
  pointer.y = e.clientY;
  const elapsedMs = e.timeStamp - pointer.prevT;
  pointer.prev2Dx = pointer.prevDx;
  pointer.prev2Dy = pointer.prevDy;
  const dx = pointer.x - pointer.px;
  const dy = pointer.y - pointer.py;
  addInterpolatedImpulses(
    pointer.px,
    pointer.py,
    pointer.x,
    pointer.y,
    elapsedMs,
    pointer.down,
    pointer.prev2Dx,
    pointer.prev2Dy,
    pointer.prevDx,
    pointer.prevDy,
    dx,
    dy,
  );
  pointer.prevDx = dx;
  pointer.prevDy = dy;
  pointer.px = pointer.x;
  pointer.py = pointer.y;
  pointer.prevT = e.timeStamp;
}

function onPointerUp() {
  pointer.down = false;
}

window.addEventListener('resize', resize);
window.addEventListener('keydown', (e) => {
  if (e.key.toLowerCase() === 'r') {
    reset();
  }
});
canvas.addEventListener('pointerdown', onPointerDown);
canvas.addEventListener('pointermove', onPointerMove);
window.addEventListener('pointerup', onPointerUp);
window.addEventListener('pointercancel', onPointerUp);

function start() {
  ctx = canvas.getContext('2d', { alpha: false });
  if (!ctx) {
    throw new Error('Canvas2D context is required.');
  }

  resize();

  bindControls();
  loop();
}

start();
