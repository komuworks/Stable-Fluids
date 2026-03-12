const canvas = document.getElementById('fluid');
const ctx = canvas.getContext('2d', { alpha: false });

const SIM = {
  dt: 0.1,
  diffusion: 0.00008,
  viscosity: 0.00003,
  iterations: 18,
  dyeScale: 120,
  forceScale: 1,
  forceVelocityResponse: 1,
  dyeVelocityResponse: 0,
  impulseSpacing: 20,
  fade: 0.992,
  gridScale: 0.22,
};

let N = 0;
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

const pointer = {
  down: false,
  x: 0,
  y: 0,
  px: 0,
  py: 0,
  hasPrev: false,
  prevT: 0,
  prevDx: 0,
  prevDy: 0,
  hue: 210,
};

function idx(x, y) {
  return x + (N + 2) * y;
}

function allocateGrid(n) {
  N = n;
  size = (N + 2) * (N + 2);

  u = new Float32Array(size);
  v = new Float32Array(size);
  uPrev = new Float32Array(size);
  vPrev = new Float32Array(size);
  dens = new Float32Array(size);
  densPrev = new Float32Array(size);

  imageData = new ImageData(N, N);
  offscreen = document.createElement('canvas');
  offscreen.width = N;
  offscreen.height = N;
  offCtx = offscreen.getContext('2d');
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
  for (let i = 1; i <= N; i += 1) {
    x[idx(0, i)] = b === 1 ? -x[idx(1, i)] : x[idx(1, i)];
    x[idx(N + 1, i)] = b === 1 ? -x[idx(N, i)] : x[idx(N, i)];
    x[idx(i, 0)] = b === 2 ? -x[idx(i, 1)] : x[idx(i, 1)];
    x[idx(i, N + 1)] = b === 2 ? -x[idx(i, N)] : x[idx(i, N)];
  }

  x[idx(0, 0)] = 0.5 * (x[idx(1, 0)] + x[idx(0, 1)]);
  x[idx(0, N + 1)] = 0.5 * (x[idx(1, N + 1)] + x[idx(0, N)]);
  x[idx(N + 1, 0)] = 0.5 * (x[idx(N, 0)] + x[idx(N + 1, 1)]);
  x[idx(N + 1, N + 1)] = 0.5 * (x[idx(N, N + 1)] + x[idx(N + 1, N)]);
}

function linSolve(b, x, x0, a, c) {
  for (let k = 0; k < SIM.iterations; k += 1) {
    for (let j = 1; j <= N; j += 1) {
      for (let i = 1; i <= N; i += 1) {
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
  const a = dt * diff * N * N;
  linSolve(b, x, x0, a, 1 + 4 * a);
}

function advect(b, d, d0, velocX, velocY, dt) {
  const dt0 = dt * N;

  for (let j = 1; j <= N; j += 1) {
    for (let i = 1; i <= N; i += 1) {
      let x = i - dt0 * velocX[idx(i, j)];
      let y = j - dt0 * velocY[idx(i, j)];

      x = Math.max(0.5, Math.min(N + 0.5, x));
      y = Math.max(0.5, Math.min(N + 0.5, y));

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
  for (let j = 1; j <= N; j += 1) {
    for (let i = 1; i <= N; i += 1) {
      div[idx(i, j)] =
        -0.5 *
        (velocX[idx(i + 1, j)] - velocX[idx(i - 1, j)] + velocY[idx(i, j + 1)] - velocY[idx(i, j - 1)]) /
        N;
      p[idx(i, j)] = 0;
    }
  }

  setBoundary(0, div);
  setBoundary(0, p);
  linSolve(0, p, div, 1, 4);

  for (let j = 1; j <= N; j += 1) {
    for (let i = 1; i <= N; i += 1) {
      velocX[idx(i, j)] -= 0.5 * N * (p[idx(i + 1, j)] - p[idx(i - 1, j)]);
      velocY[idx(i, j)] -= 0.5 * N * (p[idx(i, j + 1)] - p[idx(i, j - 1)]);
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
    i: Math.max(1, Math.min(N, Math.floor(x * N) + 1)),
    j: Math.max(1, Math.min(N, Math.floor(y * N) + 1)),
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
  for (let oy = -2; oy <= 2; oy += 1) {
    for (let ox = -2; ox <= 2; ox += 1) {
      const dyeI = Math.max(1, Math.min(N, end.i + ox));
      const dyeJ = Math.max(1, Math.min(N, end.j + oy));
      const dyeK = idx(dyeI, dyeJ);

      densPrev[dyeK] += (SIM.dyeScale + color * 0.25) * dyeFactor;
    }
  }

  pointer.hue = (pointer.hue + 2.4) % 360;
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

function addInterpolatedImpulses(fromX, fromY, toX, toY, elapsedMs, isDrag, prevDx, prevDy) {
  const spacing = Math.max(1, SIM.impulseSpacing);
  const distance = Math.hypot(toX - fromX, toY - fromY);
  const steps = Math.max(1, Math.ceil(distance / spacing));
  const stepElapsedMs = elapsedMs / steps;

  const handleScale = 0.28;
  const c1x = fromX + prevDx * handleScale;
  const c1y = fromY + prevDy * handleScale;
  const c2x = toX - (toX - fromX) * handleScale;
  const c2y = toY - (toY - fromY) * handleScale;

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

function renderDensity() {
  const pixels = imageData.data;
  let p = 0;

  for (let j = 1; j <= N; j += 1) {
    for (let i = 1; i <= N; i += 1) {
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

  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
}


const controls = [
  { id: 'forceAmount', key: 'forceScale', format: (v) => Math.round(v).toString() },
  { id: 'dyeAmount', key: 'dyeScale', format: (v) => Math.round(v).toString() },
  { id: 'forceVelocityResponse', key: 'forceVelocityResponse', format: (v) => Number(v).toFixed(2) },
  { id: 'dyeVelocityResponse', key: 'dyeVelocityResponse', format: (v) => Number(v).toFixed(2) },
  { id: 'impulseSpacing', key: 'impulseSpacing', format: (v) => `${Number(v).toFixed(1)}px` },
];

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
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
}

function loop() {
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

  const nextN = Math.max(64, Math.floor(Math.min(w, h) * SIM.gridScale));
  allocateGrid(nextN);
}

function onPointerDown(e) {
  pointer.down = true;
  pointer.x = e.clientX;
  pointer.y = e.clientY;
  pointer.px = e.clientX;
  pointer.py = e.clientY;
  pointer.hasPrev = true;
  pointer.prevT = e.timeStamp;
  pointer.prevDx = 0;
  pointer.prevDy = 0;
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
  const dx = pointer.x - pointer.px;
  const dy = pointer.y - pointer.py;
  addInterpolatedImpulses(
    pointer.px,
    pointer.py,
    pointer.x,
    pointer.y,
    elapsedMs,
    pointer.down,
    pointer.prevDx,
    pointer.prevDy,
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

bindControls();
resize();
loop();
