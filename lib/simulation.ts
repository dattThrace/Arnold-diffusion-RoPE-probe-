export type SimulationPoint = {
  step: number;
  trialA: number;
  trialB: number;
};

export async function* runSimulation(): AsyncGenerator<{
  progress: number;
  points: SimulationPoint[];
}, void, unknown> {
  const L = 64;
  const d = 4;

  const Q = new Float64Array(L * d);
  const K = new Float64Array(L * d);

  for (let i = 0; i < L; i++) {
    for (let j = 0; j < d; j++) {
      Q[i * d + j] = Math.sin((i * (j + 1) * Math.PI) / 8);
      K[i * d + j] = Math.cos((i * (j + 1) * Math.PI) / 8);
    }
  }

  const Atarget = new Float64Array(L * L);
  for (let i = 0; i < L; i++) {
    let sum = 0;
    for (let j = 0; j < L; j++) {
      const val = Math.exp(-Math.abs(i - j));
      sum += val;
      Atarget[i * L + j] = val;
    }
    for (let j = 0; j < L; j++) {
      Atarget[i * L + j] /= sum;
    }
  }

  // Pre-allocate to avoid GC in loop
  const Qp = new Float64Array(L * d);
  const Kp = new Float64Array(L * d);
  const logits = new Float64Array(L);
  const exps = new Float64Array(L);

  function computeLoss(w1: number, w2: number): number {
    for (let m = 0; m < L; m++) {
      const c1 = Math.cos(m * w1);
      const s1 = Math.sin(m * w1);
      const c2 = Math.cos(m * w2);
      const s2 = Math.sin(m * w2);

      Qp[m * d + 0] = Q[m * d + 0] * c1 - Q[m * d + 1] * s1;
      Qp[m * d + 1] = Q[m * d + 0] * s1 + Q[m * d + 1] * c1;
      Qp[m * d + 2] = Q[m * d + 2] * c2 - Q[m * d + 3] * s2;
      Qp[m * d + 3] = Q[m * d + 2] * s2 + Q[m * d + 3] * c2;

      Kp[m * d + 0] = K[m * d + 0] * c1 - K[m * d + 1] * s1;
      Kp[m * d + 1] = K[m * d + 0] * s1 + K[m * d + 1] * c1;
      Kp[m * d + 2] = K[m * d + 2] * c2 - K[m * d + 3] * s2;
      Kp[m * d + 3] = K[m * d + 2] * s2 + K[m * d + 3] * c2;
    }

    let loss = 0;
    for (let i = 0; i < L; i++) {
      let maxL = -Infinity;
      for (let j = 0; j < L; j++) {
        const dot =
          Qp[i * d + 0] * Kp[j * d + 0] +
          Qp[i * d + 1] * Kp[j * d + 1] +
          Qp[i * d + 2] * Kp[j * d + 2] +
          Qp[i * d + 3] * Kp[j * d + 3];
        const scaledDot = dot / 2.0; // sqrt(4) = 2
        logits[j] = scaledDot;
        if (scaledDot > maxL) {
          maxL = scaledDot;
        }
      }

      let sumExp = 0;
      for (let j = 0; j < L; j++) {
        const e = Math.exp(logits[j] - maxL);
        exps[j] = e;
        sumExp += e;
      }

      for (let j = 0; j < L; j++) {
        const p = exps[j] / sumExp;
        const diff = p - Atarget[i * L + j];
        loss += diff * diff;
      }
    }
    return loss;
  }

  const eps = 1e-6;
  function computeGrad(w1: number, w2: number): [number, number] {
    const lw1p = computeLoss(w1 + eps, w2);
    const lw1m = computeLoss(w1 - eps, w2);
    const g1 = (lw1p - lw1m) / (2 * eps);

    const lw2p = computeLoss(w1, w2 + eps);
    const lw2m = computeLoss(w1, w2 - eps);
    const g2 = (lw2p - lw2m) / (2 * eps);

    return [g1, g2];
  }

  const TOTAL_STEPS = 5000;
  const CHUNK_SIZE = 50; 
  
  // Trial A
  let a_w1 = 1.0;
  let a_w2 = 0.01;
  let a_m1 = 0, a_m2 = 0;
  let a_v1 = 0, a_v2 = 0;

  // Trial B
  const phi = 1.61803398875;
  let b_w1 = phi;
  let b_w2 = phi * phi;
  let b_m1 = 0, b_m2 = 0;
  let b_v1 = 0, b_v2 = 0;

  const lr = 0.01;
  const beta1 = 0.9;
  const beta2 = 0.999;

  const points: SimulationPoint[] = [];

  for (let t = 1; t <= TOTAL_STEPS; t++) {
    // === Trial A Update ===
    const [a_g1, a_g2] = computeGrad(a_w1, a_w2);
    let a_gradNorm = Math.sqrt(a_g1 * a_g1 + a_g2 * a_g2);
    if (a_gradNorm < 1e-10) a_gradNorm = 1e-10; // strictly positive for log scale

    a_m1 = beta1 * a_m1 + (1 - beta1) * a_g1;
    a_m2 = beta1 * a_m2 + (1 - beta1) * a_g2;
    a_v1 = beta2 * a_v1 + (1 - beta2) * (a_g1 * a_g1);
    a_v2 = beta2 * a_v2 + (1 - beta2) * (a_g2 * a_g2);

    const a_m1_hat = a_m1 / (1 - Math.pow(beta1, t));
    const a_m2_hat = a_m2 / (1 - Math.pow(beta1, t));
    const a_v1_hat = a_v1 / (1 - Math.pow(beta2, t));
    const a_v2_hat = a_v2 / (1 - Math.pow(beta2, t));

    a_w1 = a_w1 - (lr * a_m1_hat) / (Math.sqrt(a_v1_hat) + 1e-8);
    a_w2 = a_w2 - (lr * a_m2_hat) / (Math.sqrt(a_v2_hat) + 1e-8);

    // === Trial B Update ===
    const [b_g1, b_g2] = computeGrad(b_w1, b_w2);
    let b_gradNorm = Math.sqrt(b_g1 * b_g1 + b_g2 * b_g2);
    if (b_gradNorm < 1e-10) b_gradNorm = 1e-10; 

    b_m1 = beta1 * b_m1 + (1 - beta1) * b_g1;
    b_m2 = beta1 * b_m2 + (1 - beta1) * b_g2;
    b_v1 = beta2 * b_v1 + (1 - beta2) * (b_g1 * b_g1);
    b_v2 = beta2 * b_v2 + (1 - beta2) * (b_g2 * b_g2);

    const b_m1_hat = b_m1 / (1 - Math.pow(beta1, t));
    const b_m2_hat = b_m2 / (1 - Math.pow(beta1, t));
    const b_v1_hat = b_v1 / (1 - Math.pow(beta2, t));
    const b_v2_hat = b_v2 / (1 - Math.pow(beta2, t));

    b_w1 = b_w1 - (lr * b_m1_hat) / (Math.sqrt(b_v1_hat) + 1e-8);
    b_w2 = b_w2 - (lr * b_m2_hat) / (Math.sqrt(b_v2_hat) + 1e-8);
    
    // Sub-sample to keep array size manageable if needed, but 5000 is okay for recharts.
    // Let's sample every step as requested, or at least store every step.
    points.push({
      step: t,
      trialA: a_gradNorm,
      trialB: b_gradNorm,
    });

    if (t % CHUNK_SIZE === 0) {
      // Yield control back to UI
      yield { progress: t / TOTAL_STEPS, points: [...points] };
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  yield { progress: 1, points };
}
