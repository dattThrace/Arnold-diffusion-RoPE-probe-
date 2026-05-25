export interface SimulationConfig {
    d: number;
    L: number;
    tau: number;
    alpha: number;
    beta1: number;
    noiseSigma: number;
}

export interface SimulationMetrics {
    step: number;
    gnormA: number;
    driftA: number;
    resA: number;
    gnormB: number;
    driftB: number;
    resB: number;
    wA1: number;
    mA1: number;
    wB1: number;
    mB1: number;
}

export interface SimulationResult {
    metrics: SimulationMetrics[];
    step: number;
    done: boolean;
    AhA: Float64Array; // Attention Heatmap A
    AhB: Float64Array; // Attention Heatmap B
}

export class ArnoldSimulation {
    cfg: SimulationConfig;
    k: number;
    
    Q: Float64Array;
    K: Float64Array;
    targetA: Float64Array;
    
    wA: Float64Array;
    wB: Float64Array;
    wA0: Float64Array;
    wB0: Float64Array;
    
    mA: Float64Array;
    mB: Float64Array;
    vA: Float64Array;
    vB: Float64Array;
    
    step: number = 0;
    beta2: number = 0.999;
    
    Kres: Float64Array[]; // for resonance proximity
    
    // Pre-allocated scratchpads to prevent any allocations in hot execution loops
    Qp_scratch: Float64Array;
    Kp_scratch: Float64Array;
    logits_scratch: Float64Array;
    exps_scratch: Float64Array;
    
    gradA_scratch: Float64Array;
    gradB_scratch: Float64Array;
    
    ahA_buffer: Float64Array;
    ahB_buffer: Float64Array;
    
    constructor(cfg: SimulationConfig) {
        this.cfg = cfg;
        this.k = cfg.d / 2;
        
        this.Q = new Float64Array(cfg.L * cfg.d);
        this.K = new Float64Array(cfg.L * cfg.d);
        for(let i=0; i<cfg.L; i++) {
            for(let j=0; j<cfg.d; j++) {
                this.Q[i*cfg.d + j] = Math.sin((i * (j + 1) * Math.PI) / 8);
                this.K[i*cfg.d + j] = Math.cos((i * (j + 1) * Math.PI) / 8);
            }
        }
        
        this.targetA = new Float64Array(cfg.L * cfg.L);
        for (let i = 0; i < cfg.L; i++) {
            let sum = 0;
            for (let j = 0; j < cfg.L; j++) {
                const val = Math.exp(-Math.abs(i - j));
                sum += val;
                this.targetA[i * cfg.L + j] = val;
            }
            for (let j = 0; j < cfg.L; j++) {
                this.targetA[i * cfg.L + j] /= sum;
            }
        }
        
        this.wA = new Float64Array(this.k);
        this.wB = new Float64Array(this.k);
        this.wA0 = new Float64Array(this.k);
        this.wB0 = new Float64Array(this.k);
        
        const phi = 1.61803398875;
        for(let j=0; j<this.k; j++) {
            // Trial A: Geometric Initialization
            this.wA[j] = Math.pow(10000, - (2*j) / cfg.d);
            this.wA0[j] = this.wA[j];
            
            // Trial B: Algebraic/Diophantine
            this.wB[j] = Math.pow(phi, j + 1);
            this.wB0[j] = this.wB[j];
        }
        
        this.mA = new Float64Array(this.k);
        this.mB = new Float64Array(this.k);
        this.vA = new Float64Array(this.k);
        this.vB = new Float64Array(this.k);
        
        this.Kres = this.generateKres(this.k);
        
        // Initialize scratchpads matching sequence length configuration
        this.Qp_scratch = new Float64Array(cfg.L * cfg.d);
        this.Kp_scratch = new Float64Array(cfg.L * cfg.d);
        this.logits_scratch = new Float64Array(cfg.L);
        this.exps_scratch = new Float64Array(cfg.L);
        
        this.gradA_scratch = new Float64Array(this.k);
        this.gradB_scratch = new Float64Array(this.k);
        
        this.ahA_buffer = new Float64Array(cfg.L * cfg.L);
        this.ahB_buffer = new Float64Array(cfg.L * cfg.L);
    }
    
    generateKres(k: number): Float64Array[] {
        const res: Float64Array[] = [];
        const maxVal = 3;
        const compute = (path: number[], depth: number, l1: number) => {
            if (depth === k) {
                if (l1 > 0 && l1 <= 4) res.push(new Float64Array(path));
                return;
            }
            for (let v = -maxVal; v <= maxVal; v++) {
                if (l1 + Math.abs(v) <= 4) {
                    path.push(v);
                    compute(path, depth + 1, l1 + Math.abs(v));
                    path.pop();
                }
            }
        };
        compute([], 0, 0);
        return res;
    }
    
    getLossAndA(w: Float64Array, outA: Float64Array | null): number {
        const L = this.cfg.L;
        const d = this.cfg.d;
        const k = this.k;
        const tau = this.cfg.tau;
        
        const Qp = this.Qp_scratch;
        const Kp = this.Kp_scratch;
        
        for (let m = 0; m < L; m++) {
            const m_d = m * d;
            for (let j = 0; j < k; j++) {
                const j2 = 2 * j;
                const c = Math.cos(m * w[j]);
                const s = Math.sin(m * w[j]);
                const q0 = this.Q[m_d + j2];
                const q1 = this.Q[m_d + j2 + 1];
                Qp[m_d + j2] = q0 * c - q1 * s;
                Qp[m_d + j2 + 1] = q0 * s + q1 * c;
                
                const k0 = this.K[m_d + j2];
                const k1 = this.K[m_d + j2 + 1];
                Kp[m_d + j2] = k0 * c - k1 * s;
                Kp[m_d + j2 + 1] = k0 * s + k1 * c;
            }
        }
        
        let loss = 0;
        const logits = this.logits_scratch;
        const exps = this.exps_scratch;
        const scale = tau * Math.sqrt(d);
        
        for (let i = 0; i < L; i++) {
            const i_d = i * d;
            let maxL = -Infinity;
            for (let j = 0; j < L; j++) {
                const j_d = j * d;
                let dot = 0;
                for (let v = 0; v < d; v++) {
                    dot += Qp[i_d + v] * Kp[j_d + v];
                }
                dot = dot / scale;
                logits[j] = dot;
                if (dot > maxL) maxL = dot;
            }
            
            let sumExp = 0;
            for (let j = 0; j < L; j++) {
                const e = Math.exp(logits[j] - maxL);
                exps[j] = e;
                sumExp += e;
            }
            
            const i_L = i * L;
            for (let j = 0; j < L; j++) {
                const p = exps[j] / sumExp;
                if (outA) {
                    outA[i_L + j] = p;
                }
                const diff = p - this.targetA[i_L + j];
                loss += diff * diff;
            }
        }
        
        return loss;
    }
    
    getGradient(w: Float64Array, outA: Float64Array | null, gradOut: Float64Array): void {
        const eps = 1e-6;
        this.getLossAndA(w, outA);
        
        for(let j=0; j<this.k; j++) {
            const original_wj = w[j];
            
            w[j] = original_wj + eps;
            const p = this.getLossAndA(w, null);
            
            w[j] = original_wj - eps;
            const m = this.getLossAndA(w, null);
            
            w[j] = original_wj;
            let gj = (p - m) / (2 * eps);
            
            if (this.cfg.noiseSigma > 0) {
                let u = 0, v = 0;
                while(u === 0) u = Math.random();
                while(v === 0) v = Math.random();
                const noise = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
                gj += this.cfg.noiseSigma * noise;
            }
            
            gradOut[j] = gj;
        }
    }
    
    getMinDres(w: Float64Array): number {
        let minDist = Infinity;
        for (const kvec of this.Kres) {
            let dot = 0;
            for(let j=0; j<this.k; j++) {
                dot += kvec[j] * w[j];
            }
            const absDot = Math.abs(dot);
            if (absDot < minDist) minDist = absDot;
        }
        return minDist;
    }
    
    getDrift(w: Float64Array, w0: Float64Array) {
        let s = 0;
        for(let j=0; j<this.k; j++) {
            const d = w[j] - w0[j];
            s += d*d;
        }
        return Math.sqrt(s);
    }
    
    runBurst(steps: number): { maxGradA: number, maxGradB: number, driftA: number, driftB: number } {
        let maxGradA = 0;
        let maxGradB = 0;

        for (let s = 0; s < steps; s++) {
            this.step++;
            const t = this.step;
            
            this.getGradient(this.wA, null, this.gradA_scratch);
            let gnormA = 0;
            for(let j=0; j<this.k; j++){
                const gj = this.gradA_scratch[j];
                gnormA += gj*gj;
                this.mA[j] = this.cfg.beta1 * this.mA[j] + (1 - this.cfg.beta1) * gj;
                this.vA[j] = this.beta2 * this.vA[j] + (1 - this.beta2) * (gj * gj);
                const mh = this.mA[j] / (1 - Math.pow(this.cfg.beta1, t));
                const vh = this.vA[j] / (1 - Math.pow(this.beta2, t));
                this.wA[j] = this.wA[j] - (this.cfg.alpha * mh) / (Math.sqrt(vh) + 1e-8);
            }
            gnormA = Math.sqrt(gnormA);
            if(gnormA > maxGradA) maxGradA = gnormA;
            
            this.getGradient(this.wB, null, this.gradB_scratch);
            let gnormB = 0;
            for(let j=0; j<this.k; j++){
                const gj = this.gradB_scratch[j];
                gnormB += gj*gj;
                this.mB[j] = this.cfg.beta1 * this.mB[j] + (1 - this.cfg.beta1) * gj;
                this.vB[j] = this.beta2 * this.vB[j] + (1 - this.beta2) * (gj * gj);
                const mh = this.mB[j] / (1 - Math.pow(this.cfg.beta1, t));
                const vh = this.vB[j] / (1 - Math.pow(this.beta2, t));
                this.wB[j] = this.wB[j] - (this.cfg.alpha * mh) / (Math.sqrt(vh) + 1e-8);
            }
            gnormB = Math.sqrt(gnormB);
            if(gnormB > maxGradB) maxGradB = gnormB;
        }

        return {
            maxGradA,
            maxGradB,
            driftA: this.getDrift(this.wA, this.wA0),
            driftB: this.getDrift(this.wB, this.wB0)
        };
    }
    
    stepForward(steps: number): SimulationResult {
        const metrics: SimulationMetrics[] = [];
        
        for(let s=0; s<steps; s++) {
            this.step++;
            const t = this.step;
            
            // Trial A update
            this.getGradient(this.wA, this.ahA_buffer, this.gradA_scratch);
            let gnormA = 0;
            for(let j=0; j<this.k; j++){
                const gj = this.gradA_scratch[j];
                gnormA += gj*gj;
                this.mA[j] = this.cfg.beta1 * this.mA[j] + (1 - this.cfg.beta1) * gj;
                this.vA[j] = this.beta2 * this.vA[j] + (1 - this.beta2) * (gj * gj);
                const mh = this.mA[j] / (1 - Math.pow(this.cfg.beta1, t));
                const vh = this.vA[j] / (1 - Math.pow(this.beta2, t));
                this.wA[j] = this.wA[j] - (this.cfg.alpha * mh) / (Math.sqrt(vh) + 1e-8);
            }
            gnormA = Math.sqrt(gnormA);
            
            // Trial B update
            this.getGradient(this.wB, this.ahB_buffer, this.gradB_scratch);
            let gnormB = 0;
            for(let j=0; j<this.k; j++){
                const gj = this.gradB_scratch[j];
                gnormB += gj*gj;
                this.mB[j] = this.cfg.beta1 * this.mB[j] + (1 - this.cfg.beta1) * gj;
                this.vB[j] = this.beta2 * this.vB[j] + (1 - this.beta2) * (gj * gj);
                const mh = this.mB[j] / (1 - Math.pow(this.cfg.beta1, t));
                const vh = this.vB[j] / (1 - Math.pow(this.beta2, t));
                this.wB[j] = this.wB[j] - (this.cfg.alpha * mh) / (Math.sqrt(vh) + 1e-8);
            }
            gnormB = Math.sqrt(gnormB);
            
            const driftA = this.getDrift(this.wA, this.wA0);
            const driftB = this.getDrift(this.wB, this.wB0);
            const resA = this.getMinDres(this.wA);
            const resB = this.getMinDres(this.wB);
            
            const isSubsampled = (this.step % 10 === 0) || (this.step === 5000);
            
            if (isSubsampled) {
                metrics.push({
                    step: this.step,
                    // clamp to avoid 0s in log scale
                    gnormA: Math.max(gnormA, 1e-10),
                    gnormB: Math.max(gnormB, 1e-10),
                    driftA, driftB,
                    resA: Math.max(resA, 1e-6),
                    resB: Math.max(resB, 1e-6),
                    wA1: this.wA[0],
                    mA1: this.mA[0],
                    wB1: this.wB[0],
                    mB1: this.mB[0]
                });
            }
        }
        
        return {
            metrics,
            step: this.step,
            done: this.step >= 5000,
            AhA: this.ahA_buffer,
            AhB: this.ahB_buffer
        };
    }
}
