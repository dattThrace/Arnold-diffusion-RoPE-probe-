export class ArnoldAudioEngine {
    private ctx: AudioContext | null = null;
    private masterGain: GainNode | null = null;
    private noiseGain: GainNode | null = null;
    private oscillators: OscillatorNode[] = [];
    private torusGain: GainNode | null = null;
    
    private noiseBufferSource: AudioBufferSourceNode | null = null;
    private noiseFilter: BiquadFilterNode | null = null;
    
    // Which trial to listen to ('A' or 'B', or 'none')
    public listeningMode: 'none' | 'A' | 'B' = 'none';

    public initialize() {
        if (this.ctx) return;
        const CtxClass = window.AudioContext || (window as any).webkitAudioContext;
        if (!CtxClass) return;
        
        this.ctx = new CtxClass();
        
        this.masterGain = this.ctx.createGain();
        this.masterGain.gain.value = 0.5;
        this.masterGain.connect(this.ctx.destination);
        
        // Setup white noise
        const bufferSize = this.ctx.sampleRate * 2;
        const noiseBuffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
        const output = noiseBuffer.getChannelData(0);
        for (let i = 0; i < bufferSize; i++) {
            output[i] = Math.random() * 2 - 1;
        }
        
        this.noiseBufferSource = this.ctx.createBufferSource();
        this.noiseBufferSource.buffer = noiseBuffer;
        this.noiseBufferSource.loop = true;
        
        this.noiseFilter = this.ctx.createBiquadFilter();
        this.noiseFilter.type = 'bandpass';
        this.noiseFilter.frequency.value = 1200;
        
        this.noiseGain = this.ctx.createGain();
        this.noiseGain.gain.value = 0; // initially silent
        
        this.noiseBufferSource.connect(this.noiseFilter);
        this.noiseFilter.connect(this.noiseGain);
        this.noiseGain.connect(this.masterGain);
        
        this.noiseBufferSource.start();
    }
    
    public resume() {
        if (this.ctx && this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
    }
    
    public suspend() {
        if (this.ctx && this.ctx.state === 'running') {
            this.ctx.suspend();
        }
    }
    
    private setupTorus(k: number) {
        if (!this.ctx || !this.masterGain) return;
        
        // Clean up old ones
        for (const osc of this.oscillators) {
            osc.stop();
            osc.disconnect();
        }
        this.oscillators = [];
        if (this.torusGain) {
            this.torusGain.disconnect();
        }
        
        this.torusGain = this.ctx.createGain();
        this.torusGain.gain.value = 1.0 / k; // prevent clipping when combined
        this.torusGain.connect(this.masterGain);
        
        for (let i = 0; i < k; i++) {
            const osc = this.ctx.createOscillator();
            osc.type = 'sine';
            osc.connect(this.torusGain);
            osc.start();
            this.oscillators.push(osc);
        }
    }
    
    public update(k: number, w: Float64Array, m: Float64Array, gradNorm: number) {
        if (!this.ctx || this.listeningMode === 'none') {
            if (this.torusGain) this.torusGain.gain.setTargetAtTime(0, this.ctx?.currentTime || 0, 0.05);
            if (this.noiseGain) this.noiseGain.gain.setTargetAtTime(0, this.ctx?.currentTime || 0, 0.05);
            return;
        }
        
        if (this.oscillators.length !== k) {
            this.setupTorus(k);
        }
        
        if (this.torusGain) {
            this.torusGain.gain.setTargetAtTime(1.0 / k, this.ctx.currentTime, 0.05);
        }
        
        for (let i = 0; i < k; i++) {
            const f = 220 * (1 + Math.abs(w[i]));
            // Multiply m[i] by some scalar to make the drift noticeably detune the osc
            const wobble = m[i] * 50000; 
            if (isFinite(f) && isFinite(wobble)) {
                this.oscillators[i].frequency.setTargetAtTime(f, this.ctx.currentTime, 0.02);
                this.oscillators[i].detune.setTargetAtTime(wobble, this.ctx.currentTime, 0.02);
            }
        }
        
        if (this.noiseGain) {
            // Map grad norm to noise gain
            const noiseLevel = Math.min(Math.max((Math.log(1 + gradNorm) / 15.0), 0), 1.0);
            if (isFinite(noiseLevel)) {
                this.noiseGain.gain.setTargetAtTime(noiseLevel, this.ctx.currentTime, 0.02);
            }
        }
    }
    
    public destroy() {
        for (const osc of this.oscillators) {
            osc.stop();
            osc.disconnect();
        }
        if (this.noiseBufferSource) {
            this.noiseBufferSource.stop();
            this.noiseBufferSource.disconnect();
        }
        if (this.ctx) {
            this.ctx.close();
        }
    }
}
