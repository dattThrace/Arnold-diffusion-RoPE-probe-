'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis
} from 'recharts';
import { Play, RotateCcw, Activity, Volume2, VolumeX, Settings, Cpu } from 'lucide-react';
import { ArnoldSimulation, SimulationConfig, SimulationResult, SimulationMetrics } from '@/lib/engine';
import { ArnoldAudioEngine } from '@/lib/audio';

function AttentionHeatmap({ 
    data, L, title 
}: { 
    data: Float64Array | null, L: number, title: string 
}) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    
    useEffect(() => {
        if (!canvasRef.current || !data) return;
        const ctx = canvasRef.current.getContext('2d');
        if (!ctx) return;
        
        ctx.clearRect(0, 0, L, L);
        const imgData = ctx.createImageData(L, L);
        for(let i=0; i<L * L; i++) {
            const val = Math.floor(Math.min(Math.max(data[i] * 255 * 3, 0), 255)); 
            imgData.data[i*4] = val;     // R
            imgData.data[i*4+1] = val;   // G
            imgData.data[i*4+2] = val;   // B
            imgData.data[i*4+3] = 255;   // A
        }
        ctx.putImageData(imgData, 0, 0);
    }, [data, L]);

    return (
        <div className="flex flex-col items-center space-y-2">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-widest">{title}</span>
            <div className="border border-slate-200 rounded overflow-hidden aspect-square w-full max-w-[200px] bg-slate-50 flex items-center justify-center">
                {!data ? (
                    <span className="text-xs text-slate-400">No Data</span>
                ) : (
                    <canvas 
                        ref={canvasRef} 
                        width={L} 
                        height={L} 
                        className="w-full h-full object-contain [image-rendering:pixelated]"
                    />
                )}
            </div>
        </div>
    );
}

export default function ArnoldDiffusionSandbox() {
    // Config state
    const [cfg, setCfg] = useState<SimulationConfig>({
        d: 4,
        L: 64,
        tau: 1.0,
        alpha: 0.01,
        beta1: 0.9,
    });
    
    const [listeningMode, setListeningMode] = useState<'none' | 'A' | 'B'>('none');
    
    // Engine refs
    const simRef = useRef<ArnoldSimulation | null>(null);
    const audioRef = useRef<ArnoldAudioEngine | null>(null);
    const animRef = useRef<number>(0);
    
    // UI state
    const [isRunning, setIsRunning] = useState(false);
    const [points, setPoints] = useState<SimulationMetrics[]>([]);
    const [heatmaps, setHeatmaps] = useState<{AhA: Float64Array | null, AhB: Float64Array | null}>({AhA: null, AhB: null});
    const [progress, setProgress] = useState(0);

    useEffect(() => {
        audioRef.current = new ArnoldAudioEngine();
        return () => {
            audioRef.current?.destroy();
            if(animRef.current) cancelAnimationFrame(animRef.current);
        };
    }, []);

    useEffect(() => {
        if(audioRef.current) {
            audioRef.current.listeningMode = listeningMode;
            if (listeningMode !== 'none') {
                audioRef.current.initialize();
                audioRef.current.resume();
            }
        }
    }, [listeningMode]);

    const runChunk = useCallback(function chunkStep() {
        if (!simRef.current) return;
        const result = simRef.current.stepForward(20);
        
        setPoints(prev => {
            const next = [...prev, ...result.metrics];
            return next;
        });
        setProgress(result.step / 5000);
        setHeatmaps({ AhA: result.AhA, AhB: result.AhB });
        
        if (audioRef.current && result.metrics.length > 0) {
            const last = result.metrics[result.metrics.length - 1];
            // Provide data depending on what we are listening to
            if (listeningMode === 'A') {
                audioRef.current.update(simRef.current.k, simRef.current.wA, simRef.current.mA, last.gnormA);
            } else if (listeningMode === 'B') {
                audioRef.current.update(simRef.current.k, simRef.current.wB, simRef.current.mB, last.gnormB);
            }
        }
        
        if (!result.done) {
            animRef.current = requestAnimationFrame(chunkStep);
        } else {
            setIsRunning(false);
            if (audioRef.current) audioRef.current.suspend();
        }
    }, [listeningMode]);

    const startSim = () => {
        if(animRef.current) cancelAnimationFrame(animRef.current);
        setPoints([]);
        setProgress(0);
        setHeatmaps({ AhA: null, AhB: null });
        simRef.current = new ArnoldSimulation(cfg);
        setIsRunning(true);
        if(audioRef.current && listeningMode !== 'none') {
            audioRef.current.initialize();
            audioRef.current.resume();
        }
        animRef.current = requestAnimationFrame(runChunk);
    };

    const stopSim = () => {
        setIsRunning(false);
        if (animRef.current) cancelAnimationFrame(animRef.current);
        if (audioRef.current) audioRef.current.suspend();
    };

    return (
        <div className="min-h-screen bg-slate-900 text-slate-100 font-sans p-6 md:p-8 flex flex-col xl:flex-row gap-8">
            {/* Sidebar Controls */}
            <aside className="w-full xl:w-80 flex-shrink-0 space-y-8">
                <div className="space-y-2">
                    <h1 className="text-2xl font-semibold tracking-tight text-white flex items-center gap-2">
                        <Activity className="w-6 h-6 text-orange-500" />
                        Arnold Diffusion
                    </h1>
                    <p className="text-slate-400 text-sm leading-relaxed">
                        RoPE Phase Space Dynamics mapping AdamW momentum scatter along resonant invariant tori.
                    </p>
                </div>

                <div className="bg-slate-800 rounded-xl p-5 border border-slate-700 space-y-6">
                    <h2 className="text-sm font-semibold flex items-center gap-2 text-slate-200">
                        <Settings className="w-4 h-4 text-slate-400" />
                        Manifold Parameters
                    </h2>
                    
                    <div className="space-y-4">
                        <div className="space-y-1">
                            <label className="text-xs font-medium text-slate-400 flex justify-between">
                                <span>Topology (d)</span>
                                <span className="text-slate-200">{cfg.d}</span>
                            </label>
                            <input 
                                type="range" min="2" max="16" step="2" value={cfg.d} 
                                onChange={e => setCfg({...cfg, d: +e.target.value})}
                                disabled={isRunning} className="w-full accent-orange-500"
                            />
                        </div>
                        <div className="space-y-1">
                            <label className="text-xs font-medium text-slate-400 flex justify-between">
                                <span>Sequence Length (L)</span>
                                <span className="text-slate-200">{cfg.L}</span>
                            </label>
                            <input 
                                type="range" min="16" max="128" step="16" value={cfg.L} 
                                onChange={e => setCfg({...cfg, L: +e.target.value})}
                                disabled={isRunning} className="w-full accent-orange-500"
                            />
                        </div>
                        <div className="space-y-1">
                            <label className="text-xs font-medium text-slate-400 flex justify-between">
                                <span>Temperature (τ)</span>
                                <span className="text-slate-200">{cfg.tau.toFixed(1)}</span>
                            </label>
                            <input 
                                type="range" min="0.1" max="5.0" step="0.1" value={cfg.tau} 
                                onChange={e => setCfg({...cfg, tau: +e.target.value})}
                                disabled={isRunning} className="w-full accent-orange-500"
                            />
                        </div>
                        <div className="space-y-1">
                            <label className="text-xs font-medium text-slate-400 flex justify-between">
                                <span>Optimizer Pacing (α)</span>
                                <span className="text-slate-200">{cfg.alpha.toFixed(3)}</span>
                            </label>
                            <input 
                                type="range" min="0.001" max="0.1" step="0.001" value={cfg.alpha} 
                                onChange={e => setCfg({...cfg, alpha: +e.target.value})}
                                disabled={isRunning} className="w-full accent-orange-500"
                            />
                        </div>
                    </div>
                </div>

                <div className="bg-slate-800 rounded-xl p-5 border border-slate-700 space-y-6">
                    <h2 className="text-sm font-semibold flex items-center gap-2 text-slate-200">
                        <Volume2 className="w-4 h-4 text-slate-400" />
                        Sonification Engine
                    </h2>
                    <div className="flex gap-2">
                        {['none', 'A', 'B'].map((mode) => (
                            <button
                                key={mode}
                                onClick={() => setListeningMode(mode as any)}
                                className={`flex-1 py-1.5 px-3 rounded-md border text-xs font-medium transition-colors ${
                                    listeningMode === mode 
                                    ? 'bg-orange-500/20 border-orange-500 text-orange-400' 
                                    : 'bg-slate-900 border-slate-700 text-slate-400 hover:border-slate-500'
                                }`}
                            >
                                {mode === 'none' ? 'Mute' : `Trial ${mode}`}
                            </button>
                        ))}
                    </div>
                </div>

                <div className="space-y-3">
                    <button
                        onClick={isRunning ? stopSim : startSim}
                        className={`w-full py-3 px-4 rounded-xl font-medium tracking-wide flex items-center justify-center gap-2 transition-all ${
                            isRunning 
                                ? 'bg-rose-500/10 text-rose-500 hover:bg-rose-500/20' 
                                : 'bg-slate-100 text-slate-900 hover:bg-white'
                        }`}
                    >
                        {isRunning ? <RotateCcw className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                        <span>{isRunning ? 'Halt Execution' : 'Ignite Optimizer'}</span>
                    </button>
                    {isRunning && (
                        <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                            <div className="h-full bg-orange-500 transition-all duration-300" style={{width: `${progress * 100}%`}} />
                        </div>
                    )}
                </div>
            </aside>

            {/* Dashboard grid */}
            <main className="flex-1 min-w-0 flex flex-col gap-6">
                
                {/* Visual Grid 1: Explosions & Distance */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Gradient Explosion */}
                    <div className="bg-slate-800 rounded-xl p-5 border border-slate-700 min-h-[300px] flex flex-col">
                        <div className="mb-4">
                            <h3 className="text-sm font-semibold text-slate-200">The Explosion Tracker</h3>
                            <p className="text-xs text-slate-500">L2 Gradient Norm ||∇ω||₂ (Log Scale)</p>
                        </div>
                        <div className="flex-1 min-h-[220px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={points} margin={{top: 5, right: 5, left: 5, bottom: 5}}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                                    <XAxis dataKey="step" stroke="#64748b" tick={{fill: '#64748b', fontSize: 10}} tickLine={false} axisLine={false} />
                                    <YAxis scale="log" domain={['auto', 'auto']} stroke="#64748b" tick={{fill: '#64748b', fontSize: 10}} tickLine={false} axisLine={false} width={40} tickFormatter={v => v.toExponential(0)}/>
                                    <Tooltip contentStyle={{backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px'}} labelStyle={{color: '#94a3b8'}} itemStyle={{fontSize: '12px'}} formatter={(v: any) => typeof v === 'number' ? v.toExponential(3) : v} />
                                    <Line type="monotone" name="A (Geometric)" dataKey="gnormA" stroke="#f43f5e" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                                    <Line type="monotone" name="B (Algebraic)" dataKey="gnormB" stroke="#3b82f6" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Parameter Drift */}
                    <div className="bg-slate-800 rounded-xl p-5 border border-slate-700 min-h-[300px] flex flex-col">
                        <div className="mb-4">
                            <h3 className="text-sm font-semibold text-slate-200">Parameter Drift</h3>
                            <p className="text-xs text-slate-500">L2 Distance from Initialization ||ω_t - ω_0||₂</p>
                        </div>
                        <div className="flex-1 min-h-[220px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={points} margin={{top: 5, right: 5, left: 5, bottom: 5}}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                                    <XAxis dataKey="step" stroke="#64748b" tick={{fill: '#64748b', fontSize: 10}} tickLine={false} axisLine={false} />
                                    <YAxis stroke="#64748b" tick={{fill: '#64748b', fontSize: 10}} tickLine={false} axisLine={false} width={30}/>
                                    <Tooltip contentStyle={{backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px'}} labelStyle={{color: '#94a3b8'}} itemStyle={{fontSize: '12px'}} formatter={(v: any) => typeof v === 'number' ? v.toFixed(4) : v} />
                                    <Line type="monotone" name="A Drift" dataKey="driftA" stroke="#f43f5e" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                                    <Line type="monotone" name="B Drift" dataKey="driftB" stroke="#3b82f6" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>

                {/* Visual Grid 2: Phase Space & Resonance Proximity */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Phase Space Scatter */}
                    <div className="bg-slate-800 rounded-xl p-5 border border-slate-700 min-h-[300px] flex flex-col">
                        <div className="mb-4 flex items-center justify-between">
                            <div>
                                <h3 className="text-sm font-semibold text-slate-200">Phase Space Trace</h3>
                                <p className="text-xs text-slate-500">ω₁ vs m₁ (Parameter & Momentum Coordinate)</p>
                            </div>
                            <Cpu className="w-5 h-5 text-slate-500" />
                        </div>
                        <div className="flex-1 min-h-[220px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <ScatterChart margin={{top: 10, right: 10, left: -20, bottom: -10}}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                    <XAxis type="number" dataKey="wA1" name="ω₁" domain={['auto', 'auto']} tick={{fontSize: 10, fill: '#64748b'}} />
                                    <YAxis type="number" dataKey="mA1" name="m₁" domain={['auto', 'auto']} tick={{fontSize: 10, fill: '#64748b'}} />
                                    <ZAxis type="number" range={[10, 10]} />
                                    <Tooltip cursor={{strokeDasharray: '3 3'}} contentStyle={{backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px'}} formatter={(v: any) => typeof v === 'number' ? v.toExponential(2) : v} />
                                    <Scatter name="Trial A" data={points} fill="#f43f5e" shape="circle" fillOpacity={0.6} isAnimationActive={false} />
                                </ScatterChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Resonance Proximity */}
                    <div className="bg-slate-800 rounded-xl p-5 border border-slate-700 min-h-[300px] flex flex-col">
                        <div className="mb-4 flex items-center justify-between">
                            <div>
                                <h3 className="text-sm font-semibold text-slate-200">Resonance Proximity Metric</h3>
                                <p className="text-xs text-slate-500">min |⟨k, ω⟩| for low-order integers</p>
                            </div>
                        </div>
                        <div className="flex-1 min-h-[220px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={points} margin={{top: 5, right: 5, left: 5, bottom: 5}}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                                    <XAxis dataKey="step" stroke="#64748b" tick={{fill: '#64748b', fontSize: 10}} tickLine={false} axisLine={false} />
                                    <YAxis scale="log" domain={['auto', 'auto']} stroke="#64748b" tick={{fill: '#64748b', fontSize: 10}} tickLine={false} axisLine={false} width={40} tickFormatter={v => v.toExponential(0)}/>
                                    <Tooltip contentStyle={{backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px'}} labelStyle={{color: '#94a3b8'}} itemStyle={{fontSize: '12px'}} formatter={(v: any) => typeof v === 'number' ? v.toExponential(3) : v} />
                                    <Line type="step" name="A Divisor" dataKey="resA" stroke="#f43f5e" strokeWidth={1} dot={false} isAnimationActive={false} opacity={0.8} />
                                    <Line type="step" name="B Divisor" dataKey="resB" stroke="#3b82f6" strokeWidth={1} dot={false} isAnimationActive={false} opacity={0.8} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>

                {/* Visual Row 3: Heatmaps */}
                <div className="bg-slate-800 rounded-xl p-5 border border-slate-700 flex flex-col md:flex-row justify-around items-center py-8">
                    <AttentionHeatmap data={heatmaps.AhA} L={cfg.L} title="Trial A (Geometric Base)" />
                    <div className="w-px h-32 bg-slate-700 hidden md:block" />
                    <AttentionHeatmap data={heatmaps.AhB} L={cfg.L} title="Trial B (Algebraic Base)" />
                </div>
            </main>
        </div>
    );
}
