'use client';

import React, { useState, useRef, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import { Activity, Play, RotateCcw } from 'lucide-react';
import Link from 'next/link';
import { ArnoldSimulation } from '@/lib/engine';

interface SweepPoint {
    L: number;
    maxGradA: number;
    maxGradB: number;
    driftA: number;
    driftB: number;
}

export default function SweepPage() {
    const sweepLengths = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512];
    const [points, setPoints] = useState<SweepPoint[]>([]);
    const [isRunning, setIsRunning] = useState(false);
    const [currentL, setCurrentL] = useState<number | null>(null);
    const cancelRef = useRef(false);

    const runSweep = async () => {
        setIsRunning(true);
        setPoints([]);
        cancelRef.current = false;

        for (const L of sweepLengths) {
            if (cancelRef.current) break;
            setCurrentL(L);
            
            // Allow React to render state and keep UI responsive
            await new Promise(r => setTimeout(r, 50));
            
            const sim = new ArnoldSimulation({
                d: 4,
                L: L,
                tau: 5.0,
                alpha: 0.05,
                beta1: 0.9,
                noiseSigma: 0.024
            });

            // Run 1000 steps without waiting
            const burst = sim.runBurst(1000);
            const pt: SweepPoint = {
                L: L,
                maxGradA: burst.maxGradA,
                maxGradB: burst.maxGradB,
                driftA: burst.driftA,
                driftB: burst.driftB
            };

            setPoints(prev => [...prev, pt]);
        }

        setIsRunning(false);
        setCurrentL(null);
    };

    const stopSweep = () => {
        cancelRef.current = true;
    };

    return (
        <div className="min-h-screen bg-slate-900 text-slate-100 font-sans p-6 md:p-8 flex flex-col items-center">
            <div className="w-full max-w-6xl space-y-8">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
                    <div className="space-y-2">
                        <h1 className="text-2xl font-semibold tracking-tight text-white flex items-center gap-2">
                            <Activity className="w-6 h-6 text-orange-500" />
                            Context Length Sweep
                        </h1>
                        <p className="text-slate-400 text-sm leading-relaxed max-w-2xl">
                            Analyzing topological stability of AdamW manifold against increasing sequence length (L). Fixed $d=4$, 1000 steps per context length.
                        </p>
                    </div>
                    
                    <div className="flex bg-slate-800 rounded-lg p-1 border border-slate-700">
                        <Link href="/" className="px-4 py-2 text-sm font-medium rounded-md text-slate-400 hover:text-slate-200 transition-colors">
                            Diagnostics
                        </Link>
                        <span className="px-4 py-2 text-sm font-medium rounded-md bg-slate-700 text-white shadow-sm">
                            Context Sweep
                        </span>
                    </div>
                </div>

                <div className="flex gap-4">
                    <button
                        onClick={isRunning ? stopSweep : runSweep}
                        className={`py-2 px-6 rounded-md text-sm font-bold transition-colors flex items-center gap-2 
                            ${isRunning ? 'bg-red-500 hover:bg-red-600 text-white shadow-lg shadow-red-500/20' 
                                        : 'bg-orange-500 hover:bg-orange-600 text-white shadow-lg shadow-orange-500/20'}`}
                    >
                        {isRunning ? <RotateCcw className="w-4 h-4" /> : <Play className="w-4 h-4 ml-1" />}
                        {isRunning ? 'Stop Sweep' : 'Run Context Sweep'}
                    </button>
                    {isRunning && currentL !== null && (
                        <div className="flex items-center text-sm font-medium text-slate-300 bg-slate-800 px-4 py-2 rounded-md border border-slate-700">
                            Processing L = {currentL}...
                        </div>
                    )}
                </div>

                {/* Charts */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="bg-slate-800 rounded-xl p-5 border border-slate-700 space-y-4">
                        <h3 className="text-sm font-semibold text-slate-200 tracking-wide">
                            Arnold Web Density <span className="text-slate-500 font-normal ml-2">Max Gradient Shock vs. Length (Log Scale)</span>
                        </h3>
                        <div className="h-80 w-full bg-slate-900 rounded border border-slate-700/50 p-2 relative">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={points} margin={{top: 10, right: 10, left: 0, bottom: 0}}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                    <XAxis dataKey="L" type="number" stroke="#64748b" tick={{fill: '#64748b', fontSize: 10}} tickLine={false} axisLine={false} domain={['dataMin', 'dataMax']} />
                                    <YAxis scale="log" domain={['auto', 'auto']} stroke="#64748b" tick={{fill: '#64748b', fontSize: 10}} tickLine={false} axisLine={false} width={40} tickFormatter={v => v.toExponential(0)}/>
                                    <Tooltip contentStyle={{backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px'}} labelStyle={{color: '#94a3b8'}} itemStyle={{fontSize: '12px'}} formatter={(v: any) => typeof v === 'number' ? v.toExponential(3) : v} labelFormatter={(v) => `L = ${v}`} />
                                    <Legend verticalAlign="top" height={36} iconType="plainline" wrapperStyle={{fontSize: '12px'}} />
                                    <Line type="monotone" name="Trial A (Geometric)" dataKey="maxGradA" stroke="#f43f5e" strokeWidth={2} dot={{r: 4, strokeWidth: 2}} activeDot={{r: 6}} isAnimationActive={!!isRunning} />
                                    <Line type="monotone" name="Trial B (Algebraic)" dataKey="maxGradB" stroke="#3b82f6" strokeWidth={2} dot={{r: 4, strokeWidth: 2}} activeDot={{r: 6}} isAnimationActive={!!isRunning} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    <div className="bg-slate-800 rounded-xl p-5 border border-slate-700 space-y-4">
                        <h3 className="text-sm font-semibold text-slate-200 tracking-wide">
                            Topological Ejection <span className="text-slate-500 font-normal ml-2">Terminal Parameter Drift vs. Length</span>
                        </h3>
                        <div className="h-80 w-full bg-slate-900 rounded border border-slate-700/50 p-2 relative">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={points} margin={{top: 10, right: 10, left: 0, bottom: 0}}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                    <XAxis dataKey="L" type="number" stroke="#64748b" tick={{fill: '#64748b', fontSize: 10}} tickLine={false} axisLine={false} domain={['dataMin', 'dataMax']} />
                                    <YAxis type="number" domain={[0, 'auto']} stroke="#64748b" tick={{fill: '#64748b', fontSize: 10}} tickLine={false} axisLine={false} width={40} tickFormatter={v => v.toExponential(0)}/>
                                    <Tooltip contentStyle={{backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px'}} labelStyle={{color: '#94a3b8'}} itemStyle={{fontSize: '12px'}} formatter={(v: any) => typeof v === 'number' ? v.toExponential(3) : v} labelFormatter={(v) => `L = ${v}`} />
                                    <Legend verticalAlign="top" height={36} iconType="plainline" wrapperStyle={{fontSize: '12px'}} />
                                    <Line type="stepAfter" name="Trial A (Geometric)" dataKey="driftA" stroke="#f43f5e" strokeWidth={2} dot={{r: 4, strokeWidth: 2}} activeDot={{r: 6}} isAnimationActive={!!isRunning} />
                                    <Line type="stepAfter" name="Trial B (Algebraic)" dataKey="driftB" stroke="#3b82f6" strokeWidth={2} dot={{r: 4, strokeWidth: 2}} activeDot={{r: 6}} isAnimationActive={!!isRunning} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
