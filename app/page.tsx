import dynamic from 'next/dynamic';

const Sandbox = dynamic(() => import('./sandbox'), { 
    ssr: false,
    loading: () => (
        <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center p-6 space-y-4">
            <div className="w-8 h-8 border-4 border-slate-700 border-t-indigo-500 rounded-full animate-spin"></div>
            <p className="text-slate-400 text-sm font-mono tracking-widest uppercase">Initializing Simulation Engine...</p>
        </div>
    )
});

export default function Page() {
    return <Sandbox />;
}
