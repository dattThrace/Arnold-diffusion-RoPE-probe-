'use client';

import React, { useEffect } from 'react';

export default function ErrorBoundary({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('App Router Boundary Caught Error:', error);
  }, [error]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center justify-center p-6 font-sans">
      <div className="w-full max-w-md p-6 bg-slate-900 border border-slate-800 rounded-xl space-y-4 shadow-xl">
        <h2 className="text-lg font-bold text-red-400 flex items-center gap-2">
          ⚠️ Runtime Fault Detected
        </h2>
        <div className="p-4 bg-slate-950 rounded border border-slate-800/80">
          <p className="text-xs font-mono text-slate-300 break-words whitespace-pre-wrap">
            {error?.message || 'Unknown execution error'}
          </p>
          {error?.stack && (
            <p className="text-[10px] font-mono text-slate-500 mt-2 max-h-32 overflow-y-auto break-words">
              {error.stack}
            </p>
          )}
        </div>
        <p className="text-xs text-slate-400 leading-normal">
          Topological manifold simulation engine runtime recoverability mode.
        </p>
        <button
          onClick={() => reset()}
          className="w-full py-2 bg-rose-500 hover:bg-rose-600 active:bg-rose-700 text-slate-100 rounded-md text-xs font-semibold transition-colors"
        >
          Reset Engine State
        </button>
      </div>
    </div>
  );
}
