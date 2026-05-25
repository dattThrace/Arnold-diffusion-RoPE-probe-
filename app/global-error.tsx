'use client';

import React, { useEffect } from 'react';

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('Global Error Boundary Caught:', error);
  }, [error]);

  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center justify-center p-6 font-sans">
        <div className="w-full max-w-md p-6 bg-slate-900 border border-slate-800 rounded-xl space-y-4 shadow-xl">
          <h2 className="text-lg font-bold text-red-400 flex items-center gap-2">
            ⚠️ Global Runtime Fault
          </h2>
          <div className="p-4 bg-slate-950 rounded border border-slate-800/80">
            <p className="text-xs font-mono text-slate-300 break-words whitespace-pre-wrap">
              {error?.message || 'A critical global environment loading error occurred.'}
            </p>
          </div>
          <button
            onClick={() => reset()}
            className="w-full py-2 bg-rose-500 hover:bg-rose-600 active:bg-rose-700 text-slate-100 rounded-md text-xs font-semibold"
          >
            Recover Applet
          </button>
        </div>
      </body>
    </html>
  );
}
