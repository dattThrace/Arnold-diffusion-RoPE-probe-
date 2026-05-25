# High-Dimensional Arnold Diffusion Sandbox

An interactive, in-browser numerical experiment for exploring weak instabilities (Arnold diffusion) in nearly integrable Hamiltonian systems across high dimensions (D-dimensional tori). This sandbox implements a simplified pseudo-spectral integration scheme mixed with stochastic symplectic noise to map robust resonant zones geometrically and algebraically.

## Overview

The simulator models action-angle variables slowly tunneling through chaotic resonance layers—a phenomenon first rigorously proven bounded for $n \ge 3$ dimensions by V.I. Arnold (1964). 

Our custom "Topological Manifold Engine" operates entirely client-side using JavaScript typed arrays (`Float64Array`).
By leveraging aggressive time-slicing and a `MessageChannel` macrotask scheduler, the engine scales compute loops up to thousands of iterations per second without compromising the React UI thread's 60 FPS responsiveness rate.

### Features
*   **Geometric vs. Algebraic Base Optimizations**: Side-by-side benchmarking of momentum updates and stability drifts between different Diophantine approximations.
*   **Dimension Scaling ($L$ up to 1024)**: Run $O(L^2)$ matrix-reduction scale heatmaps dynamically.
*   **Time-Sliced Compute Architecture**: Harnesses `MessageChannel` for dynamic intra-chunk execution loops (~12ms budget), avoiding JS's aggressive `setTimeout` 4ms clamps.
*   **WebM Synthesis in Browser**: Generates high-quality encoded video recordings of the diffusion heat map utilizing headless `OffscreenCanvas` & `MediaRecorder`, completely free of server dependency.

## Development Setup

The application is built on Next.js 15+ (App Router) and TailwindCSS v4. 
It requires Node.js v18 or later.

1. **Install Dependencies**
   ```bash
   npm install
   ```
2. **Start the Development Server**
   ```bash
   npm run dev
   ```
   The application will boot at [http://localhost:3000](http://localhost:3000). 
   You can view the diffusion monitor directly from the index `/` path.

## Deployment Guide

This applet generates a fully static client-side build if required, but currently is configured via the standard Next.js full-stack container environment optimized for Google Cloud Run (and similar Docker hosts).

### Cloud Run (Standard Docker Deploy)

The easiest way to ship is via `gcloud` or any standard Dockerized hosting provider:

```bash
# 1. Build the production application
npm run build 

# 2. Start the production server
npm start 
```

**Dockerfile Example (for containerization):**
```dockerfile
FROM node:18-alpine AS runner
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
ENV NODE_ENV=production
ENV PORT=3000
CMD ["npm", "start"]
```

### Static Export (GitHub Pages / Vercel)
If you wish to deploy the sandbox strictly via CDN (as no backend API routes are required), you can modify `next.config.ts`:

```typescript
// next.config.ts
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export',
  // Required since next/image is used
  images: {
    unoptimized: true
  }
};

export default nextConfig;
```
Then simply run:
```bash
npm run build
```
Your static assets will be populated into the `/out` directory, which can be uploaded to GitHub Pages, Cloudflare Pages, S3, or Firebase Hosting.

## Code Map
* `app/page.tsx`: Core UI Dashboard and scheduler. Contains the `MessageChannel` tick loop for non-blocking compute.
* `lib/engine.ts`: The standalone Numerical Engine. Written in pure Mathematics using optimized nested Loops + Typed Arrays (Float64).
* `components/`: Contains isolated logic chunks like the responsive Heatmap `<canvas>` renderers or Charts.

## Scientific Disclaimer
This simulation uses abbreviated symplectic approximation mechanisms paired with noise, designed primarily for high-speed topological visualizations and geometric proofs of concept in JavaScript. Deep exploration of astronomical invariants requires strict Lie-series integrators not bundled in this web sandbox.
