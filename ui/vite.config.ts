import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// In dev, run `sentinel dev` (port 2512) and `npm run dev`; the API is proxied.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:2512',
      '/ws': { target: 'ws://localhost:2512', ws: true },
    },
  },
  build: { outDir: 'dist', sourcemap: false },
});
