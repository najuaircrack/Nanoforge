import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': path.resolve(__dirname, './src') } },
  build: { outDir: '../src/nanoforge/web_dist', emptyOutDir: true },
  server: { proxy: { '/api': 'http://localhost:7860', '/stream': 'http://localhost:7860' } },
})
