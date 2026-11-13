import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// In dev, the FastAPI server (`rag-lab serve`) runs on :8000 and Vite proxies /api to it.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
})
