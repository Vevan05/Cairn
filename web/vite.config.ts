import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/search': 'http://localhost:8000',
      '/answer': 'http://localhost:8000',
      '/events': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/experiments': 'http://localhost:8000',
      '/eval': 'http://localhost:8000',
    },
  },
})
