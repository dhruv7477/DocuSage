import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import http from 'http'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  root: 'src',
  build: {
    outDir: '../dist'
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  },
  server: {
    port: 3000,
    strictPort: true,
    proxy: {
      '/api/v1': {
        // 127.0.0.1:8010 — NOT localhost:8000. The host dev API lives on
        // 8010 so it can never collide with the Docker stack's api
        // container (which publishes 8000); explicit IPv4 sidesteps the
        // wslrelay [::1] listeners Docker Desktop adds for its ports.
        target: 'http://127.0.0.1:8010',
        changeOrigin: true,
        // Node 20's default agent keeps idle sockets alive ~5s; uvicorn
        // ALSO closes idle keep-alive connections at 5s. With the
        // documents panel polling every 4s the two race, and the proxy
        // occasionally reuses a socket the backend just closed →
        // intermittent "socket hang up" errors. A fresh connection per
        // request costs microseconds on localhost and ends the race.
        agent: new http.Agent({ keepAlive: false }),
        // While the backend is warming up (~30s of model loads) or
        // restarting, answer with a clean 503 instead of spamming the
        // terminal with stack traces and hanging the request.
        configure: (proxy) => {
          proxy.on('error', (err, _req, res) => {
            const r = res as import('http').ServerResponse
            if (r && typeof r.writeHead === 'function' && !r.headersSent) {
              r.writeHead(503, { 'Content-Type': 'application/json' })
              r.end(JSON.stringify({
                error_code: 'BACKEND_UNAVAILABLE',
                message: 'Backend is starting up — give it a few seconds and retry.',
              }))
            }
          })
        },
      }
    }
  }
})