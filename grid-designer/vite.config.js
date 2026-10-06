import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // RELATIVE asset URLs, so one build works wherever it is served: at the dev
  // server's root, and under GitHub Pages at /dc-dev/grid-designer/. An
  // absolute base would hard-code one of the two. The app has no client-side
  // routing, which is the case where a relative base would break.
  base: './',
  server: {
    port: 5175,
    strictPort: true,
  },
})
