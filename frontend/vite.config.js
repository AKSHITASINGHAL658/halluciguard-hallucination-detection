import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The browser talks to /api/*; Vite forwards it to FastAPI (no CORS setup needed).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.HALLUCIGUARD_API || "http://localhost:8000",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
        timeout: 1800000,
        proxyTimeout: 1800000,
      },
    },
  },
});
