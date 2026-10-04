/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        clinical: {
          dark: '#0a0d14',
          panel: '#111622',
          card: '#161d2e',
          border: '#1f293d',
          hr: '#ef4444',     // Heart Rate Red
          spo2: '#06b6d4',   // SpO2 Cyan
          bp: '#3b82f6',     // Blood Pressure Blue
          rr: '#10b981',     // Respiratory Rate Green
          riskGreen: '#22c55e',
          riskYellow: '#eab308',
          riskOrange: '#f97316',
          riskRed: '#ef4444',
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Menlo', 'Consolas', 'monospace'],
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'ping-slow': 'ping 2s cubic-bezier(0, 0, 0.2, 1) infinite',
      }
    },
  },
  plugins: [],
}
