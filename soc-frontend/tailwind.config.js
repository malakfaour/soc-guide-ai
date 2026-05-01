/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        threat: { 50: '#fff1f1', 500: '#ef4444', 700: '#b91c1c', 900: '#450a0a' },
        safe: { 50: '#f0fdf4', 500: '#22c55e', 700: '#15803d', 900: '#052e16' },
        warn: { 50: '#fefce8', 500: '#eab308', 700: '#a16207', 900: '#422006' },
        surface: {
          50: '#f8fafc', 100: '#f1f5f9', 200: '#e2e8f0',
          700: '#1e2535', 800: '#141b2d', 850: '#0f1629', 900: '#0a0f1e', 950: '#060b14'
        }
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', '"Fira Code"', 'Consolas', 'monospace'],
        display: ['"Space Grotesk"', 'sans-serif'],
        body: ['"DM Sans"', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-in': 'slideIn 0.3s ease-out',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        fadeIn: { from: { opacity: '0', transform: 'translateY(4px)' }, to: { opacity: '1', transform: 'translateY(0)' } },
        slideIn: { from: { opacity: '0', transform: 'translateX(20px)' }, to: { opacity: '1', transform: 'translateX(0)' } },
        glow: { from: { boxShadow: '0 0 5px rgba(239,68,68,0.3)' }, to: { boxShadow: '0 0 20px rgba(239,68,68,0.6)' } },
      }
    }
  },
  plugins: []
}
