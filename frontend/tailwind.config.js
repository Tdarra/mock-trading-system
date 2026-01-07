/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        terminal: {
          bg: '#0a0f14',
          panel: '#111920',
          border: '#1e2a36',
          accent: '#00d4aa',
          warning: '#fbbf24',
          danger: '#ef4444',
          muted: '#64748b',
          text: '#e2e8f0'
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        display: ['Space Grotesk', 'system-ui', 'sans-serif']
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate'
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(0, 212, 170, 0.2)' },
          '100%': { boxShadow: '0 0 20px rgba(0, 212, 170, 0.4)' }
        }
      }
    },
  },
  plugins: [],
}
