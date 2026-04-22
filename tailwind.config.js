/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        void:    '#09090f',
        surface: '#111118',
        panel:   '#16161f',
        border:  '#1e1e2e',
        dim:     '#4b5263',
        ghost:   '#363648',
        subtle:  '#7c7c96',
        bright:  '#e8e8f0',
        teal:    '#00d4c8',
        amber:   '#f59e0b',
        green:   '#22c55e',
        red:     '#ef4444',
      },
      fontFamily: {
        display: ['Inter', 'system-ui', 'sans-serif'],
        mono:    ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.65rem', { lineHeight: '1rem' }],
      },
      boxShadow: {
        panel: '0 0 0 1px rgba(0,212,200,0.06), 0 4px 24px rgba(0,0,0,0.4)',
      },
      backgroundImage: {
        grid: "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='32' height='32'%3E%3Cpath d='M0 0h32v32H0z' fill='none'/%3E%3Cpath d='M32 0H0v1h32zm0 32H0v-1h32zM0 0v32h1V0zm31 0v32h1V0z' fill='rgba(255,255,255,0.03)'/%3E%3C/svg%3E\")",
      },
      keyframes: {
        'fade-up': {
          '0%':   { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'pulse-teal': {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(0,212,200,0.4)' },
          '50%':       { boxShadow: '0 0 0 6px rgba(0,212,200,0)' },
        },
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%':      { opacity: '0' },
        },
      },
      animation: {
        'fade-up':    'fade-up 0.35s ease-out both',
        'pulse-teal': 'pulse-teal 2s ease-in-out infinite',
        blink:        'blink 1s step-end infinite',
      },
    },
  },
  plugins: [],
}
