/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // ── Holographic-HUD palette ──────────────────────────────
        // Token names are kept stable so existing className usage keeps
        // working; only the values shift to a deep-space cyan HUD.
        void:    '#04070f',
        surface: '#070d1a',
        panel:   '#0a1322',
        border:  '#16263f',
        dim:     '#46597a',
        ghost:   '#5b7099',
        subtle:  '#8ba3c7',
        bright:  '#e3f1ff',
        teal:    '#1ff0e6',   // primary HUD accent (electric cyan)
        plasma:  '#9d6bff',   // secondary accent (alien violet)
        amber:   '#ffb020',
        green:   '#3ef0a0',
        red:     '#ff476f',
      },
      fontFamily: {
        hud:     ['Orbitron', 'JetBrains Mono', 'monospace'],
        display: ['Inter', 'system-ui', 'sans-serif'],
        mono:    ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.65rem', { lineHeight: '1rem' }],
      },
      boxShadow: {
        panel: '0 0 0 1px rgba(31,240,230,0.08), 0 4px 28px rgba(0,0,0,0.55)',
        teal:  '0 0 18px rgba(31,240,230,0.45), 0 0 4px rgba(31,240,230,0.6)',
        hud:   '0 0 0 1px rgba(31,240,230,0.12), inset 0 0 24px rgba(31,240,230,0.04), 0 8px 40px rgba(0,0,0,0.6)',
        plasma: '0 0 18px rgba(157,107,255,0.45)',
      },
      backgroundImage: {
        grid: "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='40' height='40'%3E%3Cpath d='M0 0h40v40H0z' fill='none'/%3E%3Cpath d='M40 0H0v1h40zm0 40H0v-1h40zM0 0v40h1V0zm39 0v40h1V0z' fill='rgba(31,240,230,0.05)'/%3E%3C/svg%3E\")",
        scanlines: "repeating-linear-gradient(to bottom, rgba(31,240,230,0.035) 0px, rgba(31,240,230,0.035) 1px, transparent 1px, transparent 3px)",
      },
      keyframes: {
        'fade-up': {
          '0%':   { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'pulse-teal': {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(31,240,230,0.4)' },
          '50%':       { boxShadow: '0 0 0 6px rgba(31,240,230,0)' },
        },
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%':      { opacity: '0' },
        },
        'slide-in-right': {
          '0%':   { transform: 'translateX(100%)' },
          '100%': { transform: 'translateX(0)' },
        },
        aurora: {
          '0%, 100%': { transform: 'translate(0, 0) scale(1)',       opacity: '0.5' },
          '33%':      { transform: 'translate(4%, -6%) scale(1.15)', opacity: '0.8' },
          '66%':      { transform: 'translate(-5%, 4%) scale(0.95)', opacity: '0.6' },
        },
        shimmer: {
          '0%':   { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' },
        },
        // ── HUD-specific motion ──────────────────────────────────
        scanline: {
          '0%':   { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100vh)' },
        },
        'grid-scroll': {
          '0%':   { backgroundPosition: '0 0' },
          '100%': { backgroundPosition: '0 -40px' },
        },
        'reticle-spin': {
          '0%':   { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' },
        },
        'glow-pulse': {
          '0%, 100%': { opacity: '0.55' },
          '50%':      { opacity: '1' },
        },
        flicker: {
          '0%, 19%, 21%, 23%, 80%, 100%': { opacity: '1' },
          '20%, 22%, 81%':                { opacity: '0.72' },
        },
      },
      animation: {
        'fade-up':         'fade-up 0.35s ease-out both',
        'pulse-teal':      'pulse-teal 2s ease-in-out infinite',
        blink:             'blink 1s step-end infinite',
        'slide-in-right':  'slide-in-right 0.25s ease-out both',
        aurora:            'aurora 18s ease-in-out infinite',
        'aurora-slow':     'aurora 26s ease-in-out infinite reverse',
        shimmer:           'shimmer 1.6s ease-in-out infinite',
        scanline:          'scanline 7s linear infinite',
        'grid-scroll':     'grid-scroll 3s linear infinite',
        'reticle-spin':    'reticle-spin 24s linear infinite',
        'glow-pulse':      'glow-pulse 3.5s ease-in-out infinite',
        flicker:           'flicker 4s linear infinite',
      },
    },
  },
  plugins: [],
}
