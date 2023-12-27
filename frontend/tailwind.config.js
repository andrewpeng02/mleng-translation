/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      keyframes: {
        pulsate: {
          from: {
            borderColor: '#94a3b8', // slate-400
            backgroundColor: '#e2e8f0' // slate-200
          },
          to: {
            borderColor: '#1e293b', // slate-800
            backgroundColor: '#BDC9D7' 
          }
        },
      },
      animation: {
        'loading-pulsate': 'pulsate 1s linear infinite alternate',
      },
    },
  },
  plugins: [],
}