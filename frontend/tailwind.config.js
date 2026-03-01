/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Biology-inspired colors
        bacteria: {
          light: '#86efac',
          DEFAULT: '#22c55e',
          dark: '#15803d',
        },
        virus: {
          light: '#fca5a5',
          DEFAULT: '#ef4444',
          dark: '#b91c1c',
        },
        eukaryotic: {
          light: '#93c5fd',
          DEFAULT: '#3b82f6',
          dark: '#1d4ed8',
        },
        // Nucleotide colors
        nucleotide: {
          A: '#22c55e',  // Adenine - green
          T: '#ef4444',  // Thymine - red
          G: '#f59e0b',  // Guanine - amber
          C: '#3b82f6',  // Cytosine - blue
          N: '#9ca3af',  // Unknown - gray
        },
      },
    },
  },
  plugins: [],
}
