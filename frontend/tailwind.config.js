/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      colors: {
        lab: {
          bg: "#f4f6fb",
          surface: "#f9fafb",
          card: "#ffffff",
          line: "#e5e7eb",
          ink: "#111827",
          muted: "#6b7280",
          purple: "#7c3aed",
          "purple-hover": "#6d28d9",
          "purple-soft": "#ede9fe",
        },
      },
      boxShadow: {
        card: "0 1px 3px rgba(15, 23, 42, 0.06), 0 8px 24px rgba(15, 23, 42, 0.06)",
      },
    },
  },
  plugins: [],
};
