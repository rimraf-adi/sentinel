---
name: ux-design-system
description: High-craft UI/UX design guidelines for Sentinel and Sentinel Hub, inspired by Linear, Raycast, and Vercel. Enforces spatial diversity, refined typography, micro-interactions, dark-mode ergonomics, and data-dense dashboards.
---

# High-Craft UI/UX Design System for Sentinel

When designing or improving interfaces for Sentinel Dashboard, Sentinel Hub, or companion web apps, adhere to these principles:

## 1. Aesthetic Direction: "Industrial Precision & Hyper-Refined Dark Mode"
- **Backgrounds**: Deep, rich blacks (`#08080a`, `#0c0c0e`, `#121215`) rather than washed out grays.
- **Borders & Separators**: Hairline borders using subtle alpha (`rgba(255, 255, 255, 0.08)` or `border-zinc-800/80`). Avoid loud, solid borders.
- **Accents**: Neon Cyan (`#38bdf8`) for primary actions/telemetry, Emerald (`#10b981`) for health/success, Amber (`#f59e0b`) for warnings, Rose (`#f43f5e`) for critical states. Accents should always have matching glow/alpha backgrounds (e.g. `bg-sky-500/10 text-sky-400 border-sky-500/20`).

## 2. Spatial Hierarchy & Visual Weight
- **Typography Floor**: Keep `12px` (`text-xs`) as the standard floor for metadata and tags; headings should use high tracking tightening (`tracking-tight` or `-0.02em`).
- **Data Density**: Machine learning practitioners and autonomous agents need dense data displays (MACs, parameters, gradient norms, latency), but with high breathing room between distinct cards (16px to 24px gaps).
- **Spatial Diversity**: In network/DAG graphs or topology visualizers, avoid flat horizontal conveyor belts. Use layered, vertical/hierarchical positioning that visually reflects model flow (e.g. U-Net encoder descent -> bottleneck -> decoder ascent with skip connection bridges).

## 3. Micro-Interactions & Feedback
- **Instant Clipboard Actions**: 1-click copy with smooth state transition (Terminal icon -> green checkmark -> tooltip).
- **Interactive Previews**: Modals and slide-overs should feature syntax-highlighted YAML/Python previews, line numbers, and instant direct CDN links.
- **Status Dots**: Dynamic pulsing dots (`dot ok pulse`) indicating live WebSocket telemetry or active background processes.

## 4. Components
- **Card Surfaces**: Subtly elevated cards with hover elevation (`hover:border-zinc-700 hover:bg-zinc-900/80 transition-all duration-200`).
- **Tags & Badges**: Small font-mono pills with low-opacity backgrounds matching their category.
