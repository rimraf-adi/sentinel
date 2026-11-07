/**
 * Sentinel high-craft vector mark.
 * Features the signature geometric 'S' neural tensor glyph with dual terminal nodes
 * and electric cyan (#38bdf8) to vibrant emerald (#10b981) precision gradient.
 */
export function Logo({ size = 28, withGlow = true }: { size?: number; withGlow?: boolean }) {
  return (
    <svg 
      width={size} 
      height={size} 
      viewBox="0 0 36 36" 
      fill="none" 
      xmlns="http://www.w3.org/2000/svg" 
      role="img" 
      aria-label="Sentinel" 
      style={{ flexShrink: 0 }}
    >
      <defs>
        <linearGradient id="sentinel-grad" x1="15%" y1="10%" x2="85%" y2="90%">
          <stop offset="0%" stopColor="#38bdf8" />
          <stop offset="55%" stopColor="#2dd4bf" />
          <stop offset="100%" stopColor="#10b981" />
        </linearGradient>
        {withGlow && (
          <filter id="sentinel-glow" x="-25%" y="-25%" width="150%" height="150%">
            <feGaussianBlur stdDeviation="1.5" result="blur" />
            <feColorMatrix type="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 0.5 0" />
            <feMerge>
              <feMergeNode />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        )}
      </defs>

      {/* Titanium / Obsidian Squircle Base */}
      <rect 
        x="1.5" 
        y="1.5" 
        width="33" 
        height="33" 
        rx="8.5" 
        fill="#08090d" 
        stroke="rgba(255, 255, 255, 0.12)" 
        strokeWidth="1" 
      />

      {/* Ambient Depth Rim */}
      <rect 
        x="2.5" 
        y="2.5" 
        width="31" 
        height="31" 
        rx="7.5" 
        fill="none" 
        stroke="rgba(56, 189, 248, 0.08)" 
        strokeWidth="1" 
      />

      <g filter={withGlow ? "url(#sentinel-glow)" : undefined}>
        {/* Top Conduit Ribbon & Angle */}
        <path
          d="M 23.5 11.5 H 15.5 L 10.5 16.5"
          stroke="url(#sentinel-grad)"
          strokeWidth="2.75"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Central Diagonal Tensor Spine */}
        <path
          d="M 12 15.5 L 24 20.5"
          stroke="url(#sentinel-grad)"
          strokeWidth="3.2"
          strokeLinecap="round"
        />

        {/* Bottom Conduit Ribbon & Angle */}
        <path
          d="M 25.5 19.5 L 20.5 24.5 H 12.5"
          stroke="url(#sentinel-grad)"
          strokeWidth="2.75"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Terminal Tensor Nodes */}
        <circle cx="23.5" cy="11.5" r="2.4" fill="#08090d" stroke="url(#sentinel-grad)" strokeWidth="1.8" />
        <circle cx="23.5" cy="11.5" r="1.1" fill="#38bdf8" />

        <circle cx="12.5" cy="24.5" r="2.4" fill="#08090d" stroke="url(#sentinel-grad)" strokeWidth="1.8" />
        <circle cx="12.5" cy="24.5" r="1.1" fill="#10b981" />
      </g>
    </svg>
  );
}
