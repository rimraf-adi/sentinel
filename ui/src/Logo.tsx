/** Sentinel mark: two graph nodes joined by an S-shaped edge. Same artwork as public/favicon.svg. */
export function Logo({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" role="img" aria-label="Sentinel" style={{ flexShrink: 0 }}>
      <rect width="32" height="32" rx="8" fill="#ececee" />
      <path d="M21.5 10.2C13.2 10.2 13.2 16 16 16s2.8 5.8-5.5 5.8" fill="none" stroke="#0b0b0c" strokeWidth="2.5" strokeLinecap="round" />
      <circle cx="21.5" cy="10.2" r="3" fill="#0b0b0c" /><circle cx="10.5" cy="21.8" r="3" fill="#0b0b0c" />
    </svg>
  );
}
