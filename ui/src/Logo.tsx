import { useId } from 'react';

/** Sentinel mark: two graph nodes joined by an S-shaped edge. Same artwork as public/favicon.svg. */
export function Logo({ size = 28 }: { size?: number }) {
  const id = useId();
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" role="img" aria-label="Sentinel" style={{ flexShrink: 0, filter: 'drop-shadow(0 6px 14px rgba(124,92,255,.45))' }}>
      <defs>
        <linearGradient id={id} x1="3" y1="2" x2="29" y2="30" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#7c5cff" /><stop offset=".55" stopColor="#4f8cff" /><stop offset="1" stopColor="#22d3ee" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="9" fill={`url(#${id})`} />
      <path d="M21.5 10.2C13.2 10.2 13.2 16 16 16s2.8 5.8-5.5 5.8" fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" />
      <circle cx="21.5" cy="10.2" r="3" fill="#fff" /><circle cx="10.5" cy="21.8" r="3" fill="#fff" />
    </svg>
  );
}
