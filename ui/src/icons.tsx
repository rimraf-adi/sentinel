import type { SVGProps } from 'react';

const P: Record<string, string> = {
  overview: 'M3 3h7v9H3z M14 3h7v5h-7z M14 12h7v9h-7z M3 16h7v5H3z',
  runs: 'M6 4l14 8-14 8V4z',
  experiments: 'M9 3h6 M10 3v6L4.5 19a2 2 0 0 0 1.8 3h11.4a2 2 0 0 0 1.8-3L14 9V3',
  sweeps: 'M4 21v-7 M4 10V3 M12 21v-9 M12 8V3 M20 21v-5 M20 12V3 M1 14h6 M9 8h6 M17 16h6',
  system: 'M5 5h14v14H5z M9 9h6v6H9z M9 2v3 M15 2v3 M9 19v3 M15 19v3 M19 9h3 M19 14h3 M2 9h3 M2 14h3',
  components: 'M12 2l10 5-10 5L2 7l10-5z M2 17l10 5 10-5 M2 12l10 5 10-5',
  builder: 'M3 3h6v6H3z M15 15h6v6h-6z M9 6h3a3 3 0 0 1 3 3v6',
  visualizer: 'M3 3h6v6H3z M15 15h6v6h-6z M9 6h3a3 3 0 0 1 3 3v6',
  datasets: 'M12 3c4.97 0 9 1.34 9 3s-4.03 3-9 3-9-1.34-9-3 4.03-3 9-3z M3 6v6c0 1.66 4.03 3 9 3s9-1.34 9-3V6 M3 12v6c0 1.66 4.03 3 9 3s9-1.34 9-3v-6',
  code: 'M16 18l6-6-6-6 M8 6l-6 6 6 6',
  search: 'M11 3a8 8 0 1 0 0 16 8 8 0 0 0 0-16z M21 21l-4.3-4.3',
  plus: 'M12 5v14 M5 12h14',
  bolt: 'M13 2L4 14h7l-1 8 9-12h-7l1-8z',
  chart: 'M3 3v18h18 M7 15l4-5 3 3 5-7',
  flow: 'M2 12h3l3-8 4 16 3-11 3 5 4-2',
  arrow: 'M5 12h14 M13 6l6 6-6 6',
  layers: 'M12 2l10 5-10 5L2 7l10-5z',
};

export type IconName = keyof typeof P;

export function Icon({ name, size = 16, ...rest }: { name: IconName; size?: number } & Omit<SVGProps<SVGSVGElement>, 'name'>) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...rest}>
      {P[name].split(' M').map((d, i) => <path key={i} d={(i ? 'M' : '') + d} />)}
    </svg>
  );
}
