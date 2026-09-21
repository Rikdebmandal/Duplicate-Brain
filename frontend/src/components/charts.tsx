"use client";

import { useId } from "react";

// Charts are hand-drawn SVG rather than a charting library: the shapes needed
// here are simple, and it keeps the bundle free of a dependency whose defaults
// would have to be fought anyway.

const AXIS = "#273248";
const GRID = "#1a2233";
const MUTED = "#8494b3";

// ---------------------------------------------------------------------------
// radar
// ---------------------------------------------------------------------------

export interface RadarPoint {
  label: string;
  value: number;
  confidence?: number;
}

/**
 * Behavioural profile radar.
 *
 * Confidence is drawn as a second, fainter ring rather than dropped: a trait
 * estimated at 0.8 from thirty decisions and one estimated at 0.8 from two
 * look identical on a plain radar, which is exactly the confusion this system
 * is supposed to prevent.
 */
export function RadarChart({
  points,
  size = 340,
  showConfidence = true,
}: {
  points: RadarPoint[];
  size?: number;
  showConfidence?: boolean;
}) {
  const id = useId();
  if (points.length < 3) return null;

  const cx = size / 2;
  const cy = size / 2;
  const radius = size / 2 - 62;
  const step = (Math.PI * 2) / points.length;

  const at = (index: number, value: number) => {
    const angle = index * step - Math.PI / 2;
    return [cx + Math.cos(angle) * radius * value, cy + Math.sin(angle) * radius * value];
  };

  const path = (values: number[]) =>
    values
      .map((value, index) => {
        const [x, y] = at(index, Math.max(0.02, value));
        return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ") + " Z";

  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="w-full" role="img" aria-label="Behavioural profile radar">
      <defs>
        <radialGradient id={`${id}-fill`}>
          <stop offset="0%" stopColor="#5b8def" stopOpacity="0.35" />
          <stop offset="100%" stopColor="#5b8def" stopOpacity="0.12" />
        </radialGradient>
      </defs>

      {[0.25, 0.5, 0.75, 1].map((ring) => (
        <circle
          key={ring}
          cx={cx}
          cy={cy}
          r={radius * ring}
          fill="none"
          stroke={ring === 1 ? AXIS : GRID}
          strokeWidth="1"
        />
      ))}

      {/* The 0.5 ring is the no-information prior; mark it explicitly. */}
      <circle cx={cx} cy={cy} r={radius * 0.5} fill="none" stroke={AXIS} strokeDasharray="3 4" />

      {points.map((point, index) => {
        const [x, y] = at(index, 1);
        const [lx, ly] = at(index, 1.2);
        const anchor = Math.abs(lx - cx) < 12 ? "middle" : lx > cx ? "start" : "end";
        return (
          <g key={point.label}>
            <line x1={cx} y1={cy} x2={x} y2={y} stroke={GRID} strokeWidth="1" />
            <text
              x={lx}
              y={ly}
              fill={MUTED}
              fontSize="9.5"
              textAnchor={anchor}
              dominantBaseline="middle"
            >
              {point.label}
            </text>
          </g>
        );
      })}

      {showConfidence && points.some((p) => p.confidence !== undefined) && (
        <path
          d={path(points.map((p) => p.confidence ?? 0))}
          fill="none"
          stroke={MUTED}
          strokeWidth="1"
          strokeDasharray="2 3"
          opacity="0.7"
        />
      )}

      <path
        d={path(points.map((p) => p.value))}
        fill={`url(#${id}-fill)`}
        stroke="#5b8def"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />

      {points.map((point, index) => {
        const [x, y] = at(index, Math.max(0.02, point.value));
        return <circle key={point.label} cx={x} cy={y} r="2.6" fill="#5b8def" />;
      })}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// line / sparkline
// ---------------------------------------------------------------------------

export function LineChart({
  series,
  height = 180,
  yLabel,
  reference,
}: {
  series: { name: string; color: string; points: { x: number; y: number }[] }[];
  height?: number;
  yLabel?: string;
  reference?: number;
}) {
  const width = 640;
  const pad = { top: 12, right: 12, bottom: 24, left: 34 };
  const all = series.flatMap((s) => s.points);
  if (all.length === 0) return null;

  const xs = all.map((p) => p.x);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs, minX + 1);

  const px = (x: number) =>
    pad.left + ((x - minX) / (maxX - minX)) * (width - pad.left - pad.right);
  const py = (y: number) =>
    pad.top + (1 - Math.max(0, Math.min(1, y))) * (height - pad.top - pad.bottom);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label={yLabel ?? "chart"}>
      {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
        <g key={tick}>
          <line x1={pad.left} y1={py(tick)} x2={width - pad.right} y2={py(tick)} stroke={GRID} />
          <text x={pad.left - 6} y={py(tick)} fill={MUTED} fontSize="9" textAnchor="end" dominantBaseline="middle">
            {(tick * 100).toFixed(0)}
          </text>
        </g>
      ))}

      {reference !== undefined && (
        <line
          x1={pad.left}
          y1={py(reference)}
          x2={width - pad.right}
          y2={py(reference)}
          stroke={MUTED}
          strokeDasharray="4 4"
          opacity="0.6"
        />
      )}

      {series.map((s) => (
        <g key={s.name}>
          <path
            d={s.points
              .map((p, i) => `${i === 0 ? "M" : "L"}${px(p.x).toFixed(1)},${py(p.y).toFixed(1)}`)
              .join(" ")}
            fill="none"
            stroke={s.color}
            strokeWidth="2"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
          {s.points.map((p, i) => (
            <circle key={i} cx={px(p.x)} cy={py(p.y)} r="2.5" fill={s.color} />
          ))}
        </g>
      ))}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// calibration (reliability diagram)
// ---------------------------------------------------------------------------

export function CalibrationChart({
  bins,
}: {
  bins: { mean_confidence: number; observed_accuracy: number; count: number }[];
}) {
  const size = 260;
  const pad = 30;
  const scale = (v: number) => pad + v * (size - pad * 2);
  const maxCount = Math.max(1, ...bins.map((b) => b.count));

  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="w-full max-w-[280px]" role="img" aria-label="Calibration">
      <rect x={pad} y={pad} width={size - pad * 2} height={size - pad * 2} fill="none" stroke={AXIS} />
      {/* Perfect calibration: predicted probability equals observed frequency. */}
      <line
        x1={scale(0)}
        y1={size - scale(0)}
        x2={scale(1)}
        y2={size - scale(1)}
        stroke={MUTED}
        strokeDasharray="4 4"
      />
      {bins.map((bin, index) => (
        <circle
          key={index}
          cx={scale(bin.mean_confidence)}
          cy={size - scale(bin.observed_accuracy)}
          r={4 + (bin.count / maxCount) * 6}
          fill="#5b8def"
          fillOpacity="0.65"
          stroke="#5b8def"
        />
      ))}
      <text x={size / 2} y={size - 6} fill={MUTED} fontSize="9" textAnchor="middle">
        predicted probability
      </text>
      <text
        x={10}
        y={size / 2}
        fill={MUTED}
        fontSize="9"
        textAnchor="middle"
        transform={`rotate(-90 10 ${size / 2})`}
      >
        observed frequency
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------------------
// decision timeline
// ---------------------------------------------------------------------------

export function TimelineChart({
  entries,
}: {
  entries: { occurred_at: string; approach: number; importance: number; chosen_option: string }[];
}) {
  const width = 700;
  const height = 130;
  const pad = { top: 14, right: 12, bottom: 22, left: 12 };
  if (entries.length === 0) return null;

  const times = entries.map((e) => new Date(e.occurred_at).getTime());
  const minT = Math.min(...times);
  const maxT = Math.max(...times, minT + 1);
  const px = (t: number) =>
    pad.left + ((t - minT) / (maxT - minT)) * (width - pad.left - pad.right);
  const mid = (height - pad.bottom + pad.top) / 2;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label="Decision timeline">
      <line x1={pad.left} y1={mid} x2={width - pad.right} y2={mid} stroke={AXIS} />
      <text x={pad.left} y={pad.top - 2} fill={MUTED} fontSize="9">
        took the active option
      </text>
      <text x={pad.left} y={height - 6} fill={MUTED} fontSize="9">
        chose the safer option
      </text>

      {entries.map((entry, index) => {
        const x = px(new Date(entry.occurred_at).getTime());
        const offset = (entry.approach - 0.5) * 2 * (mid - pad.top - 4);
        const y = mid - offset;
        const r = 2.5 + (entry.importance / 10) * 4;
        return (
          <g key={index}>
            <line x1={x} y1={mid} x2={x} y2={y} stroke={GRID} strokeWidth="1" />
            <circle
              cx={x}
              cy={y}
              r={r}
              fill={entry.approach >= 0.5 ? "#5b8def" : "#3fb984"}
              fillOpacity="0.8"
            >
              <title>{`${entry.occurred_at.slice(0, 10)} - ${entry.chosen_option} (importance ${entry.importance}/10)`}</title>
            </circle>
          </g>
        );
      })}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// decision landscape (factor sweep)
// ---------------------------------------------------------------------------

export function SweepChart({
  points,
  optionLabel,
  currentValue,
  flipAt,
}: {
  points: { value: number; probability: number }[];
  optionLabel: string;
  currentValue: number;
  flipAt: number | null;
}) {
  const width = 260;
  const height = 84;
  const pad = { top: 8, right: 8, bottom: 14, left: 8 };
  const px = (v: number) => pad.left + v * (width - pad.left - pad.right);
  const py = (v: number) => pad.top + (1 - v) * (height - pad.top - pad.bottom);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label={`Sweep for ${optionLabel}`}>
      <line x1={pad.left} y1={py(0.5)} x2={width - pad.right} y2={py(0.5)} stroke={GRID} strokeDasharray="3 3" />
      {flipAt !== null && (
        <line x1={px(flipAt)} y1={pad.top} x2={px(flipAt)} y2={height - pad.bottom} stroke="#e0a44b" strokeWidth="1.5" />
      )}
      <path
        d={points
          .map((p, i) => `${i === 0 ? "M" : "L"}${px(p.value).toFixed(1)},${py(p.probability).toFixed(1)}`)
          .join(" ")}
        fill="none"
        stroke="#5b8def"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <circle cx={px(currentValue)} cy={py(points.find((p) => Math.abs(p.value - currentValue) < 0.13)?.probability ?? 0.5)} r="3.5" fill="#c9d3e6" />
      <text x={pad.left} y={height - 3} fill={MUTED} fontSize="8">
        0
      </text>
      <text x={width - pad.right} y={height - 3} fill={MUTED} fontSize="8" textAnchor="end">
        1
      </text>
    </svg>
  );
}
