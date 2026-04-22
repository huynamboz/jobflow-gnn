import { T } from "./_tokens";

// ─── Keyframes ────────────────────────────────────────────────────────────────

export function KeyframeStyle() {
  return (
    <style>{`
      @keyframes jb-shimmer { 0%{transform:translateX(-100%)} 100%{transform:translateX(100%)} }
      @keyframes jb-pulse   { 0%,100%{opacity:1} 50%{opacity:0.5} }
      @keyframes jb-slide   { from{transform:translateX(32px);opacity:0} to{transform:translateX(0);opacity:1} }
      @keyframes jb-fade    { from{opacity:0} to{opacity:1} }
      @keyframes jb-spin    { to{transform:rotate(360deg)} }
      .jb-row:hover td { background: ${T.surface2} !important; }
      .jb-card-hover:hover { transform:translateY(-2px); box-shadow:0 2px 4px rgba(20,18,30,.04),0 8px 24px rgba(20,18,30,.06); }
    `}</style>
  );
}

// ─── Badge ────────────────────────────────────────────────────────────────────

export type BadgeStatus = "pending" | "running" | "processing" | "done" | "error" | "cancelled" | "paused";

const BADGE_STYLES: Record<BadgeStatus, { bg: string; color: string }> = {
  pending:    { bg: T.surface3,               color: T.ink3 },
  running:    { bg: "oklch(0.93 0.05 240)",   color: "oklch(0.42 0.16 240)" },
  processing: { bg: "oklch(0.93 0.05 240)",   color: "oklch(0.42 0.16 240)" },
  done:       { bg: T.success50,              color: T.success },
  error:      { bg: T.danger50,               color: T.danger },
  cancelled:  { bg: "oklch(0.94 0.03 60)",    color: "oklch(0.52 0.12 60)" },
  paused:     { bg: "oklch(0.94 0.03 60)",    color: "oklch(0.52 0.12 60)" },
};

export function Badge({ status, label }: { status: BadgeStatus; label?: string }) {
  const s = BADGE_STYLES[status] ?? BADGE_STYLES.pending;
  const pulse = status === "running" || status === "processing";
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 5,
      padding: "3px 9px", borderRadius: 999,
      fontSize: 11.5, fontWeight: 600,
      background: s.bg, color: s.color, whiteSpace: "nowrap",
    }}>
      <span style={{
        width: 6, height: 6, borderRadius: "50%", background: "currentColor", flexShrink: 0,
        animation: pulse ? "jb-pulse 1.4s ease-in-out infinite" : undefined,
      }} />
      {label ?? status}
    </span>
  );
}

// ─── ProgressBar ──────────────────────────────────────────────────────────────

export function ProgressBar({ value, total, running, done }: {
  value: number; total: number; running?: boolean; done?: boolean;
}) {
  const pct = total > 0 ? Math.min(100, (value / total) * 100) : 0;
  return (
    <div style={{ height: 8, borderRadius: 999, background: T.surface3, overflow: "hidden" }}>
      <span style={{
        display: "block", height: "100%", borderRadius: 999,
        background: done ? T.success : T.accent,
        width: `${pct}%`,
        transition: "width 0.4s cubic-bezier(0.2,0.8,0.2,1)",
        position: "relative", overflow: "hidden",
      }}>
        {running && (
          <span style={{
            position: "absolute", inset: 0,
            background: "linear-gradient(90deg,transparent,rgba(255,255,255,0.45),transparent)",
            animation: "jb-shimmer 1.6s linear infinite",
          }} />
        )}
      </span>
    </div>
  );
}

// ─── StatCard ─────────────────────────────────────────────────────────────────

export function StatCard({ label, value, unit, accent, extra }: {
  label: string; value: React.ReactNode; unit?: string;
  accent?: boolean; extra?: React.ReactNode;
}) {
  return (
    <div style={{
      background: accent ? T.accent : T.surface,
      border: `1px solid ${accent ? T.accent : T.line}`,
      borderRadius: 16, padding: "14px 16px",
      color: accent ? "#fff" : T.ink,
    }}>
      <div style={{ fontSize: 11.5, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", color: accent ? "rgba(255,255,255,0.75)" : T.ink3 }}>
        {label}
      </div>
      <div style={{ fontSize: 28, fontWeight: 700, letterSpacing: "-0.02em", marginTop: 4, display: "flex", alignItems: "baseline", gap: 6, fontVariantNumeric: "tabular-nums" }}>
        {value}
        {unit && <span style={{ fontSize: 13, fontWeight: 500, color: accent ? "rgba(255,255,255,0.7)" : T.ink3 }}>{unit}</span>}
      </div>
      {extra}
    </div>
  );
}

// ─── Card ─────────────────────────────────────────────────────────────────────

export function Card({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div style={{ background: T.surface, border: `1px solid ${T.line}`, borderRadius: 20, ...style }}>
      {children}
    </div>
  );
}

export function CardHead({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div style={{ padding: "14px 20px", borderBottom: `1px solid ${T.line}`, display: "flex", alignItems: "center", gap: 12, ...style }}>
      {children}
    </div>
  );
}

export function CardBody({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return <div style={{ padding: 20, ...style }}>{children}</div>;
}

// ─── SegBtn ───────────────────────────────────────────────────────────────────

export function SegBtn({ options, value, onChange }: {
  options: { label: string; value: string }[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div style={{ display: "inline-flex", padding: 3, background: T.surface2, borderRadius: 12, gap: 2 }}>
      {options.map((o) => (
        <button key={o.value} type="button" onClick={() => onChange(o.value)} style={{
          border: "none", padding: "6px 12px", borderRadius: 8,
          fontSize: 12.5, fontWeight: 600, cursor: "pointer",
          background: value === o.value ? T.surface : "transparent",
          color: value === o.value ? T.ink : T.ink2,
          boxShadow: value === o.value ? "0 1px 2px rgba(20,18,30,0.04)" : "none",
        }}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

// ─── FieldChip ────────────────────────────────────────────────────────────────

export function FieldChip({ label, on, onClick }: { label: string; on: boolean; onClick?: () => void }) {
  return (
    <button type="button" onClick={onClick} style={{
      display: "inline-flex", alignItems: "center", gap: 5,
      padding: "6px 11px", borderRadius: 999,
      background: on ? T.ink : T.surface2,
      color: on ? "#fff" : T.ink2,
      border: `1px solid ${on ? T.ink : "transparent"}`,
      fontSize: 12, fontWeight: 500, cursor: onClick ? "pointer" : "default",
      fontFamily: "'JetBrains Mono',ui-monospace,monospace",
    }}>
      {on && (
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      )}
      {label}
    </button>
  );
}
