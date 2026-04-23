import { cn } from "@/lib/utils";

// ─── Keyframes (injected via globals.css) ─────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-empty-function
export function KeyframeStyle() { return null; }

// ─── Badge ────────────────────────────────────────────────────────────────────

export type BadgeStatus = "pending" | "running" | "processing" | "done" | "error" | "cancelled" | "paused";

const BADGE_CLS: Record<BadgeStatus, string> = {
  pending:    "bg-jb-surface3 text-jb-ink3",
  running:    "bg-jb-run-bg text-jb-run-text",
  processing: "bg-jb-run-bg text-jb-run-text",
  done:       "bg-jb-success50 text-jb-success",
  error:      "bg-jb-danger50 text-jb-danger",
  cancelled:  "bg-jb-cancel-bg text-jb-cancel-text",
  paused:     "bg-jb-cancel-bg text-jb-cancel-text",
};

export function Badge({ status, label }: { status: BadgeStatus; label?: string }) {
  const cls = BADGE_CLS[status] ?? BADGE_CLS.pending;
  const pulse = status === "running" || status === "processing";
  return (
    <span className={cn("inline-flex items-center gap-[5px] px-[9px] py-[3px] rounded-full text-[11.5px] font-semibold whitespace-nowrap", cls)}>
      <span className={cn("w-1.5 h-1.5 rounded-full bg-current shrink-0", pulse && "animate-[jb-pulse_1.4s_ease-in-out_infinite]")} />
      {label ?? status}
    </span>
  );
}

// ─── ProgressBar ──────────────────────────────────────────────────────────────

export function ProgressBar({ value, errors = 0, total, running, done }: {
  value: number; errors?: number; total: number; running?: boolean; done?: boolean;
}) {
  const donePct  = total > 0 ? Math.min(100, (value  / total) * 100) : 0;
  const errorPct = total > 0 ? Math.min(100 - donePct, (errors / total) * 100) : 0;
  return (
    <div className="h-2 rounded-full bg-jb-surface3 overflow-hidden flex">
      <span
        className={cn(
          "h-full relative overflow-hidden transition-[width] duration-[400ms] ease-[cubic-bezier(0.2,0.8,0.2,1)]",
          done && errors === 0 ? "rounded-full" : "rounded-l-full",
          done ? "bg-jb-success" : "bg-jb-accent",
        )}
        style={{ width: `${donePct}%` }}
      >
        {running && (
          <span className="absolute inset-0 bg-gradient-to-r from-transparent via-white/45 to-transparent animate-[jb-shimmer_1.6s_linear_infinite]" />
        )}
      </span>
      {errorPct > 0 && (
        <span
          className="h-full bg-jb-danger transition-[width] duration-[400ms] ease-[cubic-bezier(0.2,0.8,0.2,1)]"
          style={{ width: `${errorPct}%` }}
        />
      )}
    </div>
  );
}

// ─── StatCard ─────────────────────────────────────────────────────────────────

export function StatCard({ label, value, unit, accent, extra }: {
  label: string; value: React.ReactNode; unit?: string;
  accent?: boolean; extra?: React.ReactNode;
}) {
  return (
    <div className={cn(
      "rounded-2xl py-3.5 px-4 border",
      accent ? "bg-jb-accent border-jb-accent text-white" : "bg-jb-surface border-jb-line text-jb-ink",
    )}>
      <div className={cn("text-[11.5px] font-semibold uppercase tracking-[0.06em]", accent ? "text-white/75" : "text-jb-ink3")}>
        {label}
      </div>
      <div className="text-[28px] font-bold tracking-[-0.02em] mt-1 flex items-baseline gap-1.5 tabular-nums">
        {value}
        {unit && (
          <span className={cn("text-[13px] font-medium", accent ? "text-white/70" : "text-jb-ink3")}>
            {unit}
          </span>
        )}
      </div>
      {extra}
    </div>
  );
}

// ─── Card ─────────────────────────────────────────────────────────────────────

export function Card({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={cn("bg-jb-surface border border-jb-line rounded-[20px]", className)}>
      {children}
    </div>
  );
}

export function CardHead({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={cn("px-5 py-3.5 border-b border-jb-line flex items-center gap-3 flex-wrap", className)}>
      {children}
    </div>
  );
}

export function CardBody({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("p-5", className)}>{children}</div>;
}

// ─── SegBtn ───────────────────────────────────────────────────────────────────

export function SegBtn({ options, value, onChange }: {
  options: { label: string; value: string }[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="inline-flex p-[3px] bg-jb-surface2 rounded-xl gap-0.5 overflow-x-auto">
      {options.map((o) => (
        <button
          key={o.value} type="button" onClick={() => onChange(o.value)}
          className={cn(
            "px-3 py-1.5 rounded-lg text-[12.5px] font-semibold border-none whitespace-nowrap shrink-0",
            value === o.value
              ? "bg-jb-surface text-jb-ink shadow-[0_1px_2px_rgba(20,18,30,0.04)]"
              : "bg-transparent text-jb-ink2",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

// ─── FieldChip ────────────────────────────────────────────────────────────────

export function FieldChip({ label, on, onClick }: { label: string; on: boolean; onClick?: () => void }) {
  return (
    <button
      type="button" onClick={onClick}
      className={cn(
        "inline-flex items-center gap-[5px] px-[11px] py-1.5 rounded-full text-xs font-medium border font-mono transition-colors",
        on ? "bg-jb-ink text-white border-jb-ink" : "bg-jb-surface2 text-jb-ink2 border-transparent",
        !onClick && "cursor-default",
      )}
    >
      {on && (
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      )}
      {label}
    </button>
  );
}

