import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  IconChevronLeft,
  IconClock,
  IconLoader2,
  IconPlayerStop,
  IconRefresh,
  IconX,
} from "@tabler/icons-react";

import { cn } from "@/lib/utils";
import { labelingService } from "@/services/labeling.service";
import type { LabelingBatchDetail, RecentLabel } from "@/types/labeling-batch.types";
import { Badge, type BadgeStatus, Card, CardBody, CardHead, KeyframeStyle, ProgressBar, StatCard } from "../jd-batch/_primitives";

const POLL_MS = 3000;

function fmtDate(iso: string) {
  return new Date(iso).toLocaleString("vi-VN", { dateStyle: "short", timeStyle: "short" });
}

// ── Score helpers ─────────────────────────────────────────────────────────────

const SCORE_CLS: Record<number, string> = {
  0: "bg-jb-danger50 text-jb-danger",
  1: "bg-amber-100 text-amber-700",
  2: "bg-jb-success50 text-jb-success",
};
const OVERALL_CLS: Record<number, string> = {
  0: "bg-jb-danger50 text-jb-danger ring-1 ring-jb-danger/30",
  1: "bg-amber-100 text-amber-700 ring-1 ring-amber-300/40",
  2: "bg-jb-success50 text-jb-success ring-1 ring-jb-success/30",
};
const OVERALL_LABEL: Record<number, string> = {
  0: "Not suitable",
  1: "Suitable",
  2: "Strong fit",
};

function ScoreChip({ value, large }: { value: number; large?: boolean }) {
  return (
    <span className={cn(
      "inline-block rounded-md font-bold tabular-nums",
      large ? "px-3 py-1 text-[15px]" : "px-2 py-0.5 text-[11px]",
      SCORE_CLS[value] ?? "bg-jb-surface3 text-jb-ink3",
    )}>
      {value}
    </span>
  );
}

const SELECTION_CLS: Record<string, string> = {
  high_overlap:   "bg-jb-success50 text-jb-success",
  medium_overlap: "bg-jb-accent50 text-jb-accent600",
  hard_negative:  "bg-jb-danger50 text-jb-danger",
  random:         "bg-jb-surface3 text-jb-ink3",
};

function SelectionChip({ value }: { value: string }) {
  return (
    <span className={cn(
      "inline-block rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
      SELECTION_CLS[value] ?? "bg-jb-surface3 text-jb-ink3",
    )}>
      {value.replace(/_/g, " ")}
    </span>
  );
}

// ── Label drawer ──────────────────────────────────────────────────────────────

const DIM_META: { key: "skill_fit" | "seniority_fit" | "experience_fit" | "domain_fit"; label: string; descs: [string, string, string] }[] = [
  { key: "skill_fit",      label: "Skill fit",      descs: ["<30% skills matched", "30–70% matched", ">70% matched"] },
  { key: "seniority_fit",  label: "Seniority",      descs: ["≥2 levels apart", "1 level apart", "Exact / overqualified"] },
  { key: "experience_fit", label: "Experience",     descs: ["<50% of required", "50–90% of required", "Meets or exceeds"] },
  { key: "domain_fit",     label: "Domain",         descs: ["Different domain", "Related domain", "Same domain"] },
];

function DrawerSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <p className="text-[10px] font-semibold uppercase tracking-[0.07em] text-default-400 mb-2">{title}</p>
      {children}
    </div>
  );
}

function SkillPills({ skills }: { skills: string[] }) {
  if (!skills.length) return <span className="text-xs text-default-400">—</span>;
  return (
    <div className="flex flex-wrap gap-1">
      {skills.map((s) => (
        <span key={s} className="rounded-md border border-default-200 bg-default-50 px-1.5 py-0.5 text-[11px] text-default-600">{s}</span>
      ))}
    </div>
  );
}

function LabelDrawer({ row, onClose }: { row: RecentLabel; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex justify-end" onClick={onClose}>
      <div
        className="relative h-full w-full max-w-lg overflow-y-auto bg-white shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="sticky top-0 flex items-center justify-between border-b border-default-200 bg-white px-5 py-4">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-semibold text-default-900">Pair #{row.pair_id}</span>
            <SelectionChip value={row.selection} />
            <span className="text-[11px] text-default-400">{fmtDate(row.created_at)}</span>
          </div>
          <button onClick={onClose} className="rounded-lg p-1.5 text-default-400 hover:bg-default-100 shrink-0">
            <IconX size={16} />
          </button>
        </div>

        <div className="p-5 space-y-5">

          {/* Overall verdict */}
          <div className="rounded-xl border border-default-200 bg-default-50 px-4 py-3 flex items-center gap-4">
            <span className={cn("rounded-lg px-3 py-1.5 text-[28px] font-bold tabular-nums leading-none", OVERALL_CLS[row.overall])}>
              {row.overall}
            </span>
            <div className="flex-1 min-w-0">
              <p className="font-semibold text-[14px] text-default-900">{OVERALL_LABEL[row.overall]}</p>
              <div className="flex gap-3 mt-1.5 flex-wrap">
                {DIM_META.map(({ key, label }) => (
                  <span key={key} className="flex items-center gap-1 text-[11px] text-default-500">
                    <span className={cn("inline-block rounded px-1 font-bold text-[10px]", SCORE_CLS[row[key]])}>{row[key]}</span>
                    {label}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* CV card */}
          <div className="rounded-xl border border-default-200 overflow-hidden">
            <div className="bg-default-50 px-4 py-2.5 flex items-center gap-2 border-b border-default-100">
              <span className="text-[10px] font-semibold uppercase tracking-wide text-default-400">CV</span>
              <span className="font-mono text-[11px] text-default-500">#{row.cv_id}</span>
              <span className="ml-auto rounded-md bg-white border border-default-200 px-2 py-0.5 text-[11px] font-medium text-default-700">{row.cv_role || "other"}</span>
              <span className="text-[11px] text-default-500">{row.cv_seniority}</span>
              <span className="text-[11px] text-default-400">{row.cv_experience}y exp</span>
            </div>
            <div className="px-4 py-3 space-y-2.5">
              <DrawerSection title="Skills">
                <SkillPills skills={row.cv_skills} />
              </DrawerSection>
              {row.cv_text && (
                <DrawerSection title="Summary">
                  <p className="text-[11.5px] leading-relaxed text-default-600 whitespace-pre-wrap line-clamp-5">{row.cv_text}</p>
                </DrawerSection>
              )}
            </div>
          </div>

          {/* JD card */}
          <div className="rounded-xl border border-default-200 overflow-hidden">
            <div className="bg-default-50 px-4 py-2.5 flex items-center gap-2 flex-wrap border-b border-default-100">
              <span className="text-[10px] font-semibold uppercase tracking-wide text-default-400">Job</span>
              <span className="font-mono text-[11px] text-default-500">#{row.job_id}</span>
              <span className="font-semibold text-[12px] text-default-800 flex-1 min-w-0 truncate">{row.job_title}</span>
              <span className="rounded-md bg-white border border-default-200 px-2 py-0.5 text-[11px] font-medium text-default-700 shrink-0">{row.job_role || "other"}</span>
            </div>
            <div className="px-4 py-3 space-y-2.5">
              <div className="flex gap-3 text-[11.5px] text-default-500">
                <span>Seniority: <strong className="text-default-700">{row.job_seniority}</strong></span>
                <span>Exp: <strong className="text-default-700">{row.job_experience}</strong></span>
              </div>
              <DrawerSection title="Required skills">
                <SkillPills skills={row.job_skills} />
              </DrawerSection>
              {row.job_text && (
                <DrawerSection title="Description">
                  <p className="text-[11.5px] leading-relaxed text-default-600 whitespace-pre-wrap line-clamp-5">{row.job_text}</p>
                </DrawerSection>
              )}
            </div>
          </div>

          {/* Dim score breakdown */}
          <DrawerSection title="Score breakdown">
            <div className="space-y-2">
              {DIM_META.map(({ key, label, descs }) => (
                <div key={key} className="flex items-center gap-3">
                  <ScoreChip value={row[key]} large />
                  <div className="flex-1 min-w-0">
                    <p className="text-[12.5px] font-semibold text-default-800">{label}</p>
                    <p className="text-[11px] text-default-400">{descs[row[key]] ?? ""}</p>
                  </div>
                </div>
              ))}
            </div>
          </DrawerSection>

        </div>
      </div>
    </div>
  );
}

// ── Label distribution bar ────────────────────────────────────────────────────

function DistBar({ total, n0, n1, n2 }: { total: number; n0: number; n1: number; n2: number }) {
  if (total === 0) return <div className="text-jb-ink4 text-xs">No labels yet</div>;
  const pct = (n: number) => total > 0 ? ((n / total) * 100).toFixed(0) : "0";
  return (
    <div className="space-y-1.5 text-[12px]">
      {[
        { label: "Not suitable (0)", n: n0, cls: "bg-jb-danger" },
        { label: "Suitable (1)",     n: n1, cls: "bg-amber-400" },
        { label: "Strong fit (2)",   n: n2, cls: "bg-jb-success" },
      ].map(({ label, n, cls }) => (
        <div key={label} className="flex items-center gap-2">
          <span className="text-jb-ink3 w-[120px] shrink-0">{label}</span>
          <div className="flex-1 h-2 bg-jb-surface3 rounded-full overflow-hidden">
            <div className={cn("h-full rounded-full transition-[width]", cls)} style={{ width: `${pct(n)}%` }} />
          </div>
          <span className="text-jb-ink2 font-semibold w-8 text-right tabular-nums">{pct(n)}%</span>
          <span className="text-jb-ink4 w-7 tabular-nums text-right">{n}</span>
        </div>
      ))}
    </div>
  );
}

// ── Labels table ──────────────────────────────────────────────────────────────

const thCls = "text-left text-[11px] font-semibold text-jb-ink3 uppercase tracking-[0.06em] px-3 py-2.5 border-b border-jb-line bg-jb-surface2 whitespace-nowrap";
const tdCls = "px-3 py-[11px] border-b border-jb-line align-middle";

function LabelsTable({ rows, onSelect }: { rows: RecentLabel[]; onSelect: (r: RecentLabel) => void }) {
  if (rows.length === 0) {
    return <div className="text-center text-jb-ink3 py-10 text-sm">No labels yet.</div>;
  }
  return (
    <div className="overflow-x-auto max-h-[520px] overflow-y-auto">
      <table className="w-full min-w-[780px] border-separate border-spacing-0 text-[13px]">
        <thead>
          <tr>
            <th className={thCls} style={{ width: 56 }}>Pair</th>
            <th className={thCls}>CV role</th>
            <th className={thCls}>Job title</th>
            <th className={thCls}>Job role</th>
            <th className={thCls} style={{ width: 56 }}>Skill</th>
            <th className={thCls} style={{ width: 70 }}>Seniority</th>
            <th className={thCls} style={{ width: 50 }}>Exp</th>
            <th className={thCls} style={{ width: 65 }}>Domain</th>
            <th className={thCls} style={{ width: 72 }}>Overall</th>
            <th className={thCls}>Type</th>
            <th className={thCls} style={{ width: 70 }}>Time</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr
              key={r.pair_id}
              onClick={() => onSelect(r)}
              className="cursor-pointer hover:bg-jb-surface2 transition-colors"
            >
              <td className={cn(tdCls, "font-mono text-xs text-jb-ink4")}>#{r.pair_id}</td>
              <td className={cn(tdCls, "text-jb-ink2 text-xs")}>
                <span className="rounded-md bg-jb-surface2 px-1.5 py-0.5">{r.cv_role || "—"}</span>
              </td>
              <td className={cn(tdCls, "font-medium text-jb-ink max-w-[200px] truncate")} title={r.job_title}>
                {r.job_title || "—"}
              </td>
              <td className={cn(tdCls, "text-jb-ink2 text-xs")}>
                <span className="rounded-md bg-jb-surface2 px-1.5 py-0.5">{r.job_role || "—"}</span>
              </td>
              <td className={tdCls}><ScoreChip value={r.skill_fit} /></td>
              <td className={tdCls}><ScoreChip value={r.seniority_fit} /></td>
              <td className={tdCls}><ScoreChip value={r.experience_fit} /></td>
              <td className={tdCls}><ScoreChip value={r.domain_fit} /></td>
              <td className={tdCls}>
                <span className={cn("inline-block rounded-md px-2 py-0.5 text-[12px] font-bold tabular-nums", OVERALL_CLS[r.overall])}>
                  {r.overall}
                </span>
              </td>
              <td className={tdCls}><SelectionChip value={r.selection} /></td>
              <td className={cn(tdCls, "text-jb-ink4 text-xs tabular-nums whitespace-nowrap")}>
                {new Date(r.created_at).toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function LabelBatchDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const batchId = Number(id);

  const [detail, setDetail] = useState<LabelingBatchDetail | null>(null);
  const [workers, setWorkers] = useState(3);
  const [cancelling, setCancelling] = useState(false);
  const [resuming, setResuming] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [selected, setSelected] = useState<RecentLabel | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const load = useCallback(async () => {
    try {
      const d = await labelingService.getBatch(batchId);
      setDetail(d);
      return d;
    } catch { return null; }
  }, [batchId]);

  useEffect(() => { load(); }, [load]);
  useEffect(() => { if (detail) setWorkers(detail.workers); }, [detail?.workers]);

  useEffect(() => {
    if (detail?.status !== "running") return;
    pollRef.current = setInterval(async () => {
      const d = await load();
      if (d && d.status !== "running") clearInterval(pollRef.current!);
    }, POLL_MS);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [detail?.status, load]);

  const handleCancel = async () => {
    setCancelling(true); setErr(null);
    try { await labelingService.cancelBatch(batchId); await load(); }
    catch { setErr("Failed to cancel."); }
    finally { setCancelling(false); }
  };

  const handleResume = async () => {
    setResuming(true); setErr(null);
    try { await labelingService.resumeBatch(batchId, workers); await load(); }
    catch (e: unknown) {
      const msg = (e as { response?: { data?: { error?: { message?: string } } } })?.response?.data?.error?.message ?? "Failed to resume.";
      setErr(msg);
    }
    finally { setResuming(false); }
  };

  if (!detail) {
    return (
      <div className="flex items-center justify-center h-40 text-jb-ink3">
        <IconLoader2 size={20} className="animate-[jb-spin_0.7s_linear_infinite]" />
      </div>
    );
  }

  const running = detail.status === "running";
  const resumable = detail.status === "error" || detail.status === "cancelled";
  const pct = detail.total > 0 ? (detail.done_count / detail.total) * 100 : 0;
  const pending = Math.max(0, detail.total - detail.done_count - detail.error_count);
  const distTotal = detail.dist.overall_0 + detail.dist.overall_1 + detail.dist.overall_2;

  return (
    <div className="flex flex-col gap-5">
      <KeyframeStyle />

      {/* Header */}
      <div className="flex items-start gap-4 flex-wrap">
        <div className="flex-1 min-w-0">
          <button
            type="button"
            onClick={() => navigate("/admin/label-batch")}
            className="bg-transparent border-none text-jb-ink3 text-[12.5px] cursor-pointer flex items-center gap-1 p-0 mb-2.5 font-medium"
          >
            <IconChevronLeft size={13} /> All batches
          </button>
          <div className="flex items-center gap-3 flex-wrap">
            <h2 className="text-[28px] font-bold tracking-[-0.025em] m-0 text-jb-ink">
              Batch #{detail.id}
            </h2>
            <Badge status={detail.status as BadgeStatus} />
          </div>
          <div className="flex items-center gap-3.5 mt-2 flex-wrap text-jb-ink3 text-[13px]">
            <span className="flex items-center gap-[5px]">
              <IconClock size={13} />{fmtDate(detail.created_at)}
            </span>
            <span>{detail.workers} workers</span>
            {running && (
              <span className="flex items-center gap-[5px] text-jb-accent">
                <IconLoader2 size={13} className="animate-[jb-spin_1.5s_linear_infinite]" />
                Labeling…
              </span>
            )}
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-2 items-center flex-wrap pt-0 sm:pt-7">
          {!running && (
            <div className="flex items-center gap-1.5 border border-jb-line rounded-[10px] px-2.5 py-1.5 bg-jb-surface">
              <span className="text-[11px] font-semibold text-jb-ink3 uppercase tracking-wide">Workers</span>
              <button type="button" onClick={() => setWorkers((w) => Math.max(1, w - 1))}
                className="w-5 h-5 rounded-md bg-jb-surface2 text-jb-ink2 text-xs font-bold flex items-center justify-center border-none leading-none">−</button>
              <span className="text-[13px] font-bold text-jb-ink w-4 text-center tabular-nums">{workers}</span>
              <button type="button" onClick={() => setWorkers((w) => Math.min(20, w + 1))}
                className="w-5 h-5 rounded-md bg-jb-surface2 text-jb-ink2 text-xs font-bold flex items-center justify-center border-none leading-none">+</button>
            </div>
          )}
          {resumable && (
            <button
              type="button" onClick={handleResume} disabled={resuming}
              className="flex items-center gap-1.5 py-2 px-3.5 rounded-[10px] border-none bg-jb-accent50 text-jb-accent600 text-[13px] font-semibold disabled:opacity-50"
            >
              {resuming ? <IconLoader2 size={13} className="animate-[jb-spin_0.7s_linear_infinite]" /> : <IconRefresh size={13} />}
              {resuming ? "Resuming…" : "Resume"}
            </button>
          )}
          {running && (
            <button
              type="button" onClick={handleCancel} disabled={cancelling}
              className="flex items-center gap-1.5 py-2 px-3.5 rounded-[10px] border-none bg-jb-danger50 text-jb-danger text-[13px] font-semibold disabled:opacity-50"
            >
              {cancelling ? <IconLoader2 size={13} className="animate-[jb-spin_0.7s_linear_infinite]" /> : <IconPlayerStop size={13} />}
              {cancelling ? "Stopping…" : "Stop"}
            </button>
          )}
          <button
            type="button" onClick={load}
            className="flex items-center gap-1.5 py-2 px-3.5 rounded-[10px] border border-jb-line bg-jb-surface text-jb-ink2 text-[13px]"
          >
            <IconRefresh size={13} /> Refresh
          </button>
        </div>
      </div>

      {err && (
        <div className="flex items-center gap-2 px-3.5 py-2.5 rounded-xl bg-jb-danger50 text-jb-danger text-sm">
          {err}
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard
          label="Progress" value={`${pct.toFixed(0)}%`}
          accent={running}
          extra={
            <div className="mt-2.5">
              <ProgressBar value={detail.done_count} errors={detail.error_count} total={detail.total} running={running} done={detail.status === "done"} />
            </div>
          }
        />
        <StatCard label="Labeled" value={<span className="text-jb-success">{detail.done_count}</span>} unit={`/ ${detail.total}`} />
        <StatCard
          label="Errors"
          value={<span className={detail.error_count > 0 ? "text-jb-danger" : "text-jb-ink"}>{detail.error_count}</span>}
          unit={detail.error_count === 0 ? "clean" : "failed"}
        />
        <StatCard label="Pending" value={pending} unit="pairs" />
      </div>

      {/* Score distribution */}
      {distTotal > 0 && (
        <Card>
          <CardHead>
            <span className="font-semibold text-[14px]">Label distribution</span>
            <span className="text-[11.5px] text-jb-ink3 ml-auto">{distTotal} labels in this batch</span>
          </CardHead>
          <CardBody>
            <DistBar total={distTotal} n0={detail.dist.overall_0} n1={detail.dist.overall_1} n2={detail.dist.overall_2} />
          </CardBody>
        </Card>
      )}

      {/* Recent labels table */}
      <Card>
        <CardHead>
          <span className="font-semibold text-[14px]">
            Recent labels
            {running && <span className="ml-2 w-1.5 h-1.5 rounded-full bg-jb-accent inline-block align-middle animate-[jb-pulse_1.4s_infinite]" />}
          </span>
          <span className="text-[11.5px] text-jb-ink3 ml-auto">
            {detail.recent.length > 0 ? `last ${detail.recent.length} · click to inspect` : "no labels yet"}
          </span>
        </CardHead>
        <LabelsTable rows={detail.recent} onSelect={setSelected} />
      </Card>

      {selected && <LabelDrawer row={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
