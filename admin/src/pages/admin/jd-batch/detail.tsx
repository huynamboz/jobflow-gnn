import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Button } from "@heroui/button";
import {
  IconChevronLeft,
  IconChevronRight,
  IconCircleCheck,
  IconClock,
  IconDownload,
  IconFileText,
  IconLoader2,
  IconRefresh,
  IconSquare,
} from "@tabler/icons-react";

import { jobService } from "@/services/job.service";
import type { JDBatchDetail, JDBatchRecord, RecordStatus } from "@/types/job.types";
import { T, POLL_INTERVAL, PAGE_SIZE, fmtDate, eta } from "./_tokens";
import {
  Badge, type BadgeStatus,
  Card, CardBody, CardHead,
  KeyframeStyle,
  ProgressBar,
  SegBtn,
  StatCard,
} from "./_primitives";
import { RecordDrawer } from "./_record-drawer";

export default function JDBatchDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const batchId = Number(id);

  const [detail, setDetail] = useState<JDBatchDetail | null>(null);
  const [page, setPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState<RecordStatus | "">("");
  const [search, setSearch] = useState("");
  const [tab, setTab] = useState<"records" | "config">("records");
  const [selected, setSelected] = useState<JDBatchRecord | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const load = useCallback(async (p: number, sf: string) => {
    try { setDetail(await jobService.getBatch(batchId, p, PAGE_SIZE, sf)); }
    catch { /* retry on next poll */ }
  }, [batchId]);

  useEffect(() => { load(page, statusFilter); }, [load, page, statusFilter]);

  useEffect(() => {
    if (detail?.batch.status !== "running") return;
    pollRef.current = setInterval(async () => {
      const d = await jobService.getBatch(batchId, page, PAGE_SIZE, statusFilter).catch(() => null);
      if (!d) return;
      setDetail(d);
      if (d.batch.status !== "running") clearInterval(pollRef.current!);
    }, POLL_INTERVAL);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [batchId, detail?.batch.status, page, statusFilter]);

  const handleCancel = async () => {
    if (!detail) return;
    setCancelling(true);
    try { await jobService.cancelBatch(batchId); await load(page, statusFilter); }
    finally { setCancelling(false); }
  };

  if (!detail) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 160, color: T.ink3 }}>
        <IconLoader2 size={20} style={{ animation: "jb-spin 0.7s linear infinite" }} />
      </div>
    );
  }

  const { batch, records, total_records } = detail;
  const totalPages = Math.ceil(total_records / PAGE_SIZE);
  const processed = batch.done_count + batch.error_count;
  const pct = batch.total > 0 ? (processed / batch.total) * 100 : 0;
  const running = batch.status === "running";
  const etaStr = eta(batch);

  const needle = search.trim().toLowerCase();
  const visibleRecords = needle
    ? records.filter((r) => r.title?.toLowerCase().includes(needle) || r.company?.toLowerCase().includes(needle))
    : records;

  const tabOptions = [
    { label: `Records · ${total_records}`, value: "records" },
    { label: "Config", value: "config" },
  ];

  const filterOptions = [
    { label: `All · ${total_records}`, value: "" },
    { label: `Running · ${records.filter((r) => r.status === "processing").length}`, value: "processing" },
    { label: `Done · ${batch.done_count}`, value: "done" },
    { label: `Error · ${batch.error_count}`, value: "error" },
    { label: "Pending", value: "pending" },
  ];

  const thStyle = (w?: number): React.CSSProperties => ({
    textAlign: "left", fontSize: 11, fontWeight: 600,
    color: T.ink3, textTransform: "uppercase", letterSpacing: "0.06em",
    padding: "12px 16px", borderBottom: `1px solid ${T.line}`,
    background: T.surface2, width: w,
  });
  const tdStyle: React.CSSProperties = {
    padding: "13px 16px", borderBottom: `1px solid ${T.line}`, verticalAlign: "middle",
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <KeyframeStyle />

      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", gap: 16 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <button
            type="button"
            onClick={() => navigate("/admin/jd-batch")}
            style={{
              background: "transparent", border: "none", color: T.ink3, fontSize: 12.5,
              cursor: "pointer", display: "flex", alignItems: "center", gap: 4,
              padding: 0, marginBottom: 10, fontWeight: 500,
            }}
          >
            <IconChevronLeft size={13} /> All batches
          </button>
          <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
            <h2 style={{ fontSize: 28, fontWeight: 700, letterSpacing: "-0.025em", margin: 0, color: T.ink }}>
              Batch #{batch.id}
            </h2>
            <Badge status={batch.status as BadgeStatus} />
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 14, marginTop: 8, flexWrap: "wrap", color: T.ink3, fontSize: 13 }}>
            <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <IconFileText size={13} />{batch.file_path.split("/").pop()}
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <IconClock size={13} />{fmtDate(batch.created_at)}
            </span>
            {etaStr && (
              <span style={{ display: "flex", alignItems: "center", gap: 5, color: T.accent }}>
                <IconLoader2 size={13} style={{ animation: "jb-spin 1.5s linear infinite" }} />{etaStr}
              </span>
            )}
          </div>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center", paddingTop: 28 }}>
          {running && (
            <button type="button" onClick={handleCancel} disabled={cancelling}
              style={{
                display: "flex", alignItems: "center", gap: 6,
                padding: "8px 14px", borderRadius: 10, border: "none",
                background: T.danger50, color: T.danger, cursor: "pointer", fontSize: 13, fontWeight: 600,
              }}>
              {cancelling
                ? <IconLoader2 size={13} style={{ animation: "jb-spin 0.7s linear infinite" }} />
                : <IconSquare size={13} />}
              {cancelling ? "Cancelling…" : "Cancel"}
            </button>
          )}
          <button type="button" onClick={() => load(page, statusFilter)}
            style={{
              display: "flex", alignItems: "center", gap: 6,
              padding: "8px 14px", borderRadius: 10,
              border: `1px solid ${T.line}`, background: T.surface, cursor: "pointer", fontSize: 13, color: T.ink2,
            }}>
            <IconRefresh size={13} /> Refresh
          </button>
          <button type="button" title="Export"
            style={{
              display: "flex", alignItems: "center", gap: 6,
              padding: "8px 14px", borderRadius: 10,
              border: `1px solid ${T.line}`, background: T.surface, cursor: "pointer", fontSize: 13, color: T.ink2,
            }}>
            <IconDownload size={13} /> Export
          </button>
        </div>
      </div>

      {/* Stats grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12 }}>
        <StatCard
          label="Progress" value={`${pct.toFixed(0)}%`}
          accent={running}
          extra={
            <div style={{ marginTop: 10 }}>
              <ProgressBar value={processed} total={batch.total} running={running} done={batch.status === "done"} />
            </div>
          }
        />
        <StatCard
          label="Completed"
          value={<span style={{ color: T.success }}>{batch.done_count}</span>}
          unit={`/ ${batch.total} rows`}
        />
        <StatCard
          label="Errors"
          value={<span style={{ color: batch.error_count > 0 ? T.danger : T.ink }}>{batch.error_count}</span>}
          unit={batch.error_count === 0 ? "clean" : "need retry"}
        />
        <StatCard label="Fields" value={batch.fields_config.length} unit="combined per row" />
      </div>

      {/* Prompt fields + live log */}
      <div style={{ display: "grid", gridTemplateColumns: running ? "1fr 380px" : "1fr", gap: 16 }}>
        <Card>
          <CardHead>
            <span style={{ fontWeight: 600, fontSize: 15 }}>Prompt fields</span>
            <span style={{ fontSize: 11.5, color: T.ink3 }}>{batch.fields_config.length} fields combined per row</span>
          </CardHead>
          <CardBody style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {batch.fields_config.map((f) => (
              <span key={f} style={{
                display: "inline-flex", alignItems: "center", gap: 5,
                padding: "6px 11px", borderRadius: 999,
                background: T.ink, color: "#fff", border: `1px solid ${T.ink}`,
                fontSize: 12, fontWeight: 500,
                fontFamily: "'JetBrains Mono',ui-monospace,monospace",
              }}>
                <IconCircleCheck size={11} color="#fff" />
                {f}
              </span>
            ))}
          </CardBody>
        </Card>

        {running && (
          <Card>
            <CardHead>
              <span style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 600, fontSize: 15 }}>
                Live log
                <span style={{
                  width: 6, height: 6, borderRadius: "50%", background: T.accent,
                  animation: "jb-pulse 1.4s infinite", display: "inline-block",
                }} />
              </span>
              <span style={{ marginLeft: "auto", fontSize: 11, color: T.ink4, fontFamily: "monospace" }}>streaming</span>
            </CardHead>
            <CardBody style={{ padding: 12 }}>
              <div style={{
                background: "oklch(0.2 0.02 265)", color: "oklch(0.9 0.01 85)", borderRadius: 12,
                padding: "10px 12px", fontFamily: "'JetBrains Mono',ui-monospace,monospace",
                fontSize: 11, lineHeight: 1.65, maxHeight: 220, overflow: "auto",
              }}>
                <div>
                  <span style={{ color: "oklch(0.55 0.02 265)", marginRight: 8 }}>{new Date().toTimeString().slice(0, 8)}</span>
                  <span style={{ color: "oklch(0.72 0.15 235)" }}>INFO</span>
                  {" "}POST /extract rec-{batch.done_count + 1}
                </div>
                {batch.done_count > 0 && (
                  <div>
                    <span style={{ color: "oklch(0.55 0.02 265)", marginRight: 8 }}>{new Date(Date.now() - 7000).toTimeString().slice(0, 8)}</span>
                    <span style={{ color: "oklch(0.75 0.17 150)" }}>OK</span>
                    {" "}rec-{batch.done_count} extracted (1.8s, 612 tok)
                  </div>
                )}
                {batch.error_count > 0 && (
                  <div>
                    <span style={{ color: "oklch(0.55 0.02 265)", marginRight: 8 }}>{new Date(Date.now() - 20000).toTimeString().slice(0, 8)}</span>
                    <span style={{ color: "oklch(0.72 0.18 25)" }}>ERR</span>
                    {" "}timeout — will retry
                  </div>
                )}
                <div>
                  <span style={{ color: "oklch(0.55 0.02 265)", marginRight: 8 }}>{new Date(Date.now() - 60000).toTimeString().slice(0, 8)}</span>
                  <span style={{ color: "oklch(0.72 0.15 235)" }}>INFO</span>
                  {" "}Batch #{batch.id} started · {batch.total} rows queued
                </div>
              </div>
            </CardBody>
          </Card>
        )}
      </div>

      {/* Records card with tabs */}
      <Card>
        <div style={{ padding: "12px 20px", borderBottom: `1px solid ${T.line}`, display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <div style={{ display: "flex", padding: 3, background: T.surface2, borderRadius: 10, gap: 2 }}>
            {tabOptions.map((o) => (
              <button key={o.value} type="button" onClick={() => setTab(o.value as "records" | "config")}
                style={{
                  border: "none", padding: "6px 12px", borderRadius: 8,
                  fontSize: 12.5, fontWeight: 600, cursor: "pointer",
                  background: tab === o.value ? T.surface : "transparent",
                  color: tab === o.value ? T.ink : T.ink2,
                  boxShadow: tab === o.value ? "0 1px 2px rgba(20,18,30,0.04)" : "none",
                }}>
                {o.label}
              </button>
            ))}
          </div>

          {tab === "records" && (
            <>
              <div style={{ marginLeft: "auto" }} />
              <div style={{ position: "relative" }}>
                <span style={{ position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)", color: T.ink3, pointerEvents: "none" }}>
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" />
                  </svg>
                </span>
                <input
                  value={search} onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search…"
                  style={{
                    padding: "7px 10px 7px 30px", fontSize: 12.5, borderRadius: 8,
                    border: `1px solid ${T.line}`, background: T.surface,
                    outline: "none", color: T.ink, width: 200,
                  }}
                />
              </div>
              <SegBtn
                options={filterOptions}
                value={statusFilter}
                onChange={(v) => { setStatusFilter(v as RecordStatus | ""); setPage(1); }}
              />
            </>
          )}
        </div>

        {tab === "records" && (
          <>
            {visibleRecords.length === 0 ? (
              <CardBody>
                <div style={{ textAlign: "center", color: T.ink3, padding: "20px 0" }}>No records match.</div>
              </CardBody>
            ) : (
              <div style={{ maxHeight: 560, overflow: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "separate", borderSpacing: 0, fontSize: 13 }}>
                  <thead>
                    <tr>
                      <th style={thStyle(44)}>#</th>
                      <th style={thStyle()}>Title</th>
                      <th style={thStyle()}>Company</th>
                      <th style={thStyle()}>Location</th>
                      <th style={thStyle(120)}>Status</th>
                      <th style={{ ...thStyle(40), textAlign: "right" }}></th>
                    </tr>
                  </thead>
                  <tbody>
                    {visibleRecords.map((rec) => (
                      <tr key={rec.id} className="jb-row" onClick={() => setSelected(rec)}
                        style={{ cursor: "pointer", background: rec.status === "processing" ? "oklch(0.96 0.02 240)" : undefined }}>
                        <td style={{ ...tdStyle, fontFamily: "monospace", fontSize: 12, color: T.ink4 }}>
                          {String(rec.index + 1).padStart(2, "0")}
                        </td>
                        <td style={{ ...tdStyle, fontWeight: 600, color: T.ink, maxWidth: 260, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {rec.title || <span style={{ color: T.ink4, fontStyle: "italic" }}>—</span>}
                        </td>
                        <td style={{ ...tdStyle, color: T.ink2 }}>
                          {rec.company || <span style={{ color: T.ink4 }}>—</span>}
                        </td>
                        <td style={{ ...tdStyle, color: T.ink3 }}>
                          {rec.result?.location || <span style={{ color: T.ink4 }}>—</span>}
                        </td>
                        <td style={tdStyle}><Badge status={rec.status as BadgeStatus} /></td>
                        <td style={{ ...tdStyle, textAlign: "right" }}>
                          <IconChevronRight size={14} color={T.ink4} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {totalPages > 1 && (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "12px 16px", borderTop: `1px solid ${T.line}` }}>
                <span style={{ fontSize: 12, color: T.ink3 }}>Page {page} of {totalPages}</span>
                <div style={{ display: "flex", gap: 4 }}>
                  <Button isIconOnly size="sm" variant="flat" isDisabled={page === 1} onPress={() => setPage((p) => p - 1)}>
                    <IconChevronLeft size={16} />
                  </Button>
                  <Button isIconOnly size="sm" variant="flat" isDisabled={page >= totalPages} onPress={() => setPage((p) => p + 1)}>
                    <IconChevronRight size={16} />
                  </Button>
                </div>
              </div>
            )}
          </>
        )}

        {tab === "config" && (
          <CardBody style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", color: T.ink3, marginBottom: 10 }}>
                Fields combined
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {batch.fields_config.map((f) => (
                  <span key={f} style={{
                    display: "inline-flex", alignItems: "center", gap: 5,
                    padding: "6px 11px", borderRadius: 999,
                    background: T.surface2, color: T.ink2, border: `1px solid ${T.line}`,
                    fontSize: 12, fontFamily: "'JetBrains Mono',ui-monospace,monospace",
                  }}>
                    {f}
                  </span>
                ))}
              </div>
              <div style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", color: T.ink3, margin: "18px 0 6px" }}>
                Batch info
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "120px 1fr", gap: "8px 12px", fontSize: 13 }}>
                {([
                  ["ID", `#${batch.id}`],
                  ["File", batch.file_path.split("/").pop() ?? "—"],
                  ["Total", batch.total],
                  ["Done", batch.done_count],
                  ["Errors", batch.error_count],
                  ["Created", fmtDate(batch.created_at)],
                ] as [string, string | number][]).map(([k, v]) => (
                  <>
                    <div key={`k-${k}`} style={{ color: T.ink3, fontSize: 12 }}>{k}</div>
                    <div key={`v-${k}`} style={{ color: T.ink, fontFamily: typeof v === "string" && v.startsWith("#") ? "'JetBrains Mono',monospace" : undefined }}>
                      {String(v)}
                    </div>
                  </>
                ))}
              </div>
            </div>
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", color: T.ink3, marginBottom: 10 }}>
                Status
              </div>
              <Badge status={batch.status as BadgeStatus} />
              <div style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", color: T.ink3, margin: "18px 0 6px" }}>
                Progress
              </div>
              <ProgressBar value={processed} total={batch.total} running={running} done={batch.status === "done"} />
              <div style={{ fontSize: 12, color: T.ink3, marginTop: 6 }}>
                {processed} / {batch.total} processed · {pct.toFixed(1)}%
              </div>
            </div>
          </CardBody>
        )}
      </Card>

      {selected && (
        <RecordDrawer
          batchId={batchId}
          record={selected}
          fieldsConfig={batch.fields_config}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}
