import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@heroui/button";
import { IconFileText, IconLoader2, IconPlus } from "@tabler/icons-react";

import { jobService } from "@/services/job.service";
import type { JDBatch } from "@/types/job.types";
import { T, POLL_INTERVAL, fmtDate, eta } from "./_tokens";
import { Badge, type BadgeStatus, KeyframeStyle, ProgressBar, StatCard } from "./_primitives";

function BatchCard({ batch, onClick }: { batch: JDBatch; onClick: () => void }) {
  const processed = batch.done_count + batch.error_count;
  const pct = batch.total > 0 ? (processed / batch.total) * 100 : 0;

  return (
    <div
      className="jb-card-hover"
      onClick={onClick}
      style={{
        background: T.surface, border: `1px solid ${T.line}`,
        borderRadius: 20, padding: 18, cursor: "pointer",
        transition: "transform 0.14s, box-shadow 0.14s",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <Badge status={batch.status as BadgeStatus} />
        <span style={{ fontFamily: "monospace", fontSize: 11, color: T.ink4 }}>{fmtDate(batch.created_at)}</span>
      </div>
      <div style={{ fontWeight: 700, fontSize: 15, letterSpacing: "-0.01em" }}>Batch #{batch.id}</div>
      <div style={{ fontFamily: "monospace", fontSize: 12, color: T.ink3, marginTop: 2 }}>
        {batch.file_path.split("/").pop()}
      </div>

      <div style={{ marginTop: 14 }}>
        <ProgressBar
          value={processed} total={batch.total}
          running={batch.status === "running"} done={batch.status === "done"}
        />
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginTop: 8 }}>
          <span style={{ color: T.ink3 }}>{batch.done_count} / {batch.total} records</span>
          <span style={{ fontWeight: 600 }}>{pct.toFixed(0)}%</span>
        </div>
      </div>

      <div style={{ height: 1, background: T.line, margin: "14px 0 12px" }} />

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, fontSize: 11.5 }}>
        <div>
          <div style={{ textTransform: "uppercase", letterSpacing: "0.05em", fontWeight: 600, fontSize: 10, color: T.ink4 }}>Fields</div>
          <div style={{ marginTop: 2, color: T.ink2 }}>{batch.fields_config.length} combined</div>
        </div>
        <div>
          <div style={{ textTransform: "uppercase", letterSpacing: "0.05em", fontWeight: 600, fontSize: 10, color: T.ink4 }}>Errors</div>
          <div style={{ marginTop: 2, color: batch.error_count > 0 ? T.danger : T.ink2, fontWeight: 600 }}>
            {batch.error_count}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function JDBatchOverview() {
  const navigate = useNavigate();
  const [batches, setBatches] = useState<JDBatch[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try { setBatches(await jobService.listBatches()); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  useEffect(() => {
    const hasRunning = batches.some((b) => b.status === "running");
    if (!hasRunning) return;
    const id = setInterval(load, POLL_INTERVAL);
    return () => clearInterval(id);
  }, [batches, load]);

  const totalRecords = batches.reduce((s, b) => s + b.total, 0);
  const totalDone    = batches.reduce((s, b) => s + b.done_count, 0);
  const runningCount = batches.filter((b) => b.status === "running").length;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <KeyframeStyle />

      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-end", gap: 16 }}>
        <div>
          <h1 style={{ fontSize: 32, fontWeight: 700, letterSpacing: "-0.025em", margin: 0, color: T.ink }}>
            JD Batch <span style={{ fontStyle: "italic", color: T.accent, fontWeight: 400 }}>Extraction</span>
          </h1>
          <p style={{ margin: "4px 0 0", color: T.ink3, fontSize: 14 }}>
            Normalize scraped job postings into structured JSON using an LLM.
          </p>
        </div>
        <div style={{ marginLeft: "auto" }}>
          <Button color="primary" startContent={<IconPlus size={14} />} onPress={() => navigate("new")}>
            New batch
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12 }}>
        <StatCard label="Active now"         value={runningCount}            unit="running"  accent={runningCount > 0} />
        <StatCard label="Total batches"      value={batches.length}          unit="all-time" />
        <StatCard label="Records extracted"  value={totalDone.toLocaleString()} unit={`/ ${totalRecords.toLocaleString()}`} />
        <StatCard label="Last batch"         value={batches[0] ? `#${batches[0].id}` : "—"} />
      </div>

      {/* Batch grid */}
      {loading ? (
        <div style={{ display: "grid", placeItems: "center", height: 160, color: T.ink3 }}>
          <IconLoader2 size={20} style={{ animation: "jb-spin 0.7s linear infinite" }} />
        </div>
      ) : batches.length === 0 ? (
        <div style={{ display: "grid", placeItems: "center", height: 240, color: T.ink3 }}>
          <div style={{ textAlign: "center" }}>
            <IconFileText size={32} style={{ margin: "0 auto 12px", color: T.ink4 }} />
            <div style={{ fontWeight: 600, marginBottom: 4 }}>No batches yet</div>
            <div style={{ fontSize: 13 }}>Upload a JSONL file to start extraction.</div>
          </div>
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(320px,1fr))", gap: 14 }}>
          {batches.map((b) => (
            <BatchCard key={b.id} batch={b} onClick={() => navigate(String(b.id))} />
          ))}
        </div>
      )}
    </div>
  );
}
