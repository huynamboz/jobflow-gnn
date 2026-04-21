import { useEffect, useRef, useState } from "react";
import { Card, CardBody } from "@heroui/card";
import { Brain, CheckCircle, Clock, XCircle, Zap } from "lucide-react";

import { modelService } from "@/services/model.service";
import type { TrainRun } from "@/types/model.types";

const SENIORITY: Record<number, string> = { 0: "INTERN", 1: "JUNIOR", 2: "MID", 3: "SENIOR", 4: "LEAD", 5: "MANAGER" };

function fmt(n: number | null, decimals = 4) {
  if (n === null || n === undefined) return "—";
  return n.toFixed(decimals);
}

function fmtDuration(secs: number | null) {
  if (!secs) return "—";
  if (secs < 60) return `${secs}s`;
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return s > 0 ? `${m}m ${s}s` : `${m}m`;
}

function fmtDate(iso: string | null) {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString("vi-VN", { day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
}

function StatusBadge({ status }: { status: string }) {
  const cfg: Record<string, { color: string; icon: React.ReactNode }> = {
    completed: { color: "bg-green-100 text-green-700", icon: <CheckCircle className="size-3" /> },
    running:   { color: "bg-blue-100 text-blue-700",  icon: <Clock className="size-3" /> },
    failed:    { color: "bg-red-100 text-red-700",    icon: <XCircle className="size-3" /> },
  };
  const { color, icon } = cfg[status] ?? { color: "bg-gray-100 text-gray-600", icon: null };
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${color}`}>
      {icon}
      {status}
    </span>
  );
}

function MetricCell({ value, decimals = 4 }: { value: number | null; decimals?: number }) {
  if (value === null || value === undefined) return <span className="text-default-300">—</span>;
  return <span>{value.toFixed(decimals)}</span>;
}

interface DetailDrawerProps {
  run: TrainRun;
  onClose: () => void;
}

function DetailDrawer({ run, onClose }: DetailDrawerProps) {
  return (
    <div className="fixed inset-0 z-50 flex justify-end" onClick={onClose}>
      <div
        className="relative h-full w-full max-w-md overflow-y-auto bg-white shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="sticky top-0 flex items-center justify-between border-b border-default-200 bg-white px-5 py-4">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-default-900">{run.version}</span>
            {run.is_active && (
              <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-semibold text-emerald-700">ACTIVE</span>
            )}
            <StatusBadge status={run.status} />
          </div>
          <button onClick={onClose} className="rounded-lg p-1.5 text-default-400 hover:bg-default-100">
            <XCircle className="size-4" />
          </button>
        </div>

        <div className="space-y-5 p-5">
          {/* Metrics */}
          <div>
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-default-400">Evaluation Metrics</p>
            <div className="grid grid-cols-2 gap-2">
              {[
                ["AUC-ROC", fmt(run.auc_roc)],
                ["NDCG@10", fmt(run.ndcg_at_10)],
                ["Recall@5", fmt(run.recall_at_5)],
                ["Recall@10", fmt(run.recall_at_10)],
                ["Precision@5", fmt(run.precision_at_5)],
                ["Precision@10", fmt(run.precision_at_10)],
                ["MRR", fmt(run.mrr)],
                ["Best Epoch", run.best_epoch ?? "—"],
                ["Final Loss", fmt(run.final_loss)],
              ].map(([label, value]) => (
                <div key={String(label)} className="rounded-lg border border-default-100 bg-default-50 px-3 py-2">
                  <p className="text-xs text-default-400">{label}</p>
                  <p className="font-semibold text-default-800">{value}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Model config */}
          <div>
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-default-400">Model Config</p>
            <div className="space-y-1.5 rounded-xl border border-default-100 bg-default-50 px-4 py-3 text-sm">
              {[
                ["Type", run.model_type],
                ["Hidden channels", run.hidden_channels],
                ["Layers", run.num_layers],
                ["Learning rate", run.learning_rate],
              ].map(([k, v]) => (
                <div key={String(k)} className="flex justify-between">
                  <span className="text-default-500">{k}</span>
                  <span className="font-medium text-default-800">{String(v)}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Data stats */}
          <div>
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-default-400">Training Data</p>
            <div className="grid grid-cols-2 gap-2">
              {[
                ["CVs", run.num_cvs],
                ["Jobs", run.num_jobs],
                ["Pairs", run.num_pairs],
                ["Skills", run.num_skills],
              ].map(([label, value]) => (
                <div key={String(label)} className="rounded-lg border border-default-100 bg-default-50 px-3 py-2">
                  <p className="text-xs text-default-400">{label}</p>
                  <p className="font-semibold text-default-800">{value?.toLocaleString()}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Timing */}
          <div>
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-default-400">Timing</p>
            <div className="space-y-1.5 rounded-xl border border-default-100 bg-default-50 px-4 py-3 text-sm">
              {[
                ["Started", fmtDate(run.started_at)],
                ["Completed", fmtDate(run.completed_at)],
                ["Duration", fmtDuration(run.training_duration_seconds)],
              ].map(([k, v]) => (
                <div key={String(k)} className="flex justify-between">
                  <span className="text-default-500">{k}</span>
                  <span className="font-medium text-default-800">{String(v)}</span>
                </div>
              ))}
            </div>
          </div>

          {run.description && (
            <div>
              <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-default-400">Notes</p>
              <p className="rounded-xl border border-default-100 bg-default-50 px-4 py-3 text-sm text-default-700">{run.description}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function ModelsPage() {
  const [runs, setRuns] = useState<TrainRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [activating, setActivating] = useState<number | null>(null);
  const [triggering, setTriggering] = useState(false);
  const [selected, setSelected] = useState<TrainRun | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const load = () => {
    setLoading(true);
    modelService.listRuns()
      .then(setRuns)
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  // Auto-poll while any run is training
  useEffect(() => {
    const hasRunning = runs.some((r) => r.status === "running");
    if (hasRunning) {
      pollRef.current = setInterval(load, 10_000);
    } else {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    }
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [runs]);

  const handleActivate = async (run: TrainRun, e: React.MouseEvent) => {
    e.stopPropagation();
    if (run.is_active || run.status !== "completed") return;
    setActivating(run.id);
    try {
      await modelService.activateRun(run.id);
      load();
    } catch (err) {
      console.error(err);
    } finally {
      setActivating(null);
    }
  };

  const handleTrigger = async () => {
    if (triggering) return;
    setTriggering(true);
    try {
      await modelService.triggerTraining();
      load();
    } catch (err) {
      console.error(err);
    } finally {
      setTriggering(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-default-900">Model Versions</h1>
          <p className="text-default-500">Manage and activate trained GNN models</p>
        </div>
        <button
          onClick={handleTrigger}
          disabled={triggering}
          className="flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
        >
          <Zap className="size-4" />
          {triggering ? "Triggering…" : "Trigger Training"}
        </button>
      </div>

      <Card className="shadow-sm">
        <CardBody className="p-0">
          {loading ? (
            <div className="flex items-center justify-center py-16 text-default-400">Loading…</div>
          ) : runs.length === 0 ? (
            <div className="flex flex-col items-center justify-center gap-2 py-16 text-default-400">
              <Brain className="size-8" />
              <p>No training runs yet.</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b border-default-100 bg-default-50 text-xs font-semibold uppercase tracking-wide text-default-500">
                  <tr>
                    <th className="px-4 py-3 text-left">Version</th>
                    <th className="px-4 py-3 text-left">Status</th>
                    <th className="px-4 py-3 text-right">AUC-ROC</th>
                    <th className="px-4 py-3 text-right">NDCG@10</th>
                    <th className="px-4 py-3 text-right">R@10</th>
                    <th className="px-4 py-3 text-right">Epoch</th>
                    <th className="px-4 py-3 text-right">Duration</th>
                    <th className="px-4 py-3 text-left">Started</th>
                    <th className="px-4 py-3 text-center">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-default-100">
                  {runs.map((run) => (
                    <tr
                      key={run.id}
                      onClick={() => setSelected(run)}
                      className="cursor-pointer transition-colors hover:bg-default-50"
                    >
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-default-800">{run.version}</span>
                          {run.is_active && (
                            <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-semibold text-emerald-700">
                              ACTIVE
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-3"><StatusBadge status={run.status} /></td>
                      <td className="px-4 py-3 text-right font-mono"><MetricCell value={run.auc_roc} /></td>
                      <td className="px-4 py-3 text-right font-mono"><MetricCell value={run.ndcg_at_10} /></td>
                      <td className="px-4 py-3 text-right font-mono"><MetricCell value={run.recall_at_10} /></td>
                      <td className="px-4 py-3 text-right text-default-600">{run.best_epoch ?? "—"}</td>
                      <td className="px-4 py-3 text-right text-default-600">{fmtDuration(run.training_duration_seconds)}</td>
                      <td className="px-4 py-3 text-xs text-default-500">{fmtDate(run.started_at)}</td>
                      <td className="px-4 py-3 text-center">
                        {run.is_active ? (
                          <span className="text-xs text-default-400">Active</span>
                        ) : run.status === "completed" ? (
                          <button
                            onClick={(e) => handleActivate(run, e)}
                            disabled={activating === run.id}
                            className="rounded-lg border border-blue-200 bg-blue-50 px-3 py-1 text-xs font-medium text-blue-700 hover:bg-blue-100 disabled:opacity-50"
                          >
                            {activating === run.id ? "…" : "Activate"}
                          </button>
                        ) : (
                          <span className="text-xs text-default-300">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardBody>
      </Card>

      {selected && <DetailDrawer run={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
