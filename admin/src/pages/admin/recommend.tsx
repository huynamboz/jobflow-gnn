import { useRef, useState } from "react";
import { ExternalLink, Loader2, Search, Upload, X } from "lucide-react";

import { matchingService } from "@/services/matching.service";
import type { JobMatchResult } from "@/types/matching.types";

const JOB_TYPE_LABEL: Record<string, string> = {
  "full-time": "Full-time", "part-time": "Part-time",
  remote: "Remote", hybrid: "Hybrid", "on-site": "On-site",
};

function fmtSalary(min: number, max: number) {
  if (!min && !max) return null;
  const fmt = (n: number) => n >= 1_000_000 ? `${(n / 1_000_000).toFixed(0)}M` : `${(n / 1000).toFixed(0)}K`;
  if (min && max) return `${fmt(min)}–${fmt(max)}`;
  if (min) return `≥ ${fmt(min)}`;
  return `≤ ${fmt(max)}`;
}

function ScoreBar({ score, eligible }: { score: number; eligible: boolean }) {
  const pct = Math.round(score * 100);
  const color = eligible ? "bg-emerald-500" : score >= 0.5 ? "bg-amber-400" : "bg-default-300";
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-20 rounded-full bg-default-100 overflow-hidden">
        <div className={`h-full rounded-full ${color} transition-all`} style={{ width: `${pct}%` }} />
      </div>
      <span className={`text-xs font-semibold tabular-nums ${eligible ? "text-emerald-600" : "text-default-500"}`}>
        {pct}%
      </span>
      {eligible && <span className="text-[10px] font-medium text-emerald-600 border border-emerald-200 bg-emerald-50 rounded px-1">Match</span>}
    </div>
  );
}

function JobCard({ job }: { job: JobMatchResult }) {
  const salary = fmtSalary(job.salary_min, job.salary_max);
  return (
    <div className="rounded-2xl border border-default-200 bg-white p-5 space-y-3 hover:border-default-300 transition-colors">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h3 className="font-semibold text-default-900 truncate">{job.title || `Job #${job.job_id}`}</h3>
          <div className="mt-0.5 flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-default-500">
            {job.company_name && <span>{job.company_name}</span>}
            {job.location && <><span>·</span><span>{job.location}</span></>}
            {job.job_type && <><span>·</span><span>{JOB_TYPE_LABEL[job.job_type] ?? job.job_type}</span></>}
            {salary && <><span>·</span><span>{salary}</span></>}
          </div>
        </div>
        {job.source_url && (
          <a href={job.source_url} target="_blank" rel="noreferrer"
            className="shrink-0 rounded-lg border border-default-200 p-1.5 text-default-400 hover:text-blue-600 hover:border-blue-300 transition-colors">
            <ExternalLink className="size-3.5" />
          </a>
        )}
      </div>

      <ScoreBar score={job.score} eligible={job.eligible} />

      {job.matched_skills.length > 0 && (
        <div>
          <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wide text-emerald-600">
            Matched ({job.matched_skills.length})
          </p>
          <div className="flex flex-wrap gap-1">
            {job.matched_skills.map((s) => (
              <span key={s} className="rounded-md border border-emerald-200 bg-emerald-50 px-1.5 py-0.5 text-[11px] text-emerald-700">
                {s}
              </span>
            ))}
          </div>
        </div>
      )}

      {job.missing_skills.length > 0 && (
        <div>
          <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wide text-red-500">
            Missing ({job.missing_skills.length})
          </p>
          <div className="flex flex-wrap gap-1">
            {job.missing_skills.map((s) => (
              <span key={s} className="rounded-md border border-red-200 bg-red-50 px-1.5 py-0.5 text-[11px] text-red-600">
                {s}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function RecommendPage() {
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [topK, setTopK] = useState(10);
  const [results, setResults] = useState<JobMatchResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim() && !file) return;
    setLoading(true);
    setError(null);
    try {
      const res = file
        ? await matchingService.matchFile(file, topK)
        : await matchingService.matchText(text, topK);
      setResults(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to get recommendations.");
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    if (f) setText("");
  };

  const clearFile = () => {
    setFile(null);
    if (fileRef.current) fileRef.current.value = "";
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-default-900">Job Recommendations</h1>
        <p className="text-default-500">Paste CV text or upload a file to find matching jobs</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4 rounded-2xl border border-default-200 bg-white p-5">
        {/* File upload area */}
        <div
          className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-default-200 bg-default-50 px-4 py-6 text-center hover:border-blue-300 hover:bg-blue-50/30 transition-colors"
          onClick={() => fileRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            const f = e.dataTransfer.files[0];
            if (f) { setFile(f); setText(""); }
          }}
        >
          <Upload className="size-5 text-default-400 mb-1" />
          <p className="text-sm text-default-600">
            {file ? (
              <span className="flex items-center gap-1.5">
                <span className="font-medium text-blue-600">{file.name}</span>
                <button type="button" onClick={(e) => { e.stopPropagation(); clearFile(); }}>
                  <X className="size-3.5 text-default-400 hover:text-red-500" />
                </button>
              </span>
            ) : "Drop PDF/DOCX/TXT or click to browse"}
          </p>
          <input ref={fileRef} type="file" accept=".pdf,.docx,.txt" className="hidden" onChange={handleFileChange} />
        </div>

        <div className="flex items-center gap-3 text-xs text-default-400">
          <div className="h-px flex-1 bg-default-200" />
          <span>or paste CV text</span>
          <div className="h-px flex-1 bg-default-200" />
        </div>

        <textarea
          rows={8}
          placeholder="Paste your CV / resume text here..."
          value={text}
          onChange={(e) => { setText(e.target.value); if (e.target.value) clearFile(); }}
          className="w-full resize-none rounded-xl border border-default-200 bg-default-50 px-4 py-3 text-sm text-default-800 placeholder:text-default-400 focus:border-blue-300 focus:bg-white focus:outline-none transition-colors"
          disabled={!!file}
        />

        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <label className="text-sm text-default-600">Top</label>
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="h-8 rounded-lg border border-default-200 bg-white px-2 text-sm text-default-700 outline-none focus:border-blue-300"
            >
              {[5, 10, 20, 50].map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
            <span className="text-sm text-default-600">results</span>
          </div>

          <button
            type="submit"
            disabled={loading || (!text.trim() && !file)}
            className="flex items-center gap-2 rounded-xl bg-blue-600 px-5 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {loading ? <Loader2 className="size-4 animate-spin" /> : <Search className="size-4" />}
            {loading ? "Matching…" : "Find Jobs"}
          </button>
        </div>
      </form>

      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>
      )}

      {results !== null && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-default-700">
              {results.length} job{results.length !== 1 ? "s" : ""} found
              {results.filter((r) => r.eligible).length > 0 && (
                <span className="ml-2 text-emerald-600">
                  · {results.filter((r) => r.eligible).length} strong match{results.filter((r) => r.eligible).length !== 1 ? "es" : ""}
                </span>
              )}
            </p>
          </div>

          {results.length === 0 ? (
            <div className="rounded-2xl border border-default-200 bg-white py-16 text-center text-default-400">
              <Search className="size-8 mx-auto mb-2" />
              <p>No matching jobs found. Try a more detailed CV.</p>
            </div>
          ) : (
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
              {results.map((job) => <JobCard key={job.job_id} job={job} />)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
