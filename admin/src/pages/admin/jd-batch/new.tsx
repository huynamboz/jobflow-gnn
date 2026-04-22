import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@heroui/button";
import { Input } from "@heroui/input";
import {
  IconChevronLeft,
  IconChevronRight,
  IconCircleCheck,
  IconFileText,
  IconLoader2,
  IconSparkles,
  IconUpload,
  IconX,
} from "@tabler/icons-react";

import { jobService } from "@/services/job.service";
import { T, LIMIT_OPTIONS } from "./_tokens";
import { Badge, Card, CardBody, CardHead, FieldChip, KeyframeStyle } from "./_primitives";

export default function JDBatchNew() {
  const navigate = useNavigate();
  const [step, setStep] = useState<0 | 1>(0);
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState<{
    total: number; fields: string[];
    sample: Record<string, unknown>[]; filename: string;
  } | null>(null);
  const [selectedFields, setSelectedFields] = useState<string[]>([]);
  const [limit, setLimit] = useState<number | null>(null);
  const [customLimit, setCustomLimit] = useState("");
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const loadFile = async (f: File) => {
    setFile(f);
    setLoading(true);
    setError("");
    setPreview(null);
    setSelectedFields([]);
    try {
      const data = await jobService.previewBatch(f);
      setPreview(data);
      const prefer = ["title", "description", "seniority_hint", "raw_skills"];
      setSelectedFields(data.fields.filter((x) => prefer.includes(x)));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to read file");
    } finally {
      setLoading(false);
    }
  };

  const toggleField = (f: string) =>
    setSelectedFields((prev) => prev.includes(f) ? prev.filter((x) => x !== f) : [...prev, f]);

  const effectiveLimit =
    limit === null ? null : limit === -1 ? (parseInt(customLimit) || null) : limit;

  const previewText = preview && selectedFields.length > 0
    ? selectedFields.map((f) => `${f.toUpperCase()}\n${preview.sample[0]?.[f] ?? "—"}`).join("\n\n---\n\n")
    : "";

  const handleStart = async () => {
    if (!file || !preview || selectedFields.length === 0) return;
    setStarting(true);
    setError("");
    try {
      const batch = await jobService.createBatch(file, selectedFields, effectiveLimit);
      navigate(`/admin/jd-batch/${batch.id}`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to start batch");
      setStarting(false);
    }
  };

  const steps = ["Upload", "Fields & prompt", "Run"];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <KeyframeStyle />

      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-end", gap: 16 }}>
        <div>
          <h1 style={{ fontSize: 32, fontWeight: 700, letterSpacing: "-0.025em", margin: 0, color: T.ink }}>
            New <span style={{ fontStyle: "italic", color: T.accent, fontWeight: 400 }}>Batch</span>
          </h1>
          <p style={{ margin: "4px 0 0", color: T.ink3, fontSize: 14 }}>
            Upload a JSONL file and configure the extraction prompt.
          </p>
        </div>
        <div style={{ marginLeft: "auto" }}>
          <button
            type="button"
            onClick={() => navigate("/admin/jd-batch")}
            style={{
              display: "flex", alignItems: "center", gap: 6,
              padding: "8px 14px", borderRadius: 10,
              border: `1px solid ${T.line}`, background: T.surface,
              cursor: "pointer", fontSize: 13, color: T.ink2, fontWeight: 500,
            }}
          >
            <IconChevronLeft size={14} /> All batches
          </button>
        </div>
      </div>

      <div style={{ maxWidth: 900 }}>
        {/* Wizard stepper */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 28 }}>
          {steps.map((s, i) => (
            <div key={s} style={{ display: "contents" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{
                  width: 22, height: 22, borderRadius: "50%",
                  background: i <= step ? T.accent : T.surface3,
                  color: i <= step ? "#fff" : T.ink3,
                  display: "grid", placeItems: "center",
                  fontSize: 11, fontWeight: 700,
                }}>
                  {i < step ? <IconCircleCheck size={12} /> : i + 1}
                </div>
                <span style={{ fontSize: 12.5, fontWeight: 600, color: i === step ? T.ink : T.ink3 }}>{s}</span>
              </div>
              {i < steps.length - 1 && (
                <div style={{ flex: 1, height: 1.5, background: i < step ? T.accent : T.surface3 }} />
              )}
            </div>
          ))}
        </div>

        {/* Step 0 — Upload */}
        {step === 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div>
              <h2 style={{ fontSize: 28, fontWeight: 700, letterSpacing: "-0.025em", margin: 0, color: T.ink }}>
                Upload a <span style={{ fontStyle: "italic", color: T.accent }}>JSONL</span> file
              </h2>
              <p style={{ marginTop: 4, color: T.ink3, fontSize: 14 }}>
                One JSON object per line. Each object becomes one LLM extraction job.
              </p>
            </div>

            {!file ? (
              <label
                style={{
                  display: "block",
                  border: `2px dashed ${dragging ? T.accent : T.lineStrong}`,
                  borderRadius: 24, padding: "48px 24px", textAlign: "center",
                  background: dragging ? T.accent50 : `color-mix(in oklch,${T.surface2} 70%,transparent)`,
                  cursor: "pointer",
                }}
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => {
                  e.preventDefault(); setDragging(false);
                  const f = e.dataTransfer.files[0]; if (f) loadFile(f);
                }}
              >
                <input
                  ref={inputRef} type="file" accept=".jsonl,.json"
                  style={{ display: "none" }}
                  onChange={(e) => { const f = e.target.files?.[0]; if (f) loadFile(f); }}
                />
                <div style={{
                  width: 56, height: 56, borderRadius: 16, background: T.surface,
                  margin: "0 auto 16px", display: "grid", placeItems: "center",
                  color: T.accent, boxShadow: "0 1px 2px rgba(20,18,30,0.04)",
                }}>
                  <IconUpload size={22} />
                </div>
                <div style={{ fontSize: 18, fontWeight: 700, letterSpacing: "-0.01em", margin: "0 0 4px", color: T.ink }}>
                  Drop your .jsonl file here
                </div>
                <p style={{ margin: 0, color: T.ink3, fontSize: 13 }}>
                  or{" "}
                  <span style={{ color: T.accent, fontWeight: 600, textDecoration: "underline" }}>
                    browse from your computer
                  </span>{" "}
                  — max 50 MB
                </p>
              </label>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                <Card>
                  <CardBody style={{ display: "flex", alignItems: "center", gap: 14 }}>
                    {loading ? (
                      <div style={{ display: "flex", alignItems: "center", gap: 12, color: T.ink3 }}>
                        <IconLoader2 size={20} style={{ animation: "jb-spin 0.7s linear infinite" }} />
                        <span style={{ fontSize: 13 }}>Reading file…</span>
                      </div>
                    ) : (
                      <>
                        <div style={{
                          width: 44, height: 44, borderRadius: 12,
                          background: T.accent50, color: T.accent,
                          display: "grid", placeItems: "center", flexShrink: 0,
                        }}>
                          <IconFileText size={20} />
                        </div>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ fontWeight: 700 }}>{file.name}</div>
                          <div style={{ fontSize: 12.5, color: T.ink3, marginTop: 2 }}>
                            {(file.size / 1024).toFixed(1)} KB
                            {preview && (
                              <> · <strong style={{ color: T.ink2 }}>{preview.total}</strong> rows detected</>
                            )}
                          </div>
                        </div>
                        <button
                          type="button"
                          onClick={() => { setFile(null); setPreview(null); setSelectedFields([]); }}
                          style={{
                            display: "flex", alignItems: "center", gap: 6,
                            padding: "6px 10px", borderRadius: 8,
                            border: `1px solid ${T.line}`, background: T.surface2,
                            cursor: "pointer", fontSize: 12.5, color: T.ink2,
                          }}
                        >
                          <IconX size={14} /> Remove
                        </button>
                      </>
                    )}
                  </CardBody>
                </Card>

                {preview && (
                  <Card>
                    <CardHead>
                      <span style={{ fontWeight: 600, fontSize: 15 }}>Preview rows</span>
                      <Badge status="pending" label={`${preview.total} total`} />
                    </CardHead>
                    <CardBody>
                      {preview.sample.slice(0, 3).map((row, i) => (
                        <div key={i} style={{
                          display: "grid", gridTemplateColumns: "auto 1fr auto",
                          gap: 12, alignItems: "center",
                          padding: 12, marginBottom: 8, borderRadius: 12,
                          border: `1px dashed ${T.lineStrong}`,
                          background: `color-mix(in oklch,${T.surface2} 50%,transparent)`,
                        }}>
                          <span style={{ fontFamily: "monospace", fontSize: 12, color: T.ink4 }}>#{i + 1}</span>
                          <div style={{ minWidth: 0 }}>
                            <div style={{ fontWeight: 600, fontSize: 13.5 }}>
                              {String(row.title ?? row.job_title ?? "—")}
                            </div>
                            <div style={{
                              fontFamily: "monospace", fontSize: 11, color: T.ink3,
                              overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", marginTop: 2,
                            }}>
                              {JSON.stringify(row).slice(0, 80)}…
                            </div>
                          </div>
                          <Badge status="pending" label="valid" />
                        </div>
                      ))}
                      {preview.total > 3 && (
                        <div style={{ textAlign: "center", fontSize: 12, color: T.ink3, marginTop: 8 }}>
                          + {preview.total - 3} more rows
                        </div>
                      )}
                    </CardBody>
                  </Card>
                )}

                {error && <p style={{ color: T.danger, fontSize: 13 }}>{error}</p>}

                {preview && (
                  <div style={{ display: "flex", justifyContent: "flex-end" }}>
                    <Button color="primary" endContent={<IconChevronRight size={16} />} onPress={() => setStep(1)}>
                      Continue — pick fields
                    </Button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Step 1 — Fields & config */}
        {step === 1 && preview && (
          <div>
            <div style={{ marginBottom: 24 }}>
              <h2 style={{ fontSize: 28, fontWeight: 700, letterSpacing: "-0.025em", margin: "0 0 4px", color: T.ink }}>
                Build the <span style={{ fontStyle: "italic", color: T.accent }}>extraction prompt</span>
              </h2>
              <p style={{ color: T.ink3, fontSize: 14, margin: 0 }}>
                Click fields to include them in the prompt sent to the LLM per row.
              </p>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, alignItems: "start" }}>
              {/* Left — field picker */}
              <Card>
                <CardHead>
                  <span style={{ fontWeight: 600, fontSize: 15 }}>Fields from row</span>
                  <span style={{ fontSize: 12, color: T.ink3 }}>
                    {selectedFields.length} of {preview.fields.length} selected
                  </span>
                  <div style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
                    {([["All", preview.fields], ["None", []]] as [string, string[]][]).map(([lbl, val]) => (
                      <button key={lbl} type="button" onClick={() => setSelectedFields(val)}
                        style={{
                          padding: "4px 10px", borderRadius: 8,
                          border: `1px solid ${T.line}`, background: T.surface2,
                          cursor: "pointer", fontSize: 12, color: T.ink2,
                        }}>
                        {lbl}
                      </button>
                    ))}
                  </div>
                </CardHead>
                <CardBody>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                    {preview.fields.map((f) => (
                      <FieldChip key={f} label={f} on={selectedFields.includes(f)} onClick={() => toggleField(f)} />
                    ))}
                  </div>
                </CardBody>
              </Card>

              {/* Right — preview + config */}
              <div style={{ display: "flex", flexDirection: "column", gap: 16, position: "sticky", top: 90 }}>
                <Card>
                  <CardHead>
                    <span style={{ fontWeight: 600, fontSize: 15 }}>Prompt preview</span>
                    <span style={{ fontSize: 11.5, color: T.ink3 }}>row #1 — what LLM will see</span>
                  </CardHead>
                  <CardBody>
                    {selectedFields.length === 0 ? (
                      <div style={{ padding: "24px 0", textAlign: "center", color: T.ink4, fontSize: 13 }}>
                        Select at least one field to build the prompt.
                      </div>
                    ) : (
                      <pre style={{
                        background: "oklch(0.2 0.02 265)", color: "oklch(0.94 0.008 85)",
                        borderRadius: 12, padding: "12px 14px",
                        fontFamily: "'JetBrains Mono',ui-monospace,monospace",
                        fontSize: 11.5, lineHeight: 1.6, maxHeight: 280, overflow: "auto",
                        margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-word",
                      }}>
                        {previewText}
                      </pre>
                    )}
                  </CardBody>
                </Card>

                <Card>
                  <CardHead><span style={{ fontWeight: 600, fontSize: 15 }}>Record limit</span></CardHead>
                  <CardBody>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 14 }}>
                      {LIMIT_OPTIONS.map((opt) => (
                        <FieldChip
                          key={String(opt.value)}
                          label={opt.label}
                          on={limit === opt.value}
                          onClick={() => setLimit(opt.value)}
                        />
                      ))}
                      <FieldChip label="Custom" on={limit === -1} onClick={() => setLimit(-1)} />
                      {limit === -1 && (
                        <Input size="sm" type="number" min={1} placeholder="e.g. 200" className="w-28"
                          value={customLimit} onValueChange={setCustomLimit} />
                      )}
                    </div>
                    <p style={{ fontSize: 12.5, color: T.ink3, margin: 0 }}>
                      Will process{" "}
                      <strong style={{ color: T.ink }}>
                        {(effectiveLimit == null
                          ? preview.total
                          : Math.min(effectiveLimit, preview.total)
                        ).toLocaleString()}
                      </strong>{" "}
                      records · ~{Math.round(((effectiveLimit ?? preview.total) * 4) / 60)} min estimated
                    </p>
                  </CardBody>
                </Card>

                {error && <p style={{ color: T.danger, fontSize: 13 }}>{error}</p>}

                <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
                  <button type="button" onClick={() => setStep(0)}
                    style={{
                      padding: "9px 14px", borderRadius: 12,
                      border: `1px solid ${T.line}`, background: T.surface2,
                      cursor: "pointer", fontSize: 13, fontWeight: 600, color: T.ink2,
                    }}>
                    Back
                  </button>
                  <Button
                    color="primary"
                    isLoading={starting}
                    isDisabled={selectedFields.length === 0}
                    startContent={!starting && <IconSparkles size={16} />}
                    onPress={handleStart}
                  >
                    {starting ? "Starting…" : `Run · ${(effectiveLimit ?? preview.total).toLocaleString()} rows`}
                  </Button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
