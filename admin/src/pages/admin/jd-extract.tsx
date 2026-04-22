import { useState } from "react";
import { Card, CardBody } from "@heroui/card";
import { Button } from "@heroui/button";
import { Input } from "@heroui/input";
import { Select, SelectItem } from "@heroui/select";
import { Chip } from "@heroui/chip";
import {
  IconCircleCheck,
  IconDeviceFloppy,
  IconLoader2,
  IconSparkles,
  IconX,
} from "@tabler/icons-react";

import { jobService } from "@/services/job.service";
import type { JDExtractResult, JDExtractSkill } from "@/types/job.types";

const SENIORITY_OPTIONS = [
  { key: "0", label: "Intern / Fresher" },
  { key: "1", label: "Junior (< 2 yrs)" },
  { key: "2", label: "Mid-level (2–4 yrs)" },
  { key: "3", label: "Senior (5+ yrs)" },
  { key: "4", label: "Lead / Principal" },
  { key: "5", label: "Manager / Director" },
];

const DEGREE_OPTIONS = [
  { key: "0", label: "Any / Not required" },
  { key: "1", label: "High School" },
  { key: "2", label: "College / Diploma" },
  { key: "3", label: "Bachelor's" },
  { key: "4", label: "Master's / MBA" },
  { key: "5", label: "PhD / Doctorate" },
];

const JOB_TYPE_OPTIONS = [
  "full-time", "part-time", "contract", "remote", "hybrid", "on-site",
].map((v) => ({ key: v, label: v }));

const SALARY_TYPE_OPTIONS = [
  { key: "hourly", label: "per hour" },
  { key: "monthly", label: "per month" },
  { key: "annual", label: "per year" },
  { key: "unknown", label: "—" },
];

const SENIORITY_LABELS: Record<string, string> = Object.fromEntries(SENIORITY_OPTIONS.map((o) => [o.key, o.label]));
const DEGREE_LABELS: Record<string, string> = Object.fromEntries(DEGREE_OPTIONS.map((o) => [o.key, o.label]));
const SALARY_PERIOD: Record<string, string> = { hourly: "/hr", monthly: "/mo", annual: "/yr", unknown: "" };

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[11px] font-semibold uppercase tracking-wider text-default-400">{children}</p>
  );
}

function ImportanceDots({ value }: { value: number }) {
  return (
    <span className="flex gap-[3px]">
      {[1, 2, 3, 4, 5].map((i) => (
        <span key={i} className={`inline-block h-1.5 w-1.5 rounded-full ${i <= value ? "bg-primary" : "bg-default-200"}`} />
      ))}
    </span>
  );
}

function SkillsEditor({ skills, onChange }: { skills: JDExtractSkill[]; onChange: (s: JDExtractSkill[]) => void }) {
  const [name, setName] = useState("");
  const [imp, setImp] = useState("3");

  const add = () => {
    const n = name.trim();
    if (!n || skills.some((s) => s.name.toLowerCase() === n.toLowerCase())) return;
    onChange([...skills, { name: n, importance: Number(imp) }]);
    setName("");
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-1.5">
        {skills.map((s, i) => (
          <Chip
            key={i}
            variant="flat"
            size="sm"
            endContent={
              <button
                type="button"
                className="ml-0.5 text-default-400 hover:text-danger transition-colors"
                onClick={() => onChange(skills.filter((_, j) => j !== i))}
              >
                <IconX size={12} />
              </button>
            }
          >
            <span className="flex items-center gap-1.5">
              {s.name}
              <select
                className="border-0 bg-transparent p-0 text-[11px] text-default-400 outline-none"
                value={s.importance}
                onChange={(e) => {
                  const next = [...skills];
                  next[i] = { ...next[i], importance: Number(e.target.value) };
                  onChange(next);
                }}
              >
                {[1, 2, 3, 4, 5].map((v) => <option key={v} value={v}>{v}★</option>)}
              </select>
            </span>
          </Chip>
        ))}
      </div>
      <div className="flex gap-2">
        <Input
          size="sm"
          placeholder="Add skill…"
          value={name}
          onValueChange={setName}
          onKeyDown={(e) => e.key === "Enter" && add()}
          classNames={{ inputWrapper: "h-8" }}
        />
        <select
          className="h-8 rounded-lg border border-default-200 bg-white px-2 text-sm outline-none"
          value={imp}
          onChange={(e) => setImp(e.target.value)}
        >
          {[1, 2, 3, 4, 5].map((v) => <option key={v} value={v}>{v}★</option>)}
        </select>
        <Button size="sm" color="primary" isDisabled={!name.trim()} onPress={add}>Add</Button>
      </div>
    </div>
  );
}

export default function JDExtractPage() {
  const [rawText, setRawText] = useState("");
  const [extracting, setExtracting] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [saveError, setSaveError] = useState("");
  const [savedId, setSavedId] = useState<number | null>(null);
  const [form, setForm] = useState<JDExtractResult | null>(null);

  const patch = (key: keyof JDExtractResult, value: unknown) =>
    setForm((f) => f ? { ...f, [key]: value } : f);

  const handleExtract = async () => {
    if (!rawText.trim()) return;
    setExtracting(true);
    setError(""); setSaveError(""); setSavedId(null); setForm(null);
    try {
      setForm(await jobService.extractJD(rawText));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Extraction failed");
    } finally {
      setExtracting(false);
    }
  };

  const handleSave = async () => {
    if (!form) return;
    setSaving(true); setSaveError(""); setSavedId(null);
    try {
      const r = await jobService.saveJD({ ...form, raw_text: rawText });
      setSavedId(r.id);
    } catch (e: unknown) {
      setSaveError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold text-default-900">JD Extraction</h1>
        <p className="mt-0.5 text-sm text-default-400">Paste a job description — LLM extracts structured fields.</p>
      </div>

      {/* Input */}
      <Card shadow="none" className="border border-default-100">
        <CardBody className="space-y-3 p-5">
          <textarea
            className="h-52 w-full resize-none rounded-xl border border-default-200 bg-default-50 p-4 text-sm leading-relaxed text-default-800 outline-none transition-colors focus:border-primary focus:bg-white"
            placeholder="Paste raw job description here…"
            value={rawText}
            onChange={(e) => setRawText(e.target.value)}
          />
          <div className="flex items-center gap-3">
            <Button
              color="primary"
              isLoading={extracting}
              isDisabled={!rawText.trim()}
              startContent={!extracting && <IconSparkles size={16} />}
              onPress={handleExtract}
            >
              {extracting ? "Extracting…" : "Extract"}
            </Button>
            {error && <p className="text-sm text-danger">{error}</p>}
          </div>
        </CardBody>
      </Card>

      {/* Result */}
      {form && (
        <Card shadow="none" className="border border-default-100">
          <CardBody className="space-y-6 p-5">
            {/* Header */}
            <div className="flex items-center justify-between">
              <p className="font-semibold text-default-800">Extracted Information</p>
              <div className="flex items-center gap-3">
                {savedId && (
                  <span className="flex items-center gap-1.5 text-sm text-success-600">
                    <IconCircleCheck size={16} /> Saved as Job #{savedId}
                  </span>
                )}
                {saveError && <p className="text-sm text-danger">{saveError}</p>}
                <Button
                  color="success"
                  variant="flat"
                  isLoading={saving}
                  isDisabled={!form.title}
                  startContent={!saving && <IconDeviceFloppy size={16} />}
                  onPress={handleSave}
                >
                  {saving ? "Saving…" : "Save Job"}
                </Button>
              </div>
            </div>

            {/* Basic */}
            <div className="space-y-3">
              <SectionTitle>Basic Info</SectionTitle>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <Input label="Job Title" size="sm" value={form.title} onValueChange={(v) => patch("title", v)} />
                <Input label="Company" size="sm" value={form.company} onValueChange={(v) => patch("company", v)} />
                <Input label="Location" size="sm" value={form.location} onValueChange={(v) => patch("location", v)} />
                <Select
                  label="Job Type" size="sm"
                  selectedKeys={new Set([form.job_type])}
                  onSelectionChange={(k) => patch("job_type", Array.from(k)[0])}
                >
                  {JOB_TYPE_OPTIONS.map((o) => <SelectItem key={o.key}>{o.label}</SelectItem>)}
                </Select>
              </div>
            </div>

            <div className="border-t border-default-100" />

            {/* Requirements */}
            <div className="space-y-3">
              <SectionTitle>Requirements</SectionTitle>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <Select
                  label="Seniority" size="sm"
                  selectedKeys={new Set([String(form.seniority)])}
                  onSelectionChange={(k) => patch("seniority", Number(Array.from(k)[0]))}
                >
                  {SENIORITY_OPTIONS.map((o) => <SelectItem key={o.key}>{o.label}</SelectItem>)}
                </Select>
                <Select
                  label="Degree" size="sm"
                  selectedKeys={new Set([String(form.degree_requirement)])}
                  onSelectionChange={(k) => patch("degree_requirement", Number(Array.from(k)[0]))}
                >
                  {DEGREE_OPTIONS.map((o) => <SelectItem key={o.key}>{o.label}</SelectItem>)}
                </Select>
                <Input
                  label="Experience Min (yrs)" size="sm" type="number" min={0} step={0.5}
                  value={String(form.experience_min)}
                  onValueChange={(v) => patch("experience_min", parseFloat(v) || 0)}
                />
                <Input
                  label="Experience Max (yrs)" size="sm" type="number" min={0} step={0.5}
                  value={form.experience_max != null ? String(form.experience_max) : ""}
                  placeholder="No limit"
                  onValueChange={(v) => patch("experience_max", v === "" ? null : parseFloat(v))}
                />
              </div>
            </div>

            <div className="border-t border-default-100" />

            {/* Salary */}
            <div className="space-y-3">
              <SectionTitle>Salary</SectionTitle>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Input
                  label="Min" size="sm" type="number" min={0}
                  value={String(form.salary_min)}
                  onValueChange={(v) => patch("salary_min", Number(v))}
                />
                <Input
                  label="Max" size="sm" type="number" min={0}
                  value={String(form.salary_max)}
                  onValueChange={(v) => patch("salary_max", Number(v))}
                />
                <Input
                  label="Currency" size="sm"
                  value={form.salary_currency}
                  onValueChange={(v) => patch("salary_currency", v.toUpperCase())}
                />
                <Select
                  label="Period" size="sm"
                  selectedKeys={new Set([form.salary_type])}
                  onSelectionChange={(k) => patch("salary_type", Array.from(k)[0])}
                >
                  {SALARY_TYPE_OPTIONS.map((o) => <SelectItem key={o.key}>{o.label}</SelectItem>)}
                </Select>
              </div>
            </div>

            <div className="border-t border-default-100" />

            {/* Skills */}
            <div className="space-y-3">
              <SectionTitle>
                Skills{" "}
                <span className="ml-1 rounded-full bg-default-100 px-1.5 py-0.5 text-[10px] font-semibold text-default-500">
                  {form.skills.length}
                </span>
              </SectionTitle>
              <SkillsEditor skills={form.skills} onChange={(s) => patch("skills", s)} />
            </div>

            {/* Summary strip */}
            <div className="rounded-xl bg-default-50 px-4 py-3">
              <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 text-sm text-default-600">
                <span className="font-semibold text-default-800">{form.title || "—"}</span>
                {form.company && <span className="text-default-400">@ {form.company}</span>}
                {form.location && <><span className="text-default-200">·</span><span>{form.location}</span></>}
                <span className="text-default-200">·</span>
                <span>{SENIORITY_LABELS[form.seniority]}</span>
                <span className="text-default-200">·</span>
                <span>{form.job_type}</span>
                {form.salary_min > 0 && (
                  <><span className="text-default-200">·</span>
                  <span>{form.salary_currency} {form.salary_min.toLocaleString()}–{form.salary_max.toLocaleString()} {SALARY_PERIOD[form.salary_type]}</span></>
                )}
                {form.experience_min > 0 && (
                  <><span className="text-default-200">·</span>
                  <span>{form.experience_min}{form.experience_max != null ? `–${form.experience_max}` : "+"} yrs</span></>
                )}
                {form.degree_requirement > 0 && (
                  <><span className="text-default-200">·</span>
                  <span>{DEGREE_LABELS[form.degree_requirement]}</span></>
                )}
              </div>
              {form.skills.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {form.skills.map((s, i) => (
                    <Chip key={i} size="sm" variant="bordered">
                      <span className="flex items-center gap-1.5">
                        {s.name}
                        <ImportanceDots value={s.importance} />
                      </span>
                    </Chip>
                  ))}
                </div>
              )}
            </div>
          </CardBody>
        </Card>
      )}
    </div>
  );
}
