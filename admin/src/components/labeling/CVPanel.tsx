import type { LabelingCV } from "@/types/labeling.types";

interface CVPanelProps {
  cv: LabelingCV;
}

const SENIORITY_COLOR: Record<string, string> = {
  INTERN:  "bg-gray-100 text-gray-600",
  JUNIOR:  "bg-blue-100 text-blue-700",
  MID:     "bg-indigo-100 text-indigo-700",
  SENIOR:  "bg-purple-100 text-purple-700",
  LEAD:    "bg-pink-100 text-pink-700",
  MANAGER: "bg-rose-100 text-rose-700",
};

export function CVPanel({ cv }: CVPanelProps) {
  const seniorityClass = SENIORITY_COLOR[cv.seniority] ?? "bg-default-100 text-default-600";

  return (
    <div className="rounded-2xl border border-default-200 bg-white p-5 space-y-4 sticky top-4">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className={`rounded-lg px-2.5 py-1 text-xs font-semibold ${seniorityClass}`}>
            {cv.seniority}
          </span>
          <span className="text-xs text-default-500">CV #{cv.cv_id}</span>
        </div>
        <div className="flex gap-4 text-sm text-default-600 mt-2">
          <span>{cv.experience_years}y exp</span>
          <span>·</span>
          <span>{cv.education}</span>
          <span>·</span>
          <span className="capitalize">{cv.source}</span>
        </div>
      </div>

      {/* Skills */}
      <div>
        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-default-400">Skills</p>
        <div className="flex flex-wrap gap-1.5">
          {cv.skills.slice(0, 20).map((s) => (
            <span key={s} className="rounded-lg border border-default-200 bg-default-50 px-2 py-0.5 text-xs text-default-600">
              {s}
            </span>
          ))}
          {cv.skills.length > 20 && (
            <span className="text-xs text-default-400">+{cv.skills.length - 20}</span>
          )}
        </div>
      </div>

      {/* Summary */}
      {cv.text_summary && (
        <div>
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-default-400">Summary</p>
          <p className="text-xs leading-relaxed text-default-500 line-clamp-6">{cv.text_summary}</p>
        </div>
      )}
    </div>
  );
}
