export interface LabelingCV {
  cv_id: number;
  source: string;
  skills: string[];
  seniority: string;
  experience_years: number;
  education: string;
  text_summary: string;
  pdf_path: string;
}

export interface LabelingJob {
  job_id: number;
  title: string;
  skills: string[];
  seniority: string;
  salary_min: number | null;
  salary_max: number | null;
  text_summary: string;
}

export interface PairQueueItem {
  id: number;
  job: LabelingJob;
  skill_overlap_score: number;
  selection_reason: "medium_overlap" | "high_overlap" | "hard_negative" | "random";
  split: "train" | "val" | "test";
  common_skills: string[];
}

export interface LabelingProgress {
  total: number;
  labeled: number;
  skipped: number;
  pending: number;
  current_cv_pending: number;
}

export interface QueueResponse {
  cv: LabelingCV;
  pairs: PairQueueItem[];
  progress: LabelingProgress;
}

export type DimScore = 0 | 1 | 2;
export type OverallScore = 0 | 1 | 2;

export interface DimScores {
  skill_fit: DimScore;
  seniority_fit: DimScore;
  experience_fit: DimScore;
  domain_fit: DimScore;
}

export interface SubmitLabelPayload extends DimScores {
  overall: OverallScore;
  note?: string;
}

export interface LabelingStats {
  total_pairs: number;
  labeled: number;
  skipped: number;
  pending: number;
  label_distribution: Record<string, number>;
  by_reason: Record<string, { labeled: number; total: number }>;
  by_split: Record<string, { labeled: number; total: number }>;
}

export interface ExportItem {
  cv_id: number;
  job_id: number;
  label: 0 | 1;
  split: string;
  skill_fit: DimScore;
  seniority_fit: DimScore;
  experience_fit: DimScore;
  domain_fit: DimScore;
  overall: OverallScore;
}
