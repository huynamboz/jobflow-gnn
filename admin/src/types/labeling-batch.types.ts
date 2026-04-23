export interface LabelingBatch {
  id: number;
  status: "running" | "done" | "error" | "cancelled";
  total: number;
  done_count: number;
  error_count: number;
  workers: number;
  pct: number;
  created_at: string;
}

export interface LabelingQueueStats {
  total: number;
  pending: number;
  labeled: number;
  skipped: number;
}

export interface LabelingLabelStats {
  total: number;
  overall_0: number;
  overall_1: number;
  overall_2: number;
}

export interface LabelingBatchListResponse {
  batches: LabelingBatch[];
  queue: LabelingQueueStats;
  labels: LabelingLabelStats;
}

export interface RecentLabel {
  pair_id: number;
  cv_id: number;
  job_id: number;
  // CV
  cv_role: string;
  cv_seniority: string;
  cv_experience: number;
  cv_education: string;
  cv_skills: string[];
  cv_text: string;
  // Job
  job_title: string;
  job_role: string;
  job_seniority: string;
  job_experience: string;
  job_skills: string[];
  job_text: string;
  // Scores
  skill_fit: number;
  seniority_fit: number;
  experience_fit: number;
  domain_fit: number;
  overall: number;
  selection: string;
  created_at: string;
}

export interface LabelDist {
  overall_0: number;
  overall_1: number;
  overall_2: number;
}

export interface LabelingBatchDetail extends LabelingBatch {
  recent: RecentLabel[];
  dist: LabelDist;
}
