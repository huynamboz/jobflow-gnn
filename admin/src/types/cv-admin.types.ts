export interface CVSkillItem {
  skill_name: string;
  category: number;
  proficiency: number;
}

export interface AdminCVItem {
  id: number;
  file_name: string;
  seniority: number;
  experience_years: number;
  education: number;
  role_category: string;
  source: string;
  skill_count: number;
  is_active: boolean;
  created_at: string;
}

export interface WorkExperienceItem {
  title: string;
  company: string;
  duration: string;
  description: string;
}

export interface AdminCVDetail extends AdminCVItem {
  candidate_name: string;
  source_category: string;
  skills: CVSkillItem[];
  parsed_text: string;
  work_experience: WorkExperienceItem[];
}

export interface CVSkillEdit {
  name: string;
  proficiency: number;
}

export interface CVExtractResult {
  file_name: string;
  raw_text: string;
  candidate_name: string;
  experience_years: number;
  education: number;
  seniority: number;
  role_category: string;
  skills: CVSkillEdit[];
  work_experience: WorkExperienceItem[];
  llm_used: boolean;
}

export interface CVListResponse {
  success: boolean;
  data: AdminCVItem[];
  total: number;
  page: number;
  page_size: number;
}

export type CVBatchStatus = "pending" | "running" | "done" | "error" | "cancelled";
export type CVRecordStatus = "pending" | "processing" | "done" | "error";

export interface CVBatch {
  id: number;
  filter_source: string;
  filter_source_categories: string[];
  status: CVBatchStatus;
  total: number;
  done_count: number;
  error_count: number;
  created_at: string;
}

export interface CVBatchRecord {
  id: number;
  cv_id: number;
  file_name: string;
  source_category: string;
  status: CVRecordStatus;
  error_msg: string;
  role_category: string | null;
  seniority: number | null;
  experience_years: number | null;
  skill_count: number;
}

export interface CVBatchDetail {
  batch: CVBatch;
  records: CVBatchRecord[];
  total_records: number;
  page: number;
  page_size: number;
}

export interface CVRecordSkill {
  name: string;
  proficiency: number;
  importance: number;
}

export interface CVRecordDetail {
  id: number;
  cv_id: number;
  file_name: string;
  source_category: string;
  status: CVRecordStatus;
  error_msg: string | null;
  raw_text: string;
  result: {
    role_category: string | null;
    seniority: number | null;
    experience_years: number | null;
    education: number | null;
    name: string | null;
    skills: CVRecordSkill[];
    work_experience: WorkExperienceItem[];
  } | null;
}

export interface CVUploadResult {
  id: number;
  file_name: string;
  candidate_name: string;
  seniority: number;
  experience_years: number;
  education: number;
  role_category: string;
  parsed_text: string;
  source: string;
  source_category: string;
  skills: CVSkillItem[];
  work_experience: WorkExperienceItem[];
  is_active: boolean;
  created_at: string;
}
