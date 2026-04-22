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

export interface CVUploadResult {
  id: number;
  file_name: string;
  candidate_name: string;
  seniority: number;
  experience_years: number;
  education: number;
  parsed_text: string;
  source: string;
  source_category: string;
  skills: CVSkillItem[];
  work_experience: WorkExperienceItem[];
  is_active: boolean;
  created_at: string;
}
