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

export interface AdminCVDetail extends AdminCVItem {
  source_category: string;
  skills: CVSkillItem[];
  parsed_text: string;
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
  seniority: number;
  experience_years: number;
  education: number;
  parsed_text: string;
  source: string;
  source_category: string;
  skills: CVSkillItem[];
  is_active: boolean;
  created_at: string;
}
