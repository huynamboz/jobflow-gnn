export interface CVSkillItem {
  name: string;
  category: string;
  proficiency: number;
}

export interface AdminCVItem {
  id: number;
  file_name: string;
  seniority: number;
  experience_years: number;
  education: number;
  source: string;
  source_category: string;
  skill_count: number;
  is_active: boolean;
  created_at: string;
}

export interface AdminCVDetail extends AdminCVItem {
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
