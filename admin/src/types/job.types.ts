export interface JobCompany {
  id: number;
  name: string;
  industry: string;
  size: string;
  location: string;
}

export interface JobPlatform {
  id: number;
  name: string;
  slug: string;
}

export interface JobSkill {
  skill_name: string;
  category: number;
  importance: number;
}

export interface JobListItem {
  id: number;
  title: string;
  company_name: string;
  platform_name: string;
  location: string;
  seniority: number;
  job_type: string;
  salary_min: number;
  salary_max: number;
  is_active: boolean;
  date_posted: string | null;
  created_at: string;
}

export interface JobDetail {
  id: number;
  title: string;
  company: JobCompany | null;
  platform: JobPlatform | null;
  location: string;
  seniority: number;
  job_type: string;
  salary_min: number;
  salary_max: number;
  is_active: boolean;
  date_posted: string | null;
  created_at: string;
  description: string;
  responsibilities: string;
  requirements: string;
  source_url: string;
  applicant_count: number | null;
  skills: JobSkill[];
}

export interface JobListResponse {
  success: boolean;
  data: JobListItem[];
  total: number;
  page: number;
  page_size: number;
}
