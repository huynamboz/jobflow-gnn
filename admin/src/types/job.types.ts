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
  name: string;
  category: string;
  importance: number;
}

export interface JobListItem {
  id: number;
  title: string;
  company: JobCompany | null;
  platform: JobPlatform | null;
  location: string;
  seniority: number;
  job_type: string;
  salary_min: number | null;
  salary_max: number | null;
  is_active: boolean;
  posted_at: string | null;
  created_at: string;
}

export interface JobDetail extends JobListItem {
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
