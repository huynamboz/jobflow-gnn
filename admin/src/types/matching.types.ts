export interface JobMatchResult {
  job_id: number;
  score: number;
  eligible: boolean;
  matched_skills: string[];
  missing_skills: string[];
  seniority_match: boolean;
  title: string;
  company_name: string;
  location: string;
  job_type: string;
  salary_min: number;
  salary_max: number;
  source_url: string;
}

export interface MatchResponse {
  success: boolean;
  data: JobMatchResult[];
}
