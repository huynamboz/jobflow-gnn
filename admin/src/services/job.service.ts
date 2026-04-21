import { apiClient } from "@/lib/api-client";
import type { ApiSuccess } from "@/types/api.types";
import type { JobDetail, JobListResponse } from "@/types/job.types";

export interface JobFilters {
  search?: string;
  platform?: string;
  seniority?: string;
  job_type?: string;
  page?: number;
  page_size?: number;
}

class JobService {
  async listJobs(filters: JobFilters = {}): Promise<JobListResponse> {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([k, v]) => {
      if (v !== undefined && v !== "") params.set(k, String(v));
    });
    const res = await apiClient.get<JobListResponse>(`/admin/jobs/?${params}`);
    return res.data;
  }

  async getJob(id: number): Promise<JobDetail> {
    const res = await apiClient.get<ApiSuccess<JobDetail>>(`/admin/jobs/${id}/`);
    return res.data.data;
  }
}

export const jobService = new JobService();
