import { apiClient } from "@/lib/api-client";
import type { ApiSuccess } from "@/types/api.types";
import type { JDBatch, JDBatchDetail, JDExtractResult, JobDetail, JobListResponse } from "@/types/job.types";

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

  async extractJD(rawText: string): Promise<JDExtractResult> {
    const res = await apiClient.post<ApiSuccess<JDExtractResult>>("/admin/jd/extract/", { raw_text: rawText });
    return res.data.data;
  }

  async saveJD(data: JDExtractResult & { raw_text: string }): Promise<{ id: number }> {
    const res = await apiClient.post<ApiSuccess<{ id: number }>>("/admin/jd/save/", data);
    return res.data.data;
  }

  async previewBatch(file: File): Promise<{ total: number; fields: string[]; sample: Record<string, unknown>[]; filename: string }> {
    const form = new FormData();
    form.append("file", file);
    const res = await apiClient.post<ApiSuccess<{ total: number; fields: string[]; sample: Record<string, unknown>[]; filename: string }>>(
      "/admin/jd/batches/preview/", form,
      { headers: { "Content-Type": "multipart/form-data" } },
    );
    return res.data.data;
  }

  async listBatches(): Promise<JDBatch[]> {
    const res = await apiClient.get<ApiSuccess<JDBatch[]>>("/admin/jd/batches/");
    return res.data.data;
  }

  async createBatch(file: File, fieldsConfig: string[], limit: number | null): Promise<JDBatch> {
    const form = new FormData();
    form.append("file", file);
    form.append("fields_config", JSON.stringify(fieldsConfig));
    if (limit != null) form.append("limit", String(limit));
    const res = await apiClient.post<ApiSuccess<JDBatch>>("/admin/jd/batches/", form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data.data;
  }

  async getBatch(id: number, page = 1, pageSize = 50, status = ""): Promise<JDBatchDetail> {
    const p = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
    if (status) p.set("status", status);
    const res = await apiClient.get<ApiSuccess<JDBatchDetail>>(`/admin/jd/batches/${id}/?${p}`);
    return res.data.data;
  }

  async cancelBatch(id: number): Promise<void> {
    await apiClient.post(`/admin/jd/batches/${id}/cancel/`);
  }

  async getBatchRecord(batchId: number, recordId: number): Promise<import("@/types/job.types").JDBatchRecord> {
    const res = await apiClient.get<ApiSuccess<import("@/types/job.types").JDBatchRecord>>(`/admin/jd/batches/${batchId}/records/${recordId}/`);
    return res.data.data;
  }
}

export const jobService = new JobService();
