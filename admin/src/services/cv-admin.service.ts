import { apiClient } from "@/lib/api-client";
import type { ApiSuccess } from "@/types/api.types";
import type { AdminCVDetail, CVBatch, CVBatchDetail, CVExtractResult, CVListResponse, CVRecordDetail, CVUploadResult } from "@/types/cv-admin.types";

export interface CVFilters {
  search?: string;
  source?: string;
  seniority?: string;
  role_category?: string;
  page?: number;
  page_size?: number;
}

class CVAdminService {
  async listCVs(filters: CVFilters = {}): Promise<CVListResponse> {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([k, v]) => {
      if (v !== undefined && v !== "") params.set(k, String(v));
    });
    const res = await apiClient.get<CVListResponse>(`/admin/cvs/?${params}`);
    return res.data;
  }

  async getCV(id: number): Promise<AdminCVDetail> {
    const res = await apiClient.get<ApiSuccess<AdminCVDetail>>(`/admin/cvs/${id}/`);
    return res.data.data;
  }

  async uploadCV(file: File): Promise<CVUploadResult> {
    const form = new FormData();
    form.append("file", file);
    const res = await apiClient.post<ApiSuccess<CVUploadResult>>("/cvs/upload/", form);
    return res.data.data;
  }

  async extractCV(file: File): Promise<CVExtractResult> {
    const form = new FormData();
    form.append("file", file);
    const res = await apiClient.post<ApiSuccess<CVExtractResult>>("/cvs/extract/", form);
    return res.data.data;
  }

  async saveCV(data: CVExtractResult): Promise<CVUploadResult> {
    const res = await apiClient.post<ApiSuccess<CVUploadResult>>("/cvs/save/", data);
    return res.data.data;
  }

  async listCVBatches(): Promise<CVBatch[]> {
    const res = await apiClient.get<{ success: boolean; data: CVBatch[] }>("/admin/cv/batches/");
    return res.data.data;
  }

  async createCVBatch(payload: { source?: string; source_categories?: string[] }): Promise<CVBatch> {
    const res = await apiClient.post<{ success: boolean; data: CVBatch }>("/admin/cv/batches/", payload);
    return res.data.data;
  }

  async getCVBatch(id: number, page = 1, pageSize = 50, status = ""): Promise<CVBatchDetail> {
    const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
    if (status) params.set("status", status);
    const res = await apiClient.get<CVBatchDetail>(`/admin/cv/batches/${id}/?${params}`);
    return res.data;
  }

  async getCVBatchRecord(batchId: number, recordId: number): Promise<CVRecordDetail> {
    const res = await apiClient.get<{ success: boolean; data: CVRecordDetail }>(`/admin/cv/batches/${batchId}/records/${recordId}/`);
    return res.data.data;
  }

  async cancelCVBatch(id: number): Promise<void> {
    await apiClient.post(`/admin/cv/batches/${id}/cancel/`);
  }

  async exportCVs(filters: { role_category?: string; source?: string } = {}): Promise<void> {
    const params = new URLSearchParams();
    if (filters.role_category) params.set("role_category", filters.role_category);
    if (filters.source) params.set("source", filters.source);
    const res = await apiClient.get(`/admin/cvs/export/?${params}`, { responseType: "blob" });
    _downloadBlob(res.data as Blob, "cvs_extracted.json");
  }
}

function _downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export const cvAdminService = new CVAdminService();
