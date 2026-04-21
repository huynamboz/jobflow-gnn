import { apiClient } from "@/lib/api-client";
import type { ApiSuccess } from "@/types/api.types";
import type { AdminCVDetail, CVListResponse } from "@/types/cv-admin.types";

export interface CVFilters {
  search?: string;
  source?: string;
  seniority?: string;
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
}

export const cvAdminService = new CVAdminService();
