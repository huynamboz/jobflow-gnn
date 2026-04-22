import { apiClient } from "@/lib/api-client";
import type { ApiSuccess } from "@/types/api.types";
import type { LLMCallLog, LLMCallLogListResponse, LLMProvider, LLMProviderWrite, LLMTestResult } from "@/types/llm.types";

class LLMService {
  async list(): Promise<LLMProvider[]> {
    const res = await apiClient.get<ApiSuccess<LLMProvider[]>>("/admin/llm/providers/");
    return res.data.data;
  }

  async create(data: LLMProviderWrite): Promise<LLMProvider> {
    const res = await apiClient.post<ApiSuccess<LLMProvider>>("/admin/llm/providers/", data);
    return res.data.data;
  }

  async update(id: number, data: Partial<LLMProviderWrite>): Promise<LLMProvider> {
    const res = await apiClient.put<ApiSuccess<LLMProvider>>(`/admin/llm/providers/${id}/`, data);
    return res.data.data;
  }

  async delete(id: number): Promise<void> {
    await apiClient.delete(`/admin/llm/providers/${id}/`);
  }

  async activate(id: number): Promise<LLMProvider> {
    const res = await apiClient.post<ApiSuccess<LLMProvider>>(`/admin/llm/providers/${id}/activate/`);
    return res.data.data;
  }

  async test(id: number): Promise<LLMTestResult> {
    const res = await apiClient.post<ApiSuccess<LLMTestResult>>(`/admin/llm/providers/${id}/test/`);
    return res.data.data;
  }

  async listLogs(params: { status?: string; feature?: string; page?: number; page_size?: number } = {}): Promise<LLMCallLogListResponse> {
    const p = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => { if (v !== undefined && v !== "") p.set(k, String(v)); });
    const res = await apiClient.get<LLMCallLogListResponse>(`/admin/llm/logs/?${p}`);
    return res.data;
  }

  async getLog(id: number): Promise<LLMCallLog> {
    const res = await apiClient.get<ApiSuccess<LLMCallLog>>(`/admin/llm/logs/${id}/`);
    return res.data.data;
  }
}

export const llmService = new LLMService();
