import { apiClient } from "@/lib/api-client";
import type { ApiSuccess } from "@/types/api.types";
import type { TrainRun } from "@/types/model.types";

class ModelService {
  async listRuns(): Promise<TrainRun[]> {
    const res = await apiClient.get<ApiSuccess<TrainRun[]>>("/admin/training/");
    return res.data.data;
  }

  async getRun(id: number): Promise<TrainRun> {
    const res = await apiClient.get<ApiSuccess<TrainRun>>(`/admin/training/${id}/`);
    return res.data.data;
  }

  async activateRun(id: number): Promise<TrainRun> {
    const res = await apiClient.post<ApiSuccess<TrainRun>>(`/admin/training/${id}/activate/`);
    return res.data.data;
  }

  async triggerTraining(): Promise<TrainRun> {
    const res = await apiClient.post<ApiSuccess<TrainRun>>("/admin/training/trigger/");
    return res.data.data;
  }

  async updateDescription(id: number, description: string): Promise<TrainRun> {
    const res = await apiClient.patch<ApiSuccess<TrainRun>>(`/admin/training/${id}/`, { description });
    return res.data.data;
  }
}

export const modelService = new ModelService();
