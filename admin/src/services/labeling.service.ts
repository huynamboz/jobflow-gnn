import type { DimScores, OverallScore } from "@/types/labeling.types";

import { apiClient } from "@/lib/api-client";
import type { ApiSuccess } from "@/types/api.types";
import type { QueueResponse, LabelingStats, SubmitLabelPayload, ExportItem } from "@/types/labeling.types";

export function computeSuggestedOverall(dims: DimScores): OverallScore {
  const avg = (dims.skill_fit + dims.seniority_fit + dims.experience_fit + dims.domain_fit) / 4;
  if (avg >= 1.5) return 2;
  if (avg >= 0.75) return 1;
  return 0;
}

class LabelingService {
  async getQueue(): Promise<QueueResponse | null> {
    const res = await apiClient.get<ApiSuccess<QueueResponse | null>>("/labeling/queue/");
    return res.data.data;
  }

  async submitLabel(pairId: number, payload: SubmitLabelPayload): Promise<void> {
    await apiClient.post(`/labeling/${pairId}/submit/`, payload);
  }

  async skipPair(pairId: number): Promise<void> {
    await apiClient.post(`/labeling/${pairId}/skip/`);
  }

  async getStats(): Promise<LabelingStats> {
    const res = await apiClient.get<ApiSuccess<LabelingStats>>("/labeling/stats/");
    return res.data.data;
  }

  async exportLabels(): Promise<ExportItem[]> {
    const res = await apiClient.get<ApiSuccess<ExportItem[]>>("/labeling/export/");
    return res.data.data;
  }
}

export const labelingService = new LabelingService();
