import { apiClient } from "@/lib/api-client";
import type { JobMatchResult, MatchResponse } from "@/types/matching.types";

class MatchingService {
  async matchText(text: string, topK = 10): Promise<JobMatchResult[]> {
    const res = await apiClient.post<MatchResponse>("/matching/cv/", { text, top_k: topK });
    return res.data.data;
  }

  async matchFile(file: File, topK = 10): Promise<JobMatchResult[]> {
    const form = new FormData();
    form.append("file", file);
    form.append("top_k", String(topK));
    const res = await apiClient.post<MatchResponse>("/matching/cv/upload/", form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data.data;
  }
}

export const matchingService = new MatchingService();
