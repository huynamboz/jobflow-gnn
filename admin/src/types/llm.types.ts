export type LLMClientType = "openai" | "messages";

export interface LLMProvider {
  id: number;
  name: string;
  api_key: string;
  model: string;
  base_url: string;
  client_type: LLMClientType;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface LLMProviderWrite {
  name: string;
  api_key: string;
  model: string;
  base_url: string;
  client_type: LLMClientType;
}

export interface LLMTestResult {
  ok: boolean;
  message: string;
}

export interface LLMCallLog {
  id: number;
  provider_name: string | null;
  feature: string;
  status: "success" | "error";
  input_preview: string;
  output: string;
  error_message: string;
  duration_ms: number | null;
  created_at: string;
}

export interface LLMCallLogListResponse {
  success: boolean;
  data: LLMCallLog[];
  total: number;
  page: number;
  page_size: number;
}
