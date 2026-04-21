export type TrainStatus = "running" | "completed" | "failed";

export interface TrainRun {
  id: number;
  version: string;
  is_active: boolean;
  status: TrainStatus;
  description: string;
  num_jobs: number;
  num_cvs: number;
  num_pairs: number;
  num_skills: number;
  model_type: string;
  hidden_channels: number;
  num_layers: number;
  learning_rate: number;
  auc_roc: number | null;
  recall_at_5: number | null;
  recall_at_10: number | null;
  precision_at_5: number | null;
  precision_at_10: number | null;
  ndcg_at_10: number | null;
  mrr: number | null;
  best_epoch: number | null;
  final_loss: number | null;
  reranker_accuracy: number | null;
  metrics_json: Record<string, number>;
  config_json: Record<string, unknown>;
  checkpoint_path: string;
  training_duration_seconds: number | null;
  started_at: string;
  completed_at: string | null;
}
