export type PredictionLabel = 0 | 1 | 2;
export type AlertStatus = 'open' | 'investigating' | 'resolved' | 'escalated' | 'remediation';
export type SeverityLevel = 'critical' | 'high' | 'medium' | 'low';
export type PredictionModel = 'xgboost' | 'lightgbm' | 'tabnet';

export interface PredictResponse {
  prediction: PredictionLabel;
  probabilities: number[];
  model: PredictionModel;
}

export interface RemediationRecommendation {
  prediction: number;
  probability: number;
  threshold: number;
}

export interface RemediationResponse {
  account_response: RemediationRecommendation;
  endpoint_response: RemediationRecommendation;
}

export interface Alert {
  id: string;
  timestamp: Date;
  sourceIp: string;
  destIp: string;
  port: number;
  protocol: string;
  severity: SeverityLevel;
  prediction: PredictionLabel;
  probabilities: number[];
  confidence: number;
  status: AlertStatus;
  features: number[];
  incidentFeatures?: number[];
  remediationPrediction?: RemediationResponse;
  analyst?: string;
  notes?: string;
  remediationActions?: RemediationAction[];
  detectedAt: Date;
  resolvedAt?: Date;
}

export interface RemediationAction {
  id: string;
  type: 'block_ip' | 'disable_account' | 'isolate_endpoint' | 'custom';
  label: string;
  status: 'pending' | 'approved' | 'rejected';
  notes?: string;
  timestamp: Date;
}

export interface DashboardStats {
  total: number;
  truePositives: number;
  falsePositives: number;
  benignPositives: number;
  open: number;
  investigating: number;
  resolved: number;
  avgConfidence: number;
}

export const PREDICTION_LABELS: Record<PredictionLabel, string> = {
  0: 'False Positive',
  1: 'Benign Positive',
  2: 'True Positive',
};

export const PREDICTION_SHORT: Record<PredictionLabel, string> = {
  0: 'FP',
  1: 'BP',
  2: 'TP',
};

export const PREDICTION_COLORS: Record<PredictionLabel, string> = {
  0: '#22c55e',
  1: '#eab308',
  2: '#ef4444',
};

export const SEVERITY_ORDER: Record<SeverityLevel, number> = {
  critical: 4, high: 3, medium: 2, low: 1
};
