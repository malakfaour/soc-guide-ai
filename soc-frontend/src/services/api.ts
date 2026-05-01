import { PredictResponse, PredictionModel, RemediationResponse } from '../types/alert';

export const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface BackendHealthEntry {
  loaded: boolean;
  error?: string | null;
}

export interface BackendHealthResponse {
  status: string;
  models: Record<string, BackendHealthEntry>;
}

export interface MetricsClassStats {
  precision?: number;
  recall?: number;
  f1?: number;
  support?: number;
}

export interface MetricsResponse {
  confusion_matrix: number[][];
  accuracy: number;
  macro_f1: number;
  per_class: Record<string, MetricsClassStats>;
}

export interface FeatureSampleResponse {
  features: number[];
  dataset: string;
  split: 'train' | 'val' | 'test';
  row: number;
  feature_count: number;
  target?: number | null;
  source: string;
}

export async function predict(features: number[], model: PredictionModel): Promise<PredictResponse> {
  if (!Array.isArray(features) || features.length === 0 || features.some(feature => !Number.isFinite(feature))) {
    throw new Error('Prediction requires a non-empty numeric feature array.');
  }

  let res: Response;
  try {
    res = await fetch(`${BASE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features, model }),
    });
  } catch {
    throw new Error('Unable to reach the FastAPI backend at http://localhost:8000.');
  }

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Prediction failed (${res.status}): ${err || 'Unknown backend error'}`);
  }

  const data = await res.json() as PredictResponse;
  if (!Array.isArray(data.probabilities) || data.probabilities.length !== 3) {
    throw new Error('Backend returned an invalid probability payload.');
  }

  return data;
}

export async function healthStatus(): Promise<BackendHealthResponse> {
  let res: Response;
  try {
    res = await fetch(`${BASE_URL}/health`, { signal: AbortSignal.timeout(3000) });
  } catch {
    throw new Error('Unable to reach the FastAPI backend at http://localhost:8000.');
  }

  if (!res.ok) {
    throw new Error(`Health check failed (${res.status}).`);
  }

  return await res.json() as BackendHealthResponse;
}

export async function health(): Promise<boolean> {
  try {
    const response = await healthStatus();
    return response.status === 'healthy';
  } catch {
    return false;
  }
}

export async function remediationPredict(incidentFeatures: number[]): Promise<RemediationResponse> {
  if (!Array.isArray(incidentFeatures) || incidentFeatures.length === 0 || incidentFeatures.some(value => !Number.isFinite(value))) {
    throw new Error('Remediation requires a non-empty numeric incident feature array.');
  }

  let res: Response;
  try {
    res = await fetch(`${BASE_URL}/remediation-predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ incident_features: incidentFeatures }),
    });
  } catch {
    throw new Error('Unable to reach the FastAPI remediation endpoint at http://localhost:8000.');
  }

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Remediation failed (${res.status}): ${err || 'Unknown backend error'}`);
  }

  return await res.json() as RemediationResponse;
}

export async function getMetrics(): Promise<MetricsResponse> {
  const res = await fetch(`${BASE_URL}/metrics`);
  if (!res.ok) {
    throw new Error('Failed to fetch metrics');
  }
  return await res.json() as MetricsResponse;
}

export async function getSampleFeatures(split: 'train' | 'val' | 'test' = 'test', row = 0): Promise<FeatureSampleResponse> {
  const res = await fetch(`${BASE_URL}/sample-features?split=${split}&row=${row}`);
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Failed to fetch processed sample (${res.status}): ${err || 'Unknown backend error'}`);
  }
  return await res.json() as FeatureSampleResponse;
}
