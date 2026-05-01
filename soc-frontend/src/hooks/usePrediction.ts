import { useState, useCallback } from 'react';
import { predict } from '../services/api';
import { PredictResponse, PredictionModel } from '../types/alert';

interface PredictionState {
  loading: boolean;
  result: PredictResponse | null;
  error: string | null;
  features: number[] | null;
}

export function usePrediction() {
  const [state, setState] = useState<PredictionState>({
    loading: false, result: null, error: null, features: null,
  });

  const runPrediction = useCallback(async (features: number[], model: PredictionModel) => {
    setState({ loading: true, result: null, error: null, features });
    try {
      const result = await predict(features, model);
      setState({ loading: false, result, error: null, features });
      return result;
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Unknown error';
      setState({ loading: false, result: null, error: msg, features });
      return null;
    }
  }, []);

  const reset = useCallback(() => {
    setState({ loading: false, result: null, error: null, features: null });
  }, []);

  return { ...state, runPrediction, reset };
}
