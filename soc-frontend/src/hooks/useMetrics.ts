import { useEffect, useState } from 'react';
import { getMetrics, MetricsResponse } from '../services/api';
import type { PredictionModel } from '../types/alert';

interface MetricsState {
  loading: boolean;
  error: string | null;
  data: MetricsResponse | null;
}

export function useMetrics(model: PredictionModel) {
  const [state, setState] = useState<MetricsState>({
    loading: true,
    error: null,
    data: null,
  });

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      setState({ loading: true, error: null, data: null });
      try {
        const data = await getMetrics(model);
        if (!cancelled) {
          setState({ loading: false, error: null, data });
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to fetch metrics';
        if (!cancelled) {
          setState({ loading: false, error: message, data: null });
        }
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, [model]);

  return state;
}
