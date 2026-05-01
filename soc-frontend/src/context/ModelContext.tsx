import React, { createContext, useContext, useState } from 'react';
import type { PredictionModel } from '../types/alert';

interface ModelContextValue {
  model: PredictionModel;
  setModel: React.Dispatch<React.SetStateAction<PredictionModel>>;
}

const ModelContext = createContext<ModelContextValue | null>(null);

export function ModelProvider({ children }: { children: React.ReactNode }) {
  const [model, setModel] = useState<PredictionModel>('xgboost');

  return (
    <ModelContext.Provider value={{ model, setModel }}>
      {children}
    </ModelContext.Provider>
  );
}

export function useModel() {
  const context = useContext(ModelContext);
  if (!context) {
    throw new Error('useModel must be used within a ModelProvider.');
  }
  return context;
}
