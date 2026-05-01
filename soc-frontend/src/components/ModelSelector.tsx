import { BrainCircuit } from 'lucide-react';
import { useModel } from '../context/ModelContext';
import type { PredictionModel } from '../types/alert';

const MODEL_OPTIONS: { label: string; value: PredictionModel }[] = [
  { label: 'XGBoost', value: 'xgboost' },
  { label: 'LightGBM', value: 'lightgbm' },
  { label: 'TabNet', value: 'tabnet' },
];

export function ModelSelector() {
  const { model, setModel } = useModel();

  return (
    <label className="inline-flex items-center gap-3 rounded-2xl border border-white/8 bg-white/[0.03] px-3 py-2 text-xs font-mono text-slate-300">
      <BrainCircuit size={14} className="text-cyan-300" />
      <span className="uppercase tracking-[0.18em] text-slate-500">Model</span>
      <select
        value={model}
        onChange={event => setModel(event.target.value as PredictionModel)}
        className="rounded-xl border border-white/8 bg-surface-950 px-3 py-1.5 text-xs uppercase tracking-[0.18em] text-white outline-none transition focus:border-cyan-400/40"
      >
        {MODEL_OPTIONS.map(option => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}
