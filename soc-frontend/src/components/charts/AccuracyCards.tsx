interface AccuracyCardsProps {
  accuracy: number;
  macroF1: number;
}

function MetricCard({ label, value, accent }: { label: string; value: number; accent: string }) {
  return (
    <div className="soc-kpi rounded-2xl p-5">
      <p className="mb-2 text-[11px] font-mono uppercase tracking-[0.22em] text-slate-500">{label}</p>
      <p className="text-3xl font-display font-bold tracking-tight" style={{ color: accent }}>
        {(value * 100).toFixed(1)}%
      </p>
    </div>
  );
}

export function AccuracyCards({ accuracy, macroF1 }: AccuracyCardsProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <MetricCard label="Accuracy" value={accuracy} accent="#3b82f6" />
      <MetricCard label="Macro F1" value={macroF1} accent="#f59e0b" />
    </div>
  );
}
