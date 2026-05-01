import { BarChart3, AlertTriangle } from 'lucide-react';
import { ModelSelector } from '../components/ModelSelector';
import { useMetrics } from '../hooks/useMetrics';
import { AccuracyCards } from '../components/charts/AccuracyCards';
import { ConfidenceDistributionChart } from '../components/charts/ConfidenceDistributionChart';
import { ClassDistributionChart } from '../components/charts/ClassDistributionChart';
import { ConfusionMatrix } from '../components/charts/ConfusionMatrix';
import { useModel } from '../context/ModelContext';
import { InfoPill, PageHeader, PageShell, Panel } from '../components/ui';

function LoadingSkeleton() {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="soc-skeleton h-32" />
        <div className="soc-skeleton h-32" />
      </div>
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <div className="soc-skeleton h-80" />
        <div className="soc-skeleton h-80" />
      </div>
    </div>
  );
}

export default function Analytics() {
  const { model } = useModel();
  const { loading, error, data } = useMetrics(model);

  return (
    <PageShell className="space-y-6 animate-fade-in">
      <PageHeader
        eyebrow="Analytics"
        title="Backend evaluation breakdown"
        subtitle="Validation metrics rendered for fast inspection, with consistent class color-coding across every chart and summary surface."
        actions={
          <>
            <ModelSelector />
            <InfoPill icon={BarChart3} label={`Live endpoint: /metrics?model=${model}`} tone="info" />
          </>
        }
      />

      {loading && <LoadingSkeleton />}

      {!loading && error && (
        <Panel className="border-red-500/15 bg-red-500/[0.05]">
          <div className="flex items-start gap-3">
            <div className="rounded-xl border border-red-500/20 bg-red-500/10 p-2.5">
              <AlertTriangle size={16} className="text-red-300" />
            </div>
            <div>
              <p className="text-sm font-display font-semibold text-red-100">Failed to load analytics</p>
              <p className="mt-1 text-[11px] font-mono leading-5 text-red-200/80">{error}</p>
            </div>
          </div>
        </Panel>
      )}

      {!loading && data && (
        <div className="space-y-4">
          <AccuracyCards accuracy={data.accuracy} macroF1={data.macro_f1} />

          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <ClassDistributionChart distribution={data.class_distribution} />
            <ConfusionMatrix matrix={data.confusion_matrix} />
          </div>

          <ConfidenceDistributionChart data={data.confidence_distribution} />

          <Panel>
            <p className="mb-4 text-xs font-mono uppercase tracking-[0.24em] text-slate-400">Per-Class Metrics</p>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              {Object.entries(data.per_class).map(([label, stats]) => (
                <div key={label} className="rounded-2xl border border-white/8 bg-surface-900/55 p-4">
                  <p className="text-base font-display font-semibold text-white">{label}</p>
                  <div className="mt-4 space-y-2 text-[11px] font-mono text-slate-400">
                    <p>Precision: {stats.precision !== undefined ? `${(stats.precision * 100).toFixed(1)}%` : 'n/a'}</p>
                    <p>Recall: {stats.recall !== undefined ? `${(stats.recall * 100).toFixed(1)}%` : 'n/a'}</p>
                    <p>F1: {stats.f1 !== undefined ? `${(stats.f1 * 100).toFixed(1)}%` : 'n/a'}</p>
                    <p>Support: {stats.support ?? 'n/a'}</p>
                  </div>
                </div>
              ))}
            </div>
          </Panel>
        </div>
      )}
    </PageShell>
  );
}
