import { AlertTriangle, BarChart3, CheckCircle2, Database, Gauge, Target } from 'lucide-react';
import type { ElementType } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { ModelSelector } from '../components/ModelSelector';
import { AlertsOverTimeChart } from '../components/charts/AlertsOverTimeChart';
import { ClassDistributionChart } from '../components/charts/ClassDistributionChart';
import { useMetrics } from '../hooks/useMetrics';
import { useModel } from '../context/ModelContext';
import { InfoPill, PageHeader, PageShell, Panel, PanelHeader } from '../components/ui';

const CLASS_LABELS = ['FalsePositive', 'BenignPositive', 'TruePositive'];

const formatPercent = (value?: number) => (
  typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : 'n/a'
);

const sumMatrix = (matrix: number[][]) => matrix.reduce(
  (total, row) => total + row.reduce((rowTotal, value) => rowTotal + value, 0),
  0,
);

const diagonalSum = (matrix: number[][]) => matrix.reduce(
  (total, row, index) => total + (row[index] || 0),
  0,
);

function StatCard({
  icon: Icon,
  label,
  value,
  sub,
  color,
}: {
  icon: ElementType;
  label: string;
  value: number | string;
  sub?: string;
  color: string;
}) {
  return (
    <div className="soc-kpi rounded-2xl p-5 transition-all duration-200 hover:-translate-y-0.5 hover:border-white/15">
      <div className="mb-5 flex items-start justify-between gap-3">
        <div className="flex h-12 w-12 items-center justify-center rounded-2xl border" style={{ background: `${color}14`, borderColor: `${color}35` }}>
          <Icon size={18} style={{ color }} />
        </div>
        {sub ? <span className="rounded-full border border-white/8 bg-white/[0.03] px-2.5 py-1 text-[10px] font-mono uppercase tracking-[0.2em] text-slate-500">{sub}</span> : null}
      </div>
      <p className="text-3xl font-display font-bold tracking-tight text-white">{value}</p>
      <p className="mt-1 text-[11px] font-mono uppercase tracking-[0.22em] text-slate-500">{label}</p>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        {Array.from({ length: 4 }).map((_, index) => (
          <div key={index} className="soc-skeleton h-40" />
        ))}
      </div>
      <div className="soc-skeleton h-[360px]" />
    </div>
  );
}

function DashboardTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl border border-white/10 bg-surface-900/95 px-3 py-2 text-xs font-mono shadow-2xl backdrop-blur">
      <p className="mb-1 text-slate-300">{label}</p>
      {payload.map((item: any) => (
        <p key={item.dataKey} style={{ color: item.color }}>
          {item.name}: <span className="font-bold text-white">{item.value.toFixed(1)}%</span>
        </p>
      ))}
    </div>
  );
}

export default function Dashboard() {
  const { model } = useModel();
  const { loading, error, data } = useMetrics(model);

  const totalEvaluated = data ? sumMatrix(data.confusion_matrix) : 0;
  const correctPredictions = data ? diagonalSum(data.confusion_matrix) : 0;
  const classPerformance = data
    ? CLASS_LABELS.map(label => ({
      label,
      Precision: (data.per_class[label]?.precision || 0) * 100,
      Recall: (data.per_class[label]?.recall || 0) * 100,
      F1: (data.per_class[label]?.f1 || 0) * 100,
    }))
    : [];

  return (
    <PageShell className="space-y-6 animate-fade-in">
      <PageHeader
        eyebrow="SOC Overview"
        title="Model performance at a glance"
        subtitle="A clean analyst view of backend evaluation quality, coverage, and per-class stability for rapid decision-making."
        actions={
          <>
            <ModelSelector />
            <InfoPill icon={BarChart3} label={`Live source: /metrics?model=${model}`} tone="info" />
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
              <p className="text-sm font-display font-semibold text-red-100">Failed to load dashboard metrics</p>
              <p className="mt-1 text-[11px] font-mono leading-5 text-red-200/80">{error}</p>
            </div>
          </div>
        </Panel>
      )}

      {!loading && data && (
        <>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
            <StatCard icon={Gauge} label="Accuracy" value={formatPercent(data.accuracy)} color="#22d3ee" sub="backend" />
            <StatCard icon={Target} label="Macro F1" value={formatPercent(data.macro_f1)} color="#f59e0b" sub="backend" />
            <StatCard icon={Database} label="Evaluated Rows" value={totalEvaluated.toLocaleString()} color="#22c55e" sub="/metrics" />
            <StatCard icon={CheckCircle2} label="Correct Predictions" value={correctPredictions.toLocaleString()} color="#ef4444" sub="matrix diagonal" />
          </div>

          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <ClassDistributionChart distribution={data.class_distribution} />
            <AlertsOverTimeChart data={data.alerts_over_time} />
          </div>

          <Panel>
            <PanelHeader
              icon={BarChart3}
              title="Per-Class Model Quality"
              subtitle="Precision, recall, and F1 from backend evaluation metrics."
              action={<span className="text-[10px] font-mono uppercase tracking-[0.22em] text-slate-600">Backend metrics only</span>}
            />

            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={classPerformance} barSize={16} margin={{ top: 10, right: 8, left: -18, bottom: 0 }}>
                <CartesianGrid vertical={false} stroke="rgba(148,163,184,0.1)" />
                <XAxis
                  dataKey="label"
                  tick={{ fill: '#94a3b8', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  domain={[0, 100]}
                  tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                  axisLine={false}
                  tickLine={false}
                  unit="%"
                />
                <Tooltip content={<DashboardTooltip />} />
                <Bar dataKey="Precision" fill="#22c55e" radius={[6, 6, 0, 0]} />
                <Bar dataKey="Recall" fill="#eab308" radius={[6, 6, 0, 0]} />
                <Bar dataKey="F1" fill="#ef4444" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Panel>
        </>
      )}
    </PageShell>
  );
}
