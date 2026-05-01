import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { ConfidenceDistributionBucket } from '../../services/api';

interface ConfidenceDistributionChartProps {
  data: ConfidenceDistributionBucket[];
}

export function ConfidenceDistributionChart({ data }: ConfidenceDistributionChartProps) {
  return (
    <div className="soc-panel rounded-2xl border border-white/8 p-5">
      <p className="mb-3 text-xs font-mono uppercase tracking-[0.24em] text-slate-400">Confidence Distribution</p>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ top: 10, right: 8, left: -18, bottom: 0 }}>
          <CartesianGrid vertical={false} stroke="rgba(148,163,184,0.1)" />
          <XAxis
            dataKey="bucket"
            tick={{ fill: '#94a3b8', fontSize: 10, fontFamily: 'JetBrains Mono' }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            contentStyle={{
              background: 'rgba(15, 22, 41, 0.96)',
              border: '1px solid rgba(255,255,255,0.08)',
              borderRadius: '12px',
              color: '#e5e7eb',
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '12px',
            }}
          />
          <Bar dataKey="count" name="Alerts" fill="#38bdf8" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
