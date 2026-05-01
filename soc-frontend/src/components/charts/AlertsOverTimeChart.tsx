import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { AlertsOverTimePoint } from '../../services/api';

interface AlertsOverTimeChartProps {
  data: AlertsOverTimePoint[];
}

export function AlertsOverTimeChart({ data }: AlertsOverTimeChartProps) {
  return (
    <div className="soc-panel rounded-2xl border border-white/8 p-5">
      <p className="mb-3 text-xs font-mono uppercase tracking-[0.24em] text-slate-400">Alerts Over Time</p>
      <ResponsiveContainer width="100%" height={280}>
        <AreaChart data={data} margin={{ top: 10, right: 8, left: -18, bottom: 0 }}>
          <defs>
            <linearGradient id="fpGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#22c55e" stopOpacity={0.35} />
              <stop offset="100%" stopColor="#22c55e" stopOpacity={0.05} />
            </linearGradient>
            <linearGradient id="bpGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#eab308" stopOpacity={0.35} />
              <stop offset="100%" stopColor="#eab308" stopOpacity={0.05} />
            </linearGradient>
            <linearGradient id="tpGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#ef4444" stopOpacity={0.35} />
              <stop offset="100%" stopColor="#ef4444" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid vertical={false} stroke="rgba(148,163,184,0.1)" />
          <XAxis
            dataKey="date"
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
          <Legend
            formatter={value => <span className="text-xs font-mono text-slate-400">{value}</span>}
            iconType="circle"
            iconSize={8}
          />
          <Area type="monotone" dataKey="FalsePositive" stroke="#22c55e" fill="url(#fpGradient)" strokeWidth={2} />
          <Area type="monotone" dataKey="BenignPositive" stroke="#eab308" fill="url(#bpGradient)" strokeWidth={2} />
          <Area type="monotone" dataKey="TruePositive" stroke="#ef4444" fill="url(#tpGradient)" strokeWidth={2} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
