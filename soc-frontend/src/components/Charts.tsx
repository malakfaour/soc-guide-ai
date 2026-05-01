import {
  BarChart, Bar, Cell, XAxis, YAxis, Tooltip, ResponsiveContainer
} from 'recharts';
import { PREDICTION_COLORS } from '../types/alert';

const CHART_COLORS = {
  text: '#64748b',
  axis: '#94a3b8',
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl border border-white/10 bg-surface-900/95 px-3 py-2 text-xs font-mono shadow-2xl backdrop-blur">
      {label && <p className="mb-1 text-slate-400">{label}</p>}
      {payload.map((p: any, i: number) => (
        <p key={i} style={{ color: p.color || p.fill }}>
          {p.name}: <span className="font-bold text-white">{p.value}</span>
        </p>
      ))}
    </div>
  );
};

interface ProbabilityBarChartProps {
  probabilities: number[];
}

export function ProbabilityBarChart({ probabilities }: ProbabilityBarChartProps) {
  const data = [
    { name: 'False Positive', value: Math.round(probabilities[0] * 1000) / 10, fill: PREDICTION_COLORS[0] },
    { name: 'Benign Positive', value: Math.round(probabilities[1] * 1000) / 10, fill: PREDICTION_COLORS[1] },
    { name: 'True Positive', value: Math.round(probabilities[2] * 1000) / 10, fill: PREDICTION_COLORS[2] },
  ];
  return (
    <ResponsiveContainer width="100%" height={140}>
      <BarChart data={data} layout="vertical" barSize={16} margin={{ left: 8, right: 18, top: 6, bottom: 6 }}>
        <XAxis
          type="number"
          domain={[0, 100]}
          tick={{ fill: CHART_COLORS.text, fontSize: 10, fontFamily: 'JetBrains Mono' }}
          axisLine={false}
          tickLine={false}
          unit="%"
        />
        <YAxis
          type="category"
          dataKey="name"
          tick={{ fill: CHART_COLORS.axis, fontSize: 11, fontFamily: 'JetBrains Mono' }}
          axisLine={false}
          tickLine={false}
          width={118}
        />
        <Tooltip content={<CustomTooltip />} />
        <Bar dataKey="value" radius={[0, 3, 3, 0]} name="Probability">
          {data.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
