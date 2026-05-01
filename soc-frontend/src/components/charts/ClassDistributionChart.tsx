import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

const CLASS_COLORS = {
  FalsePositive: '#22c55e',
  BenignPositive: '#eab308',
  TruePositive: '#ef4444',
};

interface ClassDistributionChartProps {
  confusionMatrix: number[][];
}

export function ClassDistributionChart({ confusionMatrix }: ClassDistributionChartProps) {
  const labels = ['FalsePositive', 'BenignPositive', 'TruePositive'] as const;
  const data = labels.map((label, index) => ({
    name: label,
    value: confusionMatrix[index]?.reduce((sum, item) => sum + item, 0) ?? 0,
    color: CLASS_COLORS[label],
  }));

  return (
    <div className="soc-panel rounded-2xl border border-white/8 p-5">
      <p className="mb-3 text-xs font-mono uppercase tracking-[0.24em] text-slate-400">Class Distribution</p>
      <ResponsiveContainer width="100%" height={260}>
        <PieChart>
          <Pie data={data} dataKey="value" nameKey="name" innerRadius={58} outerRadius={88} paddingAngle={3}>
            {data.map(entry => (
              <Cell key={entry.name} fill={entry.color} stroke="transparent" />
            ))}
          </Pie>
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
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
