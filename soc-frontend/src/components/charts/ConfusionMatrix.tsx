const CLASS_LABELS = ['False Positive', 'Benign Positive', 'True Positive'];

interface ConfusionMatrixProps {
  matrix: number[][];
}

export function ConfusionMatrix({ matrix }: ConfusionMatrixProps) {
  const maxValue = Math.max(...matrix.flat(), 1);

  const cellStyle = (value: number) => {
    const intensity = value / maxValue;
    return {
      backgroundColor: `rgba(6, 182, 212, ${0.1 + intensity * 0.38})`,
      borderColor: `rgba(6, 182, 212, ${0.12 + intensity * 0.26})`,
    };
  };

  return (
    <div className="soc-panel rounded-2xl border border-white/8 p-5">
      <p className="mb-3 text-xs font-mono uppercase tracking-[0.24em] text-slate-400">Confusion Matrix</p>
      <div className="overflow-x-auto">
        <table className="w-full border-separate border-spacing-2 text-center">
          <thead>
            <tr>
              <th className="px-2 py-1 text-[10px] font-mono uppercase tracking-[0.22em] text-slate-500">Actual / Pred</th>
              {CLASS_LABELS.map(label => (
                <th key={label} className="px-2 py-1 text-[10px] font-mono uppercase tracking-[0.22em] text-slate-400">
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, rowIndex) => (
              <tr key={CLASS_LABELS[rowIndex]}>
                <td className="px-2 py-2 text-[10px] font-mono uppercase tracking-[0.22em] text-slate-400">
                  {CLASS_LABELS[rowIndex]}
                </td>
                {row.map((value, columnIndex) => (
                  <td
                    key={`${rowIndex}-${columnIndex}`}
                    className="rounded-xl border px-3 py-4 text-sm font-mono font-semibold text-white"
                    style={cellStyle(value)}
                  >
                    {value}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
