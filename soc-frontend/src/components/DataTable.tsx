import { useState, useMemo } from 'react';
import { Alert, PredictionLabel, SEVERITY_ORDER } from '../types/alert';
import { SeverityBadge, PredictionBadge, StatusBadge } from './SeverityBadge';
import { useAlertStore } from '../lib/store';
import { format } from 'date-fns';
import { ChevronUp, ChevronDown, Search, ArrowUpDown } from 'lucide-react';
import clsx from 'clsx';

type SortKey = 'timestamp' | 'severity' | 'confidence' | 'prediction' | 'status';
type SortDir = 'asc' | 'desc';

interface DataTableProps {
  alerts: Alert[];
  maxHeight?: string;
}

const predictionBarTone: Record<PredictionLabel, string> = {
  0: '#22c55e',
  1: '#eab308',
  2: '#ef4444',
};

export function DataTable({ alerts, maxHeight = '560px' }: DataTableProps) {
  const { setSelectedAlert, selectedAlert } = useAlertStore();
  const [search, setSearch] = useState('');
  const [filterPred, setFilterPred] = useState<PredictionLabel | 'all'>('all');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [sortKey, setSortKey] = useState<SortKey>('timestamp');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [page, setPage] = useState(0);
  const pageSize = 15;

  const filtered = useMemo(() => {
    let r = alerts;
    if (search) {
      const q = search.toLowerCase();
      r = r.filter(a =>
        a.id.toLowerCase().includes(q) ||
        a.sourceIp.toLowerCase().includes(q) ||
        a.destIp.toLowerCase().includes(q) ||
        a.protocol.toLowerCase().includes(q)
      );
    }
    if (filterPred !== 'all') r = r.filter(a => a.prediction === filterPred);
    if (filterStatus !== 'all') r = r.filter(a => a.status === filterStatus);

    r = [...r].sort((a, b) => {
      let va: number;
      let vb: number;
      switch (sortKey) {
        case 'timestamp': va = a.detectedAt.getTime(); vb = b.detectedAt.getTime(); break;
        case 'severity': va = SEVERITY_ORDER[a.severity]; vb = SEVERITY_ORDER[b.severity]; break;
        case 'confidence': va = a.confidence; vb = b.confidence; break;
        case 'prediction': va = a.prediction; vb = b.prediction; break;
        default: return 0;
      }
      return sortDir === 'asc' ? va - vb : vb - va;
    });
    return r;
  }, [alerts, search, filterPred, filterStatus, sortKey, sortDir]);

  const paged = filtered.slice(page * pageSize, (page + 1) * pageSize);
  const totalPages = Math.ceil(filtered.length / pageSize);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else {
      setSortKey(key);
      setSortDir('desc');
    }
  };

  const SortIcon = ({ k }: { k: SortKey }) => (
    sortKey === k
      ? sortDir === 'desc' ? <ChevronDown size={12} className="text-cyan-300" /> : <ChevronUp size={12} className="text-cyan-300" />
      : <ArrowUpDown size={10} className="text-slate-600" />
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative min-w-[240px] flex-1">
          <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            value={search}
            onChange={e => {
              setSearch(e.target.value);
              setPage(0);
            }}
            placeholder="Search alert ID, source, destination, or model..."
            className="soc-input pl-9"
          />
        </div>

        <select
          value={filterPred}
          onChange={e => {
            setFilterPred(e.target.value as any);
            setPage(0);
          }}
          className="soc-input w-auto min-w-[180px]"
        >
          <option value="all">All Predictions</option>
          <option value={0}>False Positive</option>
          <option value={1}>Benign Positive</option>
          <option value={2}>True Positive</option>
        </select>

        <select
          value={filterStatus}
          onChange={e => {
            setFilterStatus(e.target.value);
            setPage(0);
          }}
          className="soc-input w-auto min-w-[180px]"
        >
          <option value="all">All Statuses</option>
          <option value="open">Open</option>
          <option value="investigating">Investigating</option>
          <option value="resolved">Resolved</option>
          <option value="escalated">Escalated</option>
          <option value="remediation">Remediation</option>
        </select>

        <div className="ml-auto rounded-xl border border-white/8 bg-white/[0.03] px-3 py-2 text-[11px] font-mono text-slate-400">
          {filtered.length} / {alerts.length} predictions visible
        </div>
      </div>

      <div className="overflow-hidden rounded-2xl border border-white/8 bg-surface-900/50">
        <div className="overflow-x-auto" style={{ maxHeight }}>
          <table className="w-full min-w-[980px] border-separate border-spacing-0 text-sm">
            <thead className="sticky top-0 z-10 bg-surface-900/95 backdrop-blur">
              <tr>
                {[
                  { key: 'id', label: 'Alert ID', sortable: false },
                  { key: 'timestamp', label: 'Detected', sortable: true },
                  { key: 'sourceIp', label: 'Network Path', sortable: false },
                  { key: 'severity', label: 'Severity', sortable: true },
                  { key: 'prediction', label: 'Prediction', sortable: true },
                  { key: 'confidence', label: 'Confidence', sortable: true },
                  { key: 'status', label: 'Status', sortable: false },
                ].map(col => (
                  <th
                    key={col.key}
                    onClick={() => col.sortable && toggleSort(col.key as SortKey)}
                    className={clsx(
                      'border-b border-white/8 px-4 py-3 text-left text-[10px] font-mono uppercase tracking-[0.22em] text-slate-500',
                      col.sortable && 'cursor-pointer select-none hover:text-slate-200',
                    )}
                  >
                    <span className="flex items-center gap-1.5">
                      {col.label}
                      {col.sortable && <SortIcon k={col.key as SortKey} />}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {paged.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-16 text-center text-xs font-mono text-slate-500">
                    {alerts.length === 0 ? 'No backend predictions added yet' : 'No backend predictions match current filters'}
                  </td>
                </tr>
              ) : paged.map(alert => {
                const isSelected = selectedAlert?.id === alert.id;
                return (
                  <tr
                    key={alert.id}
                    onClick={() => setSelectedAlert(alert)}
                    className={clsx(
                      'soc-table-row cursor-pointer transition-all duration-200',
                      isSelected && 'outline outline-1 outline-cyan-400/40',
                    )}
                  >
                    <td className="border-b border-white/[0.05] px-4 py-3.5">
                      <div className="space-y-1">
                        <span className="text-sm font-display font-semibold text-cyan-200">{alert.id}</span>
                        <p className="text-[10px] font-mono uppercase tracking-[0.2em] text-slate-600">{alert.protocol}</p>
                      </div>
                    </td>
                    <td className="border-b border-white/[0.05] px-4 py-3.5">
                      <div className="space-y-1">
                        <span className="font-mono text-[11px] text-slate-300">{format(alert.detectedAt, 'HH:mm:ss')}</span>
                        <p className="font-mono text-[10px] text-slate-600">{format(alert.detectedAt, 'MMM d')}</p>
                      </div>
                    </td>
                    <td className="border-b border-white/[0.05] px-4 py-3.5">
                      <div className="space-y-1">
                        <div className="flex items-center gap-2 font-mono text-[11px]">
                          <span className="text-emerald-300/90">{alert.sourceIp}</span>
                          <span className="text-slate-600">to</span>
                          <span className="text-slate-300">{alert.destIp}</span>
                        </div>
                        <p className="font-mono text-[10px] text-slate-600">port {alert.port}</p>
                      </div>
                    </td>
                    <td className="border-b border-white/[0.05] px-4 py-3.5">
                      <SeverityBadge severity={alert.severity} />
                    </td>
                    <td className="border-b border-white/[0.05] px-4 py-3.5">
                      <PredictionBadge prediction={alert.prediction} size="sm" />
                    </td>
                    <td className="border-b border-white/[0.05] px-4 py-3.5">
                      <div className="space-y-2">
                        <div className="h-2 w-28 overflow-hidden rounded-full bg-white/[0.06]">
                          <div
                            className="h-full rounded-full transition-all duration-300"
                            style={{
                              width: `${alert.confidence * 100}%`,
                              backgroundColor: predictionBarTone[alert.prediction],
                              boxShadow: `0 0 18px ${predictionBarTone[alert.prediction]}55`,
                            }}
                          />
                        </div>
                        <span className="font-mono text-[11px] text-slate-300">{(alert.confidence * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                    <td className="border-b border-white/[0.05] px-4 py-3.5">
                      <StatusBadge status={alert.status} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-mono text-slate-500">
            Page {page + 1} of {totalPages}
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage(p => Math.max(0, p - 1))}
              disabled={page === 0}
              className="soc-button-secondary px-3 py-2 text-xs font-mono disabled:cursor-not-allowed disabled:opacity-40"
            >
              Prev
            </button>
            {Array.from({ length: Math.min(5, totalPages) }, (_, i) => i + Math.max(0, page - 2)).map(p => (
              <button
                key={p}
                onClick={() => setPage(p)}
                className={clsx(
                  'h-9 w-9 rounded-xl border text-xs font-mono transition-all duration-200',
                  p === page
                    ? 'border-cyan-400/30 bg-cyan-400/15 text-cyan-100'
                    : 'border-white/8 bg-white/[0.03] text-slate-400 hover:border-white/15 hover:text-white',
                )}
              >
                {p + 1}
              </button>
            ))}
            <button
              onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              className="soc-button-secondary px-3 py-2 text-xs font-mono disabled:cursor-not-allowed disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
