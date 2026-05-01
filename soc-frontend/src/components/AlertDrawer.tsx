import { useState } from 'react';
import { format } from 'date-fns';
import {
  X,
  Shield,
  Flag,
  ArrowUpCircle,
  Wrench,
  Network,
  Clock,
  ChevronRight,
  Activity,
  Binary,
} from 'lucide-react';
import { useAlertStore } from '../lib/store';
import { PredictionLabel } from '../types/alert';
import { PredictionBadge, SeverityBadge, StatusBadge } from './SeverityBadge';
import { ProbabilityBarChart } from './Charts';
import { remediationPredict } from '../services/api';
import { deriveIncidentFeatures } from '../lib/remediation';

export function AlertDrawer() {
  const {
    selectedAlert,
    setSelectedAlert,
    updateAlertStatus,
    updateAlertPrediction,
    sendToRemediation,
    escalateAlert,
    addToast,
  } = useAlertStore();
  const [remediationLoading, setRemediationLoading] = useState(false);

  if (!selectedAlert) return null;
  const a = selectedAlert;

  const handleAction = async (action: string) => {
    switch (action) {
      case 'tp':
        updateAlertPrediction(a.id, 2);
        addToast({ type: 'error', title: 'Marked as True Positive', message: `${a.id} - escalated for review` });
        break;
      case 'fp':
        updateAlertPrediction(a.id, 0);
        addToast({ type: 'success', title: 'Marked as False Positive', message: `${a.id} - closed` });
        break;
      case 'escalate':
        escalateAlert(a.id);
        addToast({ type: 'warning', title: 'Alert Escalated', message: `${a.id} - sent to L2 analyst` });
        break;
      case 'remediate':
        try {
          setRemediationLoading(true);
          const incidentFeatures = deriveIncidentFeatures(a);
          const remediation = await remediationPredict(incidentFeatures);
          sendToRemediation(a.id, remediation, incidentFeatures);
          addToast({ type: 'info', title: 'Sent to Remediation', message: `${a.id} - backend remediation workflow started` });
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown remediation error';
          addToast({ type: 'error', title: 'Remediation failed', message });
        } finally {
          setRemediationLoading(false);
        }
        break;
      case 'investigate':
        updateAlertStatus(a.id, 'investigating');
        addToast({ type: 'info', title: 'Investigation Started', message: a.id });
        break;
    }
  };

  const conf = Math.max(...a.probabilities);
  const predColor: Record<PredictionLabel, string> = {
    0: '#22c55e',
    1: '#eab308',
    2: '#ef4444',
  };

  const sectionClass = 'rounded-2xl border border-white/8 bg-surface-900/50 p-4';

  return (
    <>
      <div className="fixed inset-0 z-40 bg-slate-950/70 backdrop-blur-sm animate-fade-in" onClick={() => setSelectedAlert(null)} />

      <div className="fixed right-0 top-0 z-50 flex h-full w-[560px] max-w-full flex-col overflow-hidden border-l border-white/8 bg-surface-850 shadow-[-24px_0_60px_rgba(2,6,23,0.45)] animate-slide-in">
        <div className="border-b border-white/8 bg-surface-900/95 px-6 py-5 backdrop-blur">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <div className="mb-2 flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-full animate-pulse" style={{ backgroundColor: predColor[a.prediction], boxShadow: `0 0 16px ${predColor[a.prediction]}aa` }} />
                <p className="text-[10px] font-mono uppercase tracking-[0.32em] text-slate-500">Alert Details</p>
              </div>
              <p className="truncate text-xl font-display font-bold tracking-tight text-white">{a.id}</p>
              <p className="mt-1 text-xs font-mono text-slate-500">Analyst decision support for prediction, context, and response actions.</p>
            </div>
            <button
              onClick={() => setSelectedAlert(null)}
              className="rounded-xl border border-white/8 bg-white/[0.03] p-2 text-slate-400 transition-all duration-200 hover:border-white/15 hover:text-white"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        <div className="flex-1 space-y-4 overflow-y-auto px-6 py-5">
          <div className={sectionClass}>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex flex-wrap items-center gap-2">
                <SeverityBadge severity={a.severity} />
                <StatusBadge status={a.status} />
                <PredictionBadge prediction={a.prediction} size="sm" />
              </div>
              <span className="flex items-center gap-1.5 text-[11px] font-mono text-slate-500">
                <Clock size={11} />
                {format(a.detectedAt, 'MMM d, HH:mm:ss')}
              </span>
            </div>
          </div>

          <section className={sectionClass}>
            <div className="mb-4 flex items-center gap-2">
              <Activity size={14} className="text-cyan-300" />
              <p className="text-[11px] font-mono uppercase tracking-[0.24em] text-slate-400">Prediction</p>
            </div>
            <div className="mb-4 flex items-center justify-between gap-3">
              <div>
                <p className="text-[11px] font-mono text-slate-500">Highest confidence</p>
                <p className="mt-2 text-4xl font-display font-bold tracking-tight" style={{ color: predColor[a.prediction] }}>
                  {(conf * 100).toFixed(1)}%
                </p>
              </div>
              <div className="rounded-2xl border border-white/8 bg-surface-950/50 px-4 py-3 text-right">
                <p className="text-[10px] font-mono uppercase tracking-[0.22em] text-slate-600">Prediction class</p>
                <div className="mt-2">
                  <PredictionBadge prediction={a.prediction} />
                </div>
              </div>
            </div>
            <ProbabilityBarChart probabilities={a.probabilities} />
          </section>

          <section className={sectionClass}>
            <div className="mb-4 flex items-center gap-2">
              <Network size={14} className="text-cyan-300" />
              <p className="text-[11px] font-mono uppercase tracking-[0.24em] text-slate-400">Context</p>
            </div>
            <div className="grid grid-cols-2 gap-3">
              {[
                ['Source IP', a.sourceIp],
                ['Destination IP', a.destIp],
                ['Protocol', a.protocol],
                ['Port', String(a.port)],
              ].map(([label, value]) => (
                <div key={label} className="rounded-2xl border border-white/8 bg-surface-950/45 p-3">
                  <p className="text-[10px] font-mono uppercase tracking-[0.22em] text-slate-600">{label}</p>
                  <p className="mt-2 text-sm font-mono text-emerald-200">{value}</p>
                </div>
              ))}
            </div>
          </section>

          {a.remediationPrediction && (
            <section className={sectionClass}>
              <div className="mb-4 flex items-center gap-2">
                <Wrench size={14} className="text-cyan-300" />
                <p className="text-[11px] font-mono uppercase tracking-[0.24em] text-slate-400">Remediation Recommendation</p>
              </div>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div className="rounded-2xl border border-white/8 bg-surface-950/45 p-4">
                  <p className="text-[10px] font-mono uppercase tracking-[0.22em] text-slate-600">Account Response</p>
                  <p className="mt-2 text-base font-display font-semibold text-white">
                    {a.remediationPrediction.account_response.prediction ? 'Required' : 'Not Needed'}
                  </p>
                  <p className="mt-2 text-[11px] font-mono text-slate-400">
                    {(a.remediationPrediction.account_response.probability * 100).toFixed(1)}% confidence
                  </p>
                  <p className="text-[11px] font-mono text-slate-500">
                    threshold {(a.remediationPrediction.account_response.threshold * 100).toFixed(0)}%
                  </p>
                </div>
                <div className="rounded-2xl border border-white/8 bg-surface-950/45 p-4">
                  <p className="text-[10px] font-mono uppercase tracking-[0.22em] text-slate-600">Endpoint Response</p>
                  <p className="mt-2 text-base font-display font-semibold text-white">
                    {a.remediationPrediction.endpoint_response.prediction ? 'Required' : 'Not Needed'}
                  </p>
                  <p className="mt-2 text-[11px] font-mono text-slate-400">
                    {(a.remediationPrediction.endpoint_response.probability * 100).toFixed(1)}% confidence
                  </p>
                  <p className="text-[11px] font-mono text-slate-500">
                    threshold {(a.remediationPrediction.endpoint_response.threshold * 100).toFixed(0)}%
                  </p>
                </div>
              </div>
            </section>
          )}

          <section className={sectionClass}>
            <div className="mb-4 flex items-center gap-2">
              <Binary size={14} className="text-cyan-300" />
              <p className="text-[11px] font-mono uppercase tracking-[0.24em] text-slate-400">Features</p>
            </div>
            <div className="space-y-4">
              <div>
                <p className="mb-2 text-[10px] font-mono uppercase tracking-[0.22em] text-slate-600">Raw feature vector</p>
                <div className="max-h-28 overflow-y-auto rounded-2xl border border-white/8 bg-surface-950/55 p-3 font-mono text-[10px] leading-5 text-emerald-200/80 break-all">
                  [{a.features.map(feature => feature.toFixed(2)).join(', ')}]
                </div>
              </div>
              {a.incidentFeatures && (
                <div>
                  <p className="mb-2 text-[10px] font-mono uppercase tracking-[0.22em] text-slate-600">Incident remediation vector</p>
                  <div className="rounded-2xl border border-white/8 bg-surface-950/55 p-3 font-mono text-[10px] leading-5 text-cyan-200/80 break-all">
                    [{a.incidentFeatures.map(feature => feature.toFixed(2)).join(', ')}]
                  </div>
                </div>
              )}
            </div>
          </section>
        </div>

        <div className="border-t border-white/8 bg-surface-900/95 px-6 py-5">
          <p className="mb-4 text-[11px] font-mono uppercase tracking-[0.24em] text-slate-500">Actions</p>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <button
              onClick={() => handleAction('tp')}
              className="soc-button-danger flex items-center justify-center gap-2 font-mono text-xs"
            >
              <Shield size={12} /> Mark True Positive
            </button>
            <button
              onClick={() => handleAction('fp')}
              className="rounded-xl border border-emerald-400/20 bg-emerald-400/10 px-4 py-2.5 text-xs font-mono text-emerald-100 transition-all duration-200 hover:border-emerald-400/35 hover:bg-emerald-400/15"
            >
              <span className="flex items-center justify-center gap-2">
                <Flag size={12} /> Mark False Positive
              </span>
            </button>
            <button
              onClick={() => handleAction('escalate')}
              className="rounded-xl border border-amber-400/20 bg-amber-400/10 px-4 py-2.5 text-xs font-mono text-amber-100 transition-all duration-200 hover:border-amber-400/35 hover:bg-amber-400/15"
            >
              <span className="flex items-center justify-center gap-2">
                <ArrowUpCircle size={12} /> Escalate L2
              </span>
            </button>
            <button
              onClick={() => handleAction('remediate')}
              disabled={remediationLoading}
              className="rounded-xl border border-violet-400/20 bg-violet-400/10 px-4 py-2.5 text-xs font-mono text-violet-100 transition-all duration-200 hover:border-violet-400/35 hover:bg-violet-400/15 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <span className="flex items-center justify-center gap-2">
                <Wrench size={12} /> {remediationLoading ? 'Running Remediation...' : 'Send to Remediation'}
              </span>
            </button>
          </div>
          {a.status === 'open' && (
            <button
              onClick={() => handleAction('investigate')}
              className="soc-button-primary mt-3 flex w-full items-center justify-center gap-2 font-mono text-xs"
            >
              Start Investigation <ChevronRight size={12} />
            </button>
          )}
        </div>
      </div>
    </>
  );
}
