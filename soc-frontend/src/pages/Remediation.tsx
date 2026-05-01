import { useState } from 'react';
import type { ElementType } from 'react';
import { useAlertStore } from '../lib/store';
import { PredictionBadge, SeverityBadge } from '../components/SeverityBadge';
import { AlertDrawer } from '../components/AlertDrawer';
import { Alert, RemediationAction } from '../types/alert';
import { format } from 'date-fns';
import {
  CheckCircle,
  XCircle,
  Plus,
  Wrench,
  Shield,
  UserX,
  ServerCrash,
  FileText,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import clsx from 'clsx';
import { EmptyState, InfoPill, PageHeader, PageShell, Panel } from '../components/ui';

function ActionIcon({ type }: { type: RemediationAction['type'] }) {
  const icons: Record<RemediationAction['type'], ElementType> = {
    block_ip: Shield,
    disable_account: UserX,
    isolate_endpoint: ServerCrash,
    custom: FileText,
  };
  const Icon = icons[type];
  return <Icon size={13} />;
}

function RemediationCard({ alert }: { alert: Alert }) {
  const { updateRemediationAction, addRemediationAction, setSelectedAlert, addToast } = useAlertStore();
  const [expanded, setExpanded] = useState(true);
  const [noteInputs, setNoteInputs] = useState<Record<string, string>>({});

  const actions = alert.remediationActions || [];

  const handleAction = (actionId: string, status: 'approved' | 'rejected') => {
    updateRemediationAction(alert.id, actionId, status, noteInputs[actionId]);
    addToast({
      type: status === 'approved' ? 'success' : 'warning',
      title: `Action ${status}`,
      message: `${alert.id} - remediation step ${status}`,
    });
  };

  return (
    <Panel className="overflow-hidden p-0">
      <div className="flex flex-col gap-4 border-b border-white/8 px-5 py-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-violet-300 animate-pulse" />
            <span className="text-lg font-display font-semibold text-white">{alert.id}</span>
            <PredictionBadge prediction={alert.prediction} size="sm" />
            <SeverityBadge severity={alert.severity} />
          </div>
          <div className="flex flex-wrap items-center gap-3 text-[11px] font-mono text-slate-500">
            <span>{alert.sourceIp} to {alert.destIp}</span>
            <span>port {alert.port}</span>
            <span>{format(alert.detectedAt, 'MMM d, HH:mm')}</span>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => setSelectedAlert(alert)}
            className="soc-button-secondary px-3 py-2 text-xs font-mono"
          >
            View Details
          </button>
          <button
            onClick={() => setExpanded(e => !e)}
            className="soc-button-secondary px-3 py-2 text-xs font-mono"
          >
            <span className="flex items-center gap-2">
              {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
              {expanded ? 'Collapse' : 'Expand'}
            </span>
          </button>
        </div>
      </div>

      {expanded && (
        <div className="space-y-4 p-5">
          {alert.remediationPrediction && (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="rounded-2xl border border-white/8 bg-surface-900/55 p-4">
                <p className="text-[10px] font-mono uppercase tracking-[0.22em] text-slate-600">Account Response</p>
                <p className="mt-2 text-base font-display font-semibold text-white">
                  {alert.remediationPrediction.account_response.prediction ? 'Required' : 'Not Needed'}
                </p>
                <p className="mt-2 text-[11px] font-mono text-slate-400">
                  {(alert.remediationPrediction.account_response.probability * 100).toFixed(1)}% confidence
                </p>
                <p className="text-[11px] font-mono text-slate-500">
                  threshold {(alert.remediationPrediction.account_response.threshold * 100).toFixed(0)}%
                </p>
              </div>
              <div className="rounded-2xl border border-white/8 bg-surface-900/55 p-4">
                <p className="text-[10px] font-mono uppercase tracking-[0.22em] text-slate-600">Endpoint Response</p>
                <p className="mt-2 text-base font-display font-semibold text-white">
                  {alert.remediationPrediction.endpoint_response.prediction ? 'Required' : 'Not Needed'}
                </p>
                <p className="mt-2 text-[11px] font-mono text-slate-400">
                  {(alert.remediationPrediction.endpoint_response.probability * 100).toFixed(1)}% confidence
                </p>
                <p className="text-[11px] font-mono text-slate-500">
                  threshold {(alert.remediationPrediction.endpoint_response.threshold * 100).toFixed(0)}%
                </p>
              </div>
            </div>
          )}

          {actions.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-white/8 bg-white/[0.02] px-4 py-10 text-center text-xs font-mono text-slate-500">
              No remediation actions defined
            </div>
          ) : actions.map(action => (
            <div
              key={action.id}
              className={clsx(
                'rounded-2xl border p-4 transition-all duration-200',
                action.status === 'approved' ? 'border-emerald-400/18 bg-emerald-400/[0.05]' :
                  action.status === 'rejected' ? 'border-red-500/18 bg-red-500/[0.05] opacity-70' :
                    'border-white/8 bg-surface-900/55',
              )}
            >
              <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                <div className="space-y-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <div className="rounded-xl border border-white/8 bg-white/[0.03] p-2 text-slate-300">
                      <ActionIcon type={action.type} />
                    </div>
                    <span className="text-sm font-display font-semibold text-slate-100">{action.label}</span>
                    {action.status !== 'pending' && (
                      <span className={clsx(
                        'rounded-full border px-2.5 py-1 text-[10px] font-mono uppercase tracking-[0.18em]',
                        action.status === 'approved'
                          ? 'border-emerald-400/20 bg-emerald-400/12 text-emerald-200'
                          : 'border-red-500/20 bg-red-500/12 text-red-200',
                      )}>
                        {action.status}
                      </span>
                    )}
                  </div>
                  {action.status === 'pending' ? (
                    <input
                      value={noteInputs[action.id] || ''}
                      onChange={e => setNoteInputs(n => ({ ...n, [action.id]: e.target.value }))}
                      placeholder="Add analyst note (optional)..."
                      className="soc-input max-w-xl text-xs font-mono"
                    />
                  ) : action.notes ? (
                    <p className="text-[11px] font-mono italic text-slate-400">Note: {action.notes}</p>
                  ) : null}
                </div>

                {action.status === 'pending' && (
                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      onClick={() => handleAction(action.id, 'approved')}
                      className="rounded-xl border border-emerald-400/20 bg-emerald-400/10 px-3 py-2 text-xs font-mono text-emerald-100 transition-all duration-200 hover:border-emerald-400/35 hover:bg-emerald-400/15"
                    >
                      <span className="flex items-center gap-1.5">
                        <CheckCircle size={11} /> Approve
                      </span>
                    </button>
                    <button
                      onClick={() => handleAction(action.id, 'rejected')}
                      className="rounded-xl border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs font-mono text-red-100 transition-all duration-200 hover:border-red-400/35 hover:bg-red-500/15"
                    >
                      <span className="flex items-center gap-1.5">
                        <XCircle size={11} /> Reject
                      </span>
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}

          <div className="border-t border-white/8 pt-4">
            <p className="mb-3 text-[11px] font-mono uppercase tracking-[0.22em] text-slate-500">Add Action</p>
            <div className="flex flex-wrap gap-2">
              {(['block_ip', 'disable_account', 'isolate_endpoint'] as const).map(type => {
                const labels: Record<string, string> = {
                  block_ip: 'Block IP',
                  disable_account: 'Disable Account',
                  isolate_endpoint: 'Isolate Endpoint',
                };
                return (
                  <button
                    key={type}
                    onClick={() => addRemediationAction(alert.id, type)}
                    className="soc-button-secondary px-3 py-2 text-xs font-mono"
                  >
                    <span className="flex items-center gap-1.5">
                      <Plus size={10} /> {labels[type]}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </Panel>
  );
}

export default function Remediation() {
  const { alerts, selectedAlert } = useAlertStore();
  const remediationAlerts = alerts.filter(a => a.status === 'remediation');

  return (
    <div className="flex h-full min-w-0">
      <PageShell className="flex-1 space-y-6 animate-fade-in">
        <PageHeader
          eyebrow="Remediation"
          title="Review and approve response actions"
          subtitle="Preserve analyst control while surfacing backend remediation recommendations with clearer decision states and notes."
          actions={<InfoPill icon={Wrench} label={`${remediationAlerts.length} pending remediation`} tone="info" />}
        />

        {remediationAlerts.length === 0 ? (
          <EmptyState
            icon={CheckCircle}
            title="No alerts in remediation"
            subtitle="Send alerts to remediation from the Triage page and they will appear here with approval workflows and action notes."
          />
        ) : (
          <div className="space-y-4">
            {remediationAlerts.map(alert => (
              <RemediationCard key={alert.id} alert={alert} />
            ))}
          </div>
        )}
      </PageShell>

      {selectedAlert && <AlertDrawer />}
    </div>
  );
}
