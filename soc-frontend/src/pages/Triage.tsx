import { useAlertStore } from '../lib/store';
import { DataTable } from '../components/DataTable';
import { AlertDrawer } from '../components/AlertDrawer';
import { PredictionPanel } from '../components/PredictionPanel';
import { AlertTriangle, Table2, Zap } from 'lucide-react';
import { InfoPill, PageHeader, PageShell, Panel, PanelHeader } from '../components/ui';

export default function Triage() {
  const { alerts, selectedAlert } = useAlertStore();

  return (
    <div className="flex h-full min-w-0">
      <PageShell className="min-w-0 flex-1 space-y-6 animate-fade-in">
        <PageHeader
          eyebrow="Triage Queue"
          title="Investigate and classify incoming alerts"
          subtitle="Run live model predictions, review feature vectors, and move through the queue with clean visual priority cues."
          actions={
            <>
              <InfoPill icon={AlertTriangle} label={`${alerts.filter(a => a.status === 'open').length} open alerts`} tone="danger" />
              <InfoPill label={`${alerts.filter(a => a.status === 'investigating').length} investigating`} tone="warning" />
            </>
          }
        />

        <Panel className="overflow-hidden">
          <PanelHeader
            icon={Zap}
            title="Real-Time Prediction Test"
            subtitle="POST /predict endpoint validation against preprocessed feature vectors."
            action={<span className="text-[10px] font-mono uppercase tracking-[0.22em] text-slate-600">localhost:8000/predict</span>}
          />
          <PredictionPanel />
        </Panel>

        <Panel className="overflow-hidden" padded={false}>
          <div className="p-5 sm:p-6">
            <PanelHeader
              icon={Table2}
              title="Backend Prediction Queue"
              subtitle="Dense, readable triage table with fast scanning and clear confidence/status treatment."
            />
          </div>
          <div className="px-5 pb-5 sm:px-6 sm:pb-6">
            <DataTable alerts={alerts} />
          </div>
        </Panel>
      </PageShell>

      {selectedAlert && <AlertDrawer />}
    </div>
  );
}
