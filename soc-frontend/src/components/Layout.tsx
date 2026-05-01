import { NavLink, Outlet, useLocation } from 'react-router-dom';
import {
  AlertTriangle,
  BarChart3,
  LayoutDashboard,
  ShieldCheck,
  Terminal,
  Wifi,
  WifiOff,
  Wrench,
} from 'lucide-react';
import { useAlertStore } from '../lib/store';
import { ToastContainer } from '../components/Toast';
import { useEffect, useMemo, useState } from 'react';
import { BackendHealthResponse, healthStatus } from '../services/api';
import clsx from 'clsx';

function NavItem({ to, icon: Icon, label, badge }: { to: string; icon: any; label: string; badge?: number }) {
  return (
    <NavLink
      to={to}
      end={to === '/'}
      className={({ isActive }) =>
        clsx(
          'group relative flex items-center gap-3 rounded-2xl border px-3.5 py-3 transition-all duration-200',
          isActive
            ? 'border-cyan-400/25 bg-cyan-400/10 text-white shadow-[0_12px_30px_rgba(6,182,212,0.08)]'
            : 'border-transparent text-slate-400 hover:border-white/8 hover:bg-white/[0.03] hover:text-slate-100',
        )
      }
    >
      {({ isActive }) => (
        <>
          <span
            className={clsx(
              'flex h-9 w-9 items-center justify-center rounded-xl border transition-all duration-200',
              isActive
                ? 'border-cyan-400/20 bg-cyan-400/15 text-cyan-200'
                : 'border-white/6 bg-white/[0.03] text-slate-500 group-hover:border-white/10 group-hover:text-slate-200',
            )}
          >
            <Icon size={16} />
          </span>
          <div className="min-w-0 flex-1">
            <p className="text-sm font-display font-semibold tracking-wide">{label}</p>
            <p className="text-[10px] font-mono uppercase tracking-[0.22em] text-slate-600 group-hover:text-slate-500">
              SOC Workspace
            </p>
          </div>
          {badge !== undefined && badge > 0 && (
            <span
              className={clsx(
                'ml-auto min-w-[28px] rounded-full px-2 py-1 text-center text-[10px] font-mono font-semibold',
                isActive ? 'bg-red-500 text-white' : 'bg-white/[0.06] text-slate-300',
              )}
            >
              {badge > 99 ? '99+' : badge}
            </span>
          )}
        </>
      )}
    </NavLink>
  );
}

export default function Layout() {
  const { alerts } = useAlertStore();
  const location = useLocation();
  const [healthData, setHealthData] = useState<BackendHealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  const openCount = alerts.filter(a => a.status === 'open').length;
  const remCount = alerts.filter(a => a.status === 'remediation').length;
  const critCount = alerts.filter(a => a.prediction === 2 && a.status === 'open').length;
  const backendOnline = healthData?.status === 'healthy';
  const investigatingCount = alerts.filter(a => a.status === 'investigating').length;
  const escalatedCount = alerts.filter(a => a.status === 'escalated').length;

  const activeLabel = useMemo(() => {
    if (location.pathname.startsWith('/triage')) return 'Triage Operations';
    if (location.pathname.startsWith('/remediation')) return 'Remediation Control';
    if (location.pathname.startsWith('/analytics')) return 'Analytics';
    return 'Dashboard';
  }, [location.pathname]);

  useEffect(() => {
    const check = async () => {
      try {
        const status = await healthStatus();
        setHealthData(status);
        setHealthError(null);
      } catch (error) {
        setHealthData(null);
        setHealthError(error instanceof Error ? error.message : 'Health check failed.');
      }
    };
    check();
    const t = setInterval(check, 30000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="flex h-screen overflow-hidden bg-surface-950 text-white">
      <aside className="relative hidden w-[296px] flex-shrink-0 border-r border-white/6 bg-surface-900/95 xl:flex xl:flex-col">
        <div className="soc-grid pointer-events-none absolute inset-0 opacity-20" />

        <div className="relative border-b border-white/6 px-6 py-6">
          <div className="flex items-center gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-cyan-400/20 bg-cyan-400/10 shadow-[0_12px_30px_rgba(6,182,212,0.16)]">
              <Terminal size={20} className="text-cyan-200" />
            </div>
            <div>
              <p className="text-[11px] font-mono uppercase tracking-[0.32em] text-cyan-400/80">SOC Intelligence</p>
              <p className="mt-1 text-lg font-display font-bold tracking-tight text-white">SENTINEL Console</p>
              <p className="text-xs font-mono text-slate-500">Operational threat triage and response</p>
            </div>
          </div>
        </div>

        <div className="relative flex-1 overflow-y-auto px-4 py-5">
          <div className="mb-6 rounded-2xl border border-red-500/12 bg-red-500/8 p-4 shadow-[0_10px_30px_rgba(127,29,29,0.15)]">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-[10px] font-mono uppercase tracking-[0.28em] text-red-300/80">Active Pressure</p>
                <p className="mt-2 text-3xl font-display font-bold text-white">{critCount}</p>
                <p className="mt-1 text-xs font-mono text-red-100/70">critical open alerts awaiting triage</p>
              </div>
              <span className="mt-1 h-2.5 w-2.5 rounded-full bg-red-400 shadow-[0_0_18px_rgba(248,113,113,0.85)] animate-pulse" />
            </div>
          </div>

          <div className="space-y-2">
            <p className="px-2 text-[10px] font-mono uppercase tracking-[0.32em] text-slate-600">Navigation</p>
            <NavItem to="/" icon={LayoutDashboard} label="Dashboard" />
            <NavItem to="/triage" icon={AlertTriangle} label="Triage" badge={openCount} />
            <NavItem to="/remediation" icon={Wrench} label="Remediation" badge={remCount} />
            <NavItem to="/analytics" icon={BarChart3} label="Analytics" />
          </div>
        </div>

        <div className="relative space-y-3 border-t border-white/6 px-4 py-4">
          <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
            <div className="mb-3 flex items-center justify-between">
              <div>
                <p className="text-[10px] font-mono uppercase tracking-[0.28em] text-slate-500">System Health</p>
                <p className="mt-1 text-sm font-display font-semibold text-slate-100">FastAPI Backend</p>
              </div>
              <div
                className={clsx(
                  'flex items-center gap-2 rounded-full border px-3 py-1 text-[10px] font-mono uppercase tracking-[0.2em]',
                  !healthData && !healthError
                    ? 'border-white/8 bg-white/[0.03] text-slate-300'
                    : backendOnline
                      ? 'border-emerald-400/15 bg-emerald-400/10 text-emerald-200'
                      : 'border-red-500/15 bg-red-500/10 text-red-200',
                )}
              >
                {!healthData && !healthError ? (
                  <span className="h-2 w-2 rounded-full bg-slate-500 animate-pulse" />
                ) : backendOnline ? (
                  <Wifi size={11} />
                ) : (
                  <WifiOff size={11} />
                )}
                {!healthData && !healthError ? 'Checking' : backendOnline ? 'Online' : 'Degraded'}
              </div>
            </div>

            {healthError ? (
              <p className="text-[11px] font-mono leading-5 text-red-300/80">{healthError}</p>
            ) : healthData ? (
              <div className="space-y-2">
                {Object.entries(healthData.models).map(([name, modelStatus]) => (
                  <div key={name} className="flex items-start justify-between gap-3 rounded-xl border border-white/6 bg-surface-950/60 px-3 py-2.5">
                    <span className="text-[10px] font-mono uppercase tracking-[0.22em] text-slate-500">{name}</span>
                    <div className="text-right">
                      <p className={clsx('text-[11px] font-mono', modelStatus.loaded ? 'text-emerald-300' : 'text-red-300')}>
                        {modelStatus.loaded ? 'ready' : 'unavailable'}
                      </p>
                      {modelStatus.error ? (
                        <p className="mt-1 max-w-[120px] break-words text-[10px] font-mono text-slate-600">{modelStatus.error}</p>
                      ) : null}
                    </div>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        </div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <div className="border-b border-white/6 bg-surface-900/80 px-6 py-4 backdrop-blur">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-[10px] font-mono uppercase tracking-[0.32em] text-slate-600">Operations View</p>
              <p className="mt-1 text-lg font-display font-semibold tracking-tight text-white">{activeLabel}</p>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <div className="rounded-2xl border border-white/8 bg-white/[0.03] px-3 py-2 text-xs font-mono text-slate-300">
                {new Date().toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })}
              </div>
              <div className="rounded-2xl border border-white/8 bg-white/[0.03] px-3 py-2 text-xs font-mono text-slate-300">
                {alerts.length} queued predictions
              </div>
              <div className="rounded-2xl border border-amber-400/15 bg-amber-400/10 px-3 py-2 text-xs font-mono text-amber-200">
                {investigatingCount} investigating
              </div>
              <div className="rounded-2xl border border-cyan-400/15 bg-cyan-400/10 px-3 py-2 text-xs font-mono text-cyan-200">
                {escalatedCount} escalated
              </div>
              <div className="rounded-2xl border border-emerald-400/15 bg-emerald-400/10 px-3 py-2 text-xs font-mono text-emerald-200">
                <ShieldCheck size={12} className="mr-2 inline-block" />
                Analyst mode
              </div>
            </div>
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-hidden">
          <Outlet />
        </div>
      </main>

      <ToastContainer />
    </div>
  );
}
