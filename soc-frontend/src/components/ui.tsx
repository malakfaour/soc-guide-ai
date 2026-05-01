import clsx from 'clsx';
import { LucideIcon } from 'lucide-react';
import React from 'react';

export function PageShell({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={clsx('h-full overflow-y-auto px-6 py-6 sm:px-8 lg:px-10', className)}>{children}</div>;
}

export function PageHeader({
  eyebrow,
  title,
  subtitle,
  actions,
}: {
  eyebrow?: string;
  title: string;
  subtitle: string;
  actions?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
      <div className="space-y-2">
        {eyebrow && (
          <p className="text-[10px] font-mono uppercase tracking-[0.32em] text-cyan-400/80">{eyebrow}</p>
        )}
        <div className="space-y-1.5">
          <h1 className="text-2xl font-display font-bold tracking-tight text-white">{title}</h1>
          <p className="max-w-3xl text-sm font-mono leading-6 text-slate-400">{subtitle}</p>
        </div>
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-3">{actions}</div> : null}
    </div>
  );
}

export function Panel({
  children,
  className,
  padded = true,
}: {
  children: React.ReactNode;
  className?: string;
  padded?: boolean;
}) {
  return (
    <section
      className={clsx(
        'soc-panel rounded-2xl border border-white/6',
        padded && 'p-5 sm:p-6',
        className,
      )}
    >
      {children}
    </section>
  );
}

export function PanelHeader({
  icon: Icon,
  title,
  subtitle,
  action,
}: {
  icon?: LucideIcon;
  title: string;
  subtitle?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="mb-5 flex flex-col gap-3 border-b border-white/6 pb-4 sm:flex-row sm:items-start sm:justify-between">
      <div className="flex items-start gap-3">
        {Icon ? (
          <div className="mt-0.5 flex h-10 w-10 items-center justify-center rounded-xl border border-cyan-400/15 bg-cyan-400/10">
            <Icon size={16} className="text-cyan-300" />
          </div>
        ) : null}
        <div className="space-y-1">
          <h2 className="text-sm font-display font-semibold uppercase tracking-[0.18em] text-slate-100">{title}</h2>
          {subtitle ? <p className="text-xs font-mono text-slate-500">{subtitle}</p> : null}
        </div>
      </div>
      {action}
    </div>
  );
}

export function InfoPill({
  icon: Icon,
  label,
  tone = 'neutral',
}: {
  icon?: LucideIcon;
  label: string;
  tone?: 'neutral' | 'info' | 'danger' | 'warning' | 'success';
}) {
  const tones = {
    neutral: 'border-white/8 bg-white/[0.03] text-slate-300',
    info: 'border-cyan-400/20 bg-cyan-400/10 text-cyan-200',
    danger: 'border-red-500/20 bg-red-500/10 text-red-200',
    warning: 'border-amber-400/20 bg-amber-400/10 text-amber-200',
    success: 'border-emerald-400/20 bg-emerald-400/10 text-emerald-200',
  };

  return (
    <div className={clsx('inline-flex items-center gap-2 rounded-xl border px-3 py-2 text-xs font-mono', tones[tone])}>
      {Icon ? <Icon size={12} /> : null}
      <span>{label}</span>
    </div>
  );
}

export function EmptyState({
  icon: Icon,
  title,
  subtitle,
}: {
  icon: LucideIcon;
  title: string;
  subtitle: string;
}) {
  return (
    <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-white/8 bg-white/[0.02] px-6 py-16 text-center">
      <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-white/8 bg-surface-900">
        <Icon size={22} className="text-slate-500" />
      </div>
      <p className="text-base font-display font-semibold text-slate-100">{title}</p>
      <p className="mt-2 max-w-md text-sm font-mono leading-6 text-slate-500">{subtitle}</p>
    </div>
  );
}
