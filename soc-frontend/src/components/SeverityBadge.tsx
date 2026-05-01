import { SeverityLevel, PredictionLabel } from '../types/alert';
import { AlertTriangle, CheckCircle, Info, ShieldAlert, XCircle } from 'lucide-react';
import clsx from 'clsx';
import type { ElementType } from 'react';

interface SeverityBadgeProps {
  severity?: SeverityLevel;
  className?: string;
}

interface PredictionBadgeProps {
  prediction: PredictionLabel;
  className?: string;
  size?: 'sm' | 'md';
}

export function SeverityBadge({ severity, className }: SeverityBadgeProps) {
  const configs: Record<SeverityLevel, { label: string; classes: string; icon: ElementType }> = {
    critical: { label: 'CRITICAL', classes: 'border-red-500/20 bg-red-500/12 text-red-200 shadow-[0_0_24px_rgba(239,68,68,0.14)]', icon: ShieldAlert },
    high: { label: 'HIGH', classes: 'border-orange-400/20 bg-orange-400/12 text-orange-200', icon: AlertTriangle },
    medium: { label: 'MEDIUM', classes: 'border-yellow-400/20 bg-yellow-400/12 text-yellow-200', icon: Info },
    low: { label: 'LOW', classes: 'border-emerald-400/20 bg-emerald-400/12 text-emerald-200', icon: CheckCircle },
  };
  if (!severity) return null;
  const { label, classes, icon: Icon } = configs[severity];
  return (
    <span className={clsx(
      'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-mono font-semibold tracking-[0.22em]',
      classes, className
    )}>
      <Icon size={10} />
      {label}
    </span>
  );
}

export function PredictionBadge({ prediction, className, size = 'md' }: PredictionBadgeProps) {
  const configs: Record<PredictionLabel, { label: string; classes: string }> = {
    0: { label: 'FP', classes: 'border-emerald-400/20 bg-emerald-400/12 text-emerald-200' },
    1: { label: 'BP', classes: 'border-yellow-400/20 bg-yellow-400/12 text-yellow-200' },
    2: { label: 'TP', classes: 'border-red-500/20 bg-red-500/12 text-red-200 shadow-[0_0_24px_rgba(239,68,68,0.12)]' },
  };
  const fullLabels: Record<PredictionLabel, string> = {
    0: 'False Positive', 1: 'Benign Positive', 2: 'True Positive'
  };
  const { label, classes } = configs[prediction];
  return (
    <span className={clsx(
      'inline-flex items-center rounded-full border font-mono font-bold uppercase tracking-[0.18em]',
      size === 'sm' ? 'px-2 py-1 text-[10px]' : 'px-3 py-1.5 text-xs',
      classes, className
    )} title={fullLabels[prediction]}>
      {size === 'sm' ? label : fullLabels[prediction]}
    </span>
  );
}

interface StatusBadgeProps {
  status: string;
  className?: string;
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  const configs: Record<string, { classes: string; dot: string }> = {
    open: { classes: 'text-cyan-200 border-cyan-400/20 bg-cyan-400/10', dot: 'bg-cyan-300' },
    investigating: { classes: 'text-amber-200 border-amber-400/20 bg-amber-400/10', dot: 'bg-amber-300 animate-pulse' },
    resolved: { classes: 'text-emerald-200 border-emerald-400/20 bg-emerald-400/10', dot: 'bg-emerald-300' },
    escalated: { classes: 'text-orange-200 border-orange-400/20 bg-orange-400/10', dot: 'bg-orange-300 animate-pulse' },
    remediation: { classes: 'text-violet-200 border-violet-400/20 bg-violet-400/10', dot: 'bg-violet-300 animate-pulse' },
  };
  const cfg = configs[status] || configs.open;
  return (
    <span className={clsx(
      'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-mono uppercase tracking-[0.18em]',
      cfg.classes, className
    )}>
      <span className={clsx('w-1.5 h-1.5 rounded-full', cfg.dot)} />
      {status}
    </span>
  );
}
