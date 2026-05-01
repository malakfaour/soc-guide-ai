import { useAlertStore } from '../lib/store';
import { CheckCircle, XCircle, AlertTriangle, Info, X } from 'lucide-react';
import clsx from 'clsx';

export function ToastContainer() {
  const { toasts, removeToast } = useAlertStore();

  if (!toasts.length) return null;

  const configs = {
    success: { icon: CheckCircle, classes: 'border-emerald-400/20 bg-surface-900/95 text-emerald-100', iconClass: 'text-emerald-300' },
    error: { icon: XCircle, classes: 'border-red-500/20 bg-surface-900/95 text-red-100', iconClass: 'text-red-300' },
    warning: { icon: AlertTriangle, classes: 'border-yellow-400/20 bg-surface-900/95 text-yellow-100', iconClass: 'text-yellow-300' },
    info: { icon: Info, classes: 'border-cyan-400/20 bg-surface-900/95 text-cyan-100', iconClass: 'text-cyan-300' },
  };

  return (
    <div className="fixed bottom-5 right-5 z-[100] flex max-w-sm flex-col gap-3">
      {toasts.map(toast => {
        const cfg = configs[toast.type];
        const Icon = cfg.icon;
        return (
          <div key={toast.id}
            className={clsx(
              'flex items-start gap-3 rounded-2xl border px-4 py-3 shadow-2xl backdrop-blur animate-fade-in',
              cfg.classes
            )}>
            <div className="mt-0.5 rounded-xl border border-white/8 bg-white/[0.03] p-2">
              <Icon size={16} className={clsx('flex-shrink-0', cfg.iconClass)} />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-display font-semibold">{toast.title}</p>
              {toast.message && <p className="mt-1 text-[11px] font-mono leading-5 opacity-70">{toast.message}</p>}
            </div>
            <button onClick={() => removeToast(toast.id)}
              className="flex-shrink-0 text-current opacity-50 transition-opacity hover:opacity-100">
              <X size={12} />
            </button>
          </div>
        );
      })}
    </div>
  );
}
