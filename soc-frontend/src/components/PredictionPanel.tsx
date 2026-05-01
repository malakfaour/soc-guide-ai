import { useRef, useState } from 'react';
import { Upload, Zap, RotateCcw, AlertTriangle, Code2, FileJson, Database } from 'lucide-react';
import clsx from 'clsx';
import { usePrediction } from '../hooks/usePrediction';
import { ProbabilityBarChart } from './Charts';
import { PredictionBadge } from './SeverityBadge';
import { Alert, PredictionLabel, PredictionModel, PREDICTION_LABELS, SeverityLevel } from '../types/alert';
import { useAlertStore } from '../lib/store';
import { FeatureSampleResponse, getSampleFeatures } from '../services/api';

const MODEL_FEATURE_COUNTS: Record<PredictionModel, number> = {
  xgboost: 44,
  lightgbm: 44,
  tabnet: 44,
};

function parseFeaturesFromText(text: string): number[] | null {
  try {
    const clean = text.trim();
    const arr = JSON.parse(clean.startsWith('[') ? clean : `[${clean}]`);
    if (Array.isArray(arr) && arr.every(value => typeof value === 'number' && Number.isFinite(value))) {
      return arr;
    }
    return null;
  } catch {
    const parts = text
      .split(/[,\s]+/)
      .map(value => parseFloat(value.trim()))
      .filter(value => !Number.isNaN(value) && Number.isFinite(value));

    return parts.length > 0 ? parts : null;
  }
}

function severityFromBackendPrediction(prediction: PredictionLabel, confidence: number): SeverityLevel {
  if (prediction === 2) return confidence >= 0.85 ? 'critical' : 'high';
  if (prediction === 1) return 'medium';
  return 'low';
}

export function PredictionPanel() {
  const [mode, setMode] = useState<'json' | 'csv'>('json');
  const [selectedModel, setSelectedModel] = useState<PredictionModel>('xgboost');
  const [inputText, setInputText] = useState('');
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [parseError, setParseError] = useState('');
  const [sampleRow, setSampleRow] = useState(0);
  const [sampleMeta, setSampleMeta] = useState<FeatureSampleResponse | null>(null);
  const [sampleLoading, setSampleLoading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const { loading, result, error, features, runPrediction, reset } = usePrediction();
  const { addAlert, addToast } = useAlertStore();
  const expectedFeatureCount = MODEL_FEATURE_COUNTS[selectedModel];

  const handleLoadProcessedSample = async () => {
    setParseError('');
    setSampleLoading(true);
    try {
      const sample = await getSampleFeatures('test', sampleRow);
      if (sample.feature_count !== expectedFeatureCount) {
        throw new Error(`Processed sample has ${sample.feature_count} features, but ${selectedModel} expects ${expectedFeatureCount}.`);
      }

      setMode('json');
      setInputText(JSON.stringify(sample.features));
      setSampleMeta(sample);
      setSampleRow(sample.row + 1);
      addToast({
        type: 'success',
        title: 'Loaded processed test row',
        message: `${sample.source} row ${sample.row}${sample.target !== null && sample.target !== undefined ? ` -> ${PREDICTION_LABELS[sample.target as PredictionLabel]}` : ''}`,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load processed sample.';
      setParseError(message);
      addToast({ type: 'error', title: 'Sample load failed', message });
    } finally {
      setSampleLoading(false);
    }
  };

  const handlePredict = async () => {
    setParseError('');
    let parsedFeatures: number[] | null = null;

    if (mode === 'json') {
      parsedFeatures = parseFeaturesFromText(inputText);
      if (!parsedFeatures) {
        setParseError('Invalid format. Expected a numeric JSON array or comma-separated numbers.');
        return;
      }
    } else if (csvFile) {
      const text = await csvFile.text();
      const lines = text.split(/\r?\n/).filter(value => value.trim());
      if (lines.length === 0) {
        setParseError('CSV is empty.');
        return;
      }

      parsedFeatures = lines
        .map(line => parseFeaturesFromText(line))
        .find(values => values?.length === expectedFeatureCount) || null;
      if (!parsedFeatures) {
        setParseError(`Could not find a numeric CSV row with ${expectedFeatureCount} features.`);
        return;
      }
    } else {
      setParseError('Please upload a CSV file or switch to JSON mode.');
      return;
    }

    if (parsedFeatures.length !== expectedFeatureCount) {
      setParseError(`Model ${selectedModel} expects ${expectedFeatureCount} numeric features, but received ${parsedFeatures.length}.`);
      return;
    }

    const response = await runPrediction(parsedFeatures, selectedModel);
    if (!response) {
      addToast({
        type: 'error',
        title: 'Prediction failed',
        message: 'The backend request did not complete successfully.',
      });
      return;
    }

    const confidence = Math.max(...response.probabilities);
    const prediction = response.prediction as PredictionLabel;
    addToast({
      type: prediction === 2 ? 'error' : prediction === 1 ? 'warning' : 'success',
      title: `${response.model.toUpperCase()} -> ${PREDICTION_LABELS[prediction]}`,
      message: `Confidence: ${(confidence * 100).toFixed(1)}%`,
    });
  };

  const handleAddToTriage = () => {
    if (!result || !features) return;

    const prediction = result.prediction as PredictionLabel;
    const confidence = Math.max(...result.probabilities);
    const detectedAt = new Date();
    const alert: Alert = {
      id: `PRED-${detectedAt.getTime()}`,
      timestamp: detectedAt,
      detectedAt,
      sourceIp: sampleMeta ? `${sampleMeta.split} row ${sampleMeta.row}` : 'manual vector',
      destIp: result.model.toUpperCase(),
      port: features.length,
      protocol: result.model.toUpperCase(),
      severity: severityFromBackendPrediction(prediction, confidence),
      prediction,
      probabilities: result.probabilities,
      confidence,
      status: 'open',
      features,
      notes: sampleMeta
        ? `Processed sample: ${sampleMeta.source} row ${sampleMeta.row}${sampleMeta.target !== null && sampleMeta.target !== undefined ? `, target ${PREDICTION_LABELS[sampleMeta.target as PredictionLabel]}` : ''}`
        : 'Manual preprocessed feature vector',
    };

    addAlert(alert);
    addToast({ type: 'info', title: 'Backend prediction added to triage queue', message: alert.id });
  };

  const prediction = result?.prediction as PredictionLabel | undefined;
  const confidence = result ? Math.max(...result.probabilities) : 0;

  const resultBg: Record<PredictionLabel, string> = {
    0: 'border-emerald-400/18 bg-emerald-400/[0.05]',
    1: 'border-yellow-400/18 bg-yellow-400/[0.05]',
    2: 'border-red-500/18 bg-red-500/[0.05]',
  };

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-[1.1fr_0.9fr]">
        <div className="space-y-5">
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <div className="space-y-2">
              <label className="text-[11px] font-mono uppercase tracking-[0.22em] text-slate-500">Inference Model</label>
              <select
                value={selectedModel}
                onChange={event => setSelectedModel(event.target.value as PredictionModel)}
                className="soc-input"
              >
                <option value="xgboost">XGBoost</option>
                <option value="lightgbm">LightGBM</option>
                <option value="tabnet">TabNet</option>
              </select>
            </div>

            <div className="space-y-2">
              <label className="text-[11px] font-mono uppercase tracking-[0.22em] text-slate-500">Input Mode</label>
              <div className="flex gap-1 rounded-2xl border border-white/8 bg-surface-900 p-1">
                {(['json', 'csv'] as const).map(currentMode => (
                  <button
                    key={currentMode}
                    onClick={() => {
                      setMode(currentMode);
                      reset();
                      setSampleMeta(null);
                      setParseError('');
                    }}
                    className={clsx(
                      'flex flex-1 items-center justify-center gap-2 rounded-xl px-3 py-2 text-xs font-mono transition-all duration-200',
                      mode === currentMode ? 'bg-cyan-500 text-slate-950 shadow-[0_10px_24px_rgba(6,182,212,0.18)]' : 'text-slate-400 hover:text-slate-100',
                    )}
                  >
                    {currentMode === 'json' ? <Code2 size={12} /> : <FileJson size={12} />}
                    {currentMode.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {mode === 'json' ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <label className="text-[11px] font-mono uppercase tracking-[0.22em] text-slate-500">
                    Feature Vector
                  </label>
                  <p className="mt-1 text-[11px] font-mono text-slate-600">
                    Expected length: {expectedFeatureCount} preprocessed numeric features
                  </p>
                </div>
                <button
                  onClick={handleLoadProcessedSample}
                  disabled={sampleLoading}
                  className="soc-button-secondary flex items-center gap-2 px-3 py-2 text-xs font-mono disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {sampleLoading ? (
                    <span className="h-3.5 w-3.5 rounded-full border border-cyan-200 border-t-transparent animate-spin" />
                  ) : (
                    <Database size={12} />
                  )}
                  Load processed test row
                </button>
              </div>

              <textarea
                value={inputText}
                onChange={event => {
                  setInputText(event.target.value);
                  setSampleMeta(null);
                }}
                placeholder={`Paste ${expectedFeatureCount} preprocessed numeric features from your backend/data pipeline`}
                rows={7}
                className="soc-input resize-none font-mono text-xs leading-6 text-emerald-200"
              />

              {sampleMeta ? (
                <div className="rounded-2xl border border-cyan-400/12 bg-cyan-400/[0.05] px-3 py-2 text-[11px] font-mono text-cyan-100/90">
                  {sampleMeta.source} row {sampleMeta.row}
                  {sampleMeta.target !== null && sampleMeta.target !== undefined
                    ? ` | target ${PREDICTION_LABELS[sampleMeta.target as PredictionLabel]}`
                    : ''}
                </div>
              ) : null}
            </div>
          ) : (
            <div className="space-y-3">
              <div>
                <label className="text-[11px] font-mono uppercase tracking-[0.22em] text-slate-500">Upload Preprocessed CSV</label>
                <p className="mt-1 text-[11px] font-mono text-slate-600">
                  The first valid row with {expectedFeatureCount} numeric features will be used for inference.
                </p>
              </div>
              <div
                onClick={() => fileRef.current?.click()}
                className={clsx(
                  'rounded-2xl border-2 border-dashed px-6 py-10 text-center transition-all duration-200',
                  csvFile
                    ? 'border-cyan-400/30 bg-cyan-400/[0.05]'
                    : 'border-white/8 bg-white/[0.02] hover:border-cyan-400/18 hover:bg-cyan-400/[0.03]',
                )}
              >
                <Upload size={20} className={clsx('mx-auto mb-3', csvFile ? 'text-cyan-200' : 'text-slate-600')} />
                <p className="text-sm font-display font-semibold text-slate-100">
                  {csvFile ? csvFile.name : 'Click to upload a preprocessed CSV file'}
                </p>
                <p className="mt-2 text-[11px] font-mono text-slate-500">Feature count must match the selected model.</p>
                <input
                  ref={fileRef}
                  type="file"
                  accept=".csv"
                  className="hidden"
                  onChange={event => {
                    setCsvFile(event.target.files?.[0] || null);
                    setSampleMeta(null);
                    setParseError('');
                  }}
                />
              </div>
            </div>
          )}

          {parseError && (
            <div className="flex items-center gap-2 rounded-2xl border border-red-500/18 bg-red-500/[0.05] px-3 py-2.5 text-xs font-mono text-red-200">
              <AlertTriangle size={12} /> {parseError}
            </div>
          )}
          {error && (
            <div className="flex items-center gap-2 rounded-2xl border border-red-500/18 bg-red-500/[0.05] px-3 py-2.5 text-xs font-mono text-red-200">
              <AlertTriangle size={12} /> Backend error: {error}
            </div>
          )}

          <div className="flex gap-3">
            <button
              onClick={handlePredict}
              disabled={loading}
              className={clsx(
                'soc-button-primary flex flex-1 items-center justify-center gap-2 py-3 font-mono text-sm',
                loading && 'cursor-not-allowed border-cyan-400/12 bg-cyan-500/40 text-cyan-100',
              )}
            >
              {loading ? (
                <>
                  <span className="h-4 w-4 rounded-full border-2 border-slate-900/70 border-t-transparent animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Zap size={14} />
                  Run Prediction
                </>
              )}
            </button>
            {result ? (
              <button
                onClick={reset}
                className="soc-button-secondary flex items-center justify-center px-4"
              >
                <RotateCcw size={14} />
              </button>
            ) : null}
          </div>
        </div>

        <div className="rounded-2xl border border-white/8 bg-surface-900/40 p-5">
          {!result || prediction === undefined ? (
            <div className="flex h-full min-h-[260px] flex-col items-center justify-center text-center">
              <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-white/8 bg-white/[0.03]">
                <Zap size={22} className="text-slate-500" />
              </div>
              <p className="text-base font-display font-semibold text-slate-100">Awaiting model output</p>
              <p className="mt-2 max-w-sm text-[11px] font-mono leading-6 text-slate-500">
                Submit a feature vector to render prediction confidence and class distribution in this analyst side panel.
              </p>
            </div>
          ) : (
            <div className={clsx('space-y-5 rounded-2xl border p-5 animate-fade-in', resultBg[prediction])}>
              <div className="flex items-start justify-between gap-4">
                <div className="space-y-2">
                  <p className="text-[11px] font-mono uppercase tracking-[0.22em] text-slate-500">Model Output</p>
                  <PredictionBadge prediction={prediction} size="md" />
                  <p className="text-[11px] font-mono uppercase tracking-[0.2em] text-slate-600">
                    Engine: {result.model.toUpperCase()}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-[11px] font-mono text-slate-500">Confidence</p>
                  <p
                    className={clsx(
                      'mt-2 text-4xl font-display font-bold tracking-tight',
                      prediction === 2 ? 'text-red-200' : prediction === 1 ? 'text-yellow-200' : 'text-emerald-200',
                    )}
                  >
                    {(confidence * 100).toFixed(1)}%
                  </p>
                </div>
              </div>

              <div className="rounded-2xl border border-white/6 bg-surface-950/45 p-4">
                <p className="mb-3 text-[11px] font-mono uppercase tracking-[0.22em] text-slate-500">Probability Distribution</p>
                <ProbabilityBarChart probabilities={result.probabilities} />
              </div>

              <div className="flex items-center justify-between gap-3 border-t border-white/8 pt-4">
                <span className="text-[11px] font-mono text-slate-500">{features?.length} features analyzed</span>
                <button
                  onClick={handleAddToTriage}
                  className="soc-button-secondary px-3 py-2 text-xs font-mono"
                >
                  Add to Triage Queue
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
