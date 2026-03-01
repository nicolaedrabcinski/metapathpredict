import ClassBadge from './ClassBadge'

interface Prediction {
  method: string
  predicted_class: string
  confidence: number
  probabilities: {
    bacteria: number
    eukaryotic: number
    virus: number
  }
}

interface PredictionComparisonProps {
  predictions: {
    sample_id: number
    sample_name: string
    true_label?: string
    predictions: Prediction[]
    consensus?: string
    agreement: number
  }
}

export default function PredictionComparison({ predictions }: PredictionComparisonProps) {
  return (
    <div className="space-y-6">
      {/* Consensus */}
      <div className="flex items-center justify-between p-4 bg-slate-50 rounded-lg">
        <div>
          <span className="text-sm text-slate-500">Consensus Prediction</span>
          <div className="flex items-center gap-2 mt-1">
            {predictions.consensus && (
              <ClassBadge label={predictions.consensus} />
            )}
            <span className={`text-sm font-medium ${
              predictions.agreement >= 1 ? 'text-green-600' :
              predictions.agreement >= 0.66 ? 'text-amber-600' :
              'text-red-600'
            }`}>
              ({(predictions.agreement * 100).toFixed(0)}% agreement)
            </span>
          </div>
        </div>
        {predictions.true_label && (
          <div className="text-right">
            <span className="text-sm text-slate-500">True Label</span>
            <div className="mt-1">
              <ClassBadge label={predictions.true_label} />
            </div>
          </div>
        )}
      </div>

      {/* Method cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {predictions.predictions.map((pred) => (
          <div
            key={pred.method}
            className={`p-4 rounded-lg border-2 ${
              pred.predicted_class === predictions.true_label
                ? 'border-green-300 bg-green-50'
                : 'border-slate-200 bg-white'
            }`}
          >
            <div className="flex items-center justify-between mb-3">
              <span className="font-medium text-slate-900 uppercase">
                {pred.method}
              </span>
              <span className="text-sm text-slate-500">
                {(pred.confidence * 100).toFixed(1)}% conf.
              </span>
            </div>

            <div className="mb-3">
              <ClassBadge label={pred.predicted_class} />
            </div>

            {/* Probability bars */}
            <div className="space-y-2">
              <ProbabilityBar label="Bacteria" value={pred.probabilities.bacteria} color="bg-green-500" />
              <ProbabilityBar label="Virus" value={pred.probabilities.virus} color="bg-red-500" />
              <ProbabilityBar label="Eukaryotic" value={pred.probabilities.eukaryotic} color="bg-blue-500" />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function ProbabilityBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div>
      <div className="flex justify-between text-xs text-slate-500 mb-1">
        <span>{label}</span>
        <span>{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
        <div
          className={`h-full ${color} transition-all duration-300`}
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </div>
  )
}
