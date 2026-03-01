import ClassBadge from './ClassBadge'

interface Motif {
  pattern: string
  consensus: string
  occurrences: number
  avg_importance: number
  positions: number[]
  class_association: string
}

interface MotifViewerProps {
  motifs: Motif[]
}

export default function MotifViewer({ motifs }: MotifViewerProps) {
  if (!motifs || motifs.length === 0) {
    return (
      <p className="text-slate-500">No motifs discovered for this sample</p>
    )
  }

  return (
    <div className="space-y-4">
      {motifs.map((motif, index) => (
        <div
          key={index}
          className="p-4 bg-slate-50 rounded-lg flex items-start justify-between"
        >
          <div>
            <div className="flex items-center gap-3">
              <span className="font-mono text-lg font-bold text-slate-900 tracking-wider">
                {motif.pattern}
              </span>
              <ClassBadge label={motif.class_association} size="sm" />
            </div>
            
            <div className="mt-2 flex items-center gap-4 text-sm text-slate-600">
              <span>
                <strong>{motif.occurrences}</strong> occurrences
              </span>
              <span>
                Avg. importance: <strong>{motif.avg_importance.toFixed(3)}</strong>
              </span>
            </div>

            <div className="mt-2 text-xs text-slate-500">
              Positions: {motif.positions.slice(0, 5).map(p => p.toLocaleString()).join(', ')}
              {motif.positions.length > 5 && ` +${motif.positions.length - 5} more`}
            </div>
          </div>

          <div className="flex flex-col items-end">
            <div className="w-24 h-2 bg-slate-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-green-500"
                style={{ width: `${motif.avg_importance * 100}%` }}
              />
            </div>
            <span className="text-xs text-slate-500 mt-1">
              {(motif.avg_importance * 100).toFixed(0)}% importance
            </span>
          </div>
        </div>
      ))}

      <div className="text-sm text-slate-500 mt-4">
        <p>
          <strong>Motifs</strong> are recurring sequence patterns that the model finds important 
          for classification. Higher importance scores indicate patterns that strongly influence 
          the prediction.
        </p>
      </div>
    </div>
  )
}
