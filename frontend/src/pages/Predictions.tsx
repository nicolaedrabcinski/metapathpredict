import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { api } from '../lib/api'
import ClassBadge from '../components/ClassBadge'

export default function Predictions() {
  const { data: disagreements, isLoading } = useQuery({
    queryKey: ['disagreements'],
    queryFn: () => api.getDisagreements(0.3, 50),
  })

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Predictions</h1>
        <p className="text-slate-600 mt-1">
          Compare predictions across different classification methods
        </p>
      </div>

      {/* Method Disagreements */}
      <div className="bg-white rounded-xl shadow-sm p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">
          Method Disagreements
        </h2>
        <p className="text-slate-600 mb-4">
          Samples where Contrastive Learning and DRL methods disagree on classification
        </p>

        {isLoading ? (
          <div className="animate-pulse">Loading...</div>
        ) : disagreements?.length === 0 ? (
          <p className="text-slate-500">No significant disagreements found</p>
        ) : (
          <div className="space-y-4">
            {disagreements?.map((item) => (
              <div
                key={item.sample_id}
                className="border border-slate-200 rounded-lg p-4 hover:border-slate-300 transition-colors"
              >
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <Link
                      to={`/samples/${item.sample_id}`}
                      className="font-medium text-slate-900 hover:text-green-600"
                    >
                      {item.sample_name}
                    </Link>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-sm text-slate-500">True:</span>
                      {item.true_label ? (
                        <ClassBadge label={item.true_label} size="sm" />
                      ) : (
                        <span className="text-sm text-slate-400">Unknown</span>
                      )}
                    </div>
                  </div>
                  <div className="text-right">
                    <span className="text-sm text-slate-500">Agreement</span>
                    <p className={`font-medium ${
                      item.agreement < 0.5 ? 'text-red-600' : 
                      item.agreement < 0.8 ? 'text-amber-600' : 
                      'text-green-600'
                    }`}>
                      {(item.agreement * 100).toFixed(0)}%
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  {item.predictions.map((pred) => (
                    <div
                      key={pred.method}
                      className={`p-3 rounded-lg ${
                        pred.predicted_class === item.true_label
                          ? 'bg-green-50 border border-green-200'
                          : 'bg-slate-50'
                      }`}
                    >
                      <p className="text-sm font-medium text-slate-700 uppercase">
                        {pred.method}
                      </p>
                      <div className="flex items-center gap-2 mt-1">
                        <ClassBadge label={pred.predicted_class} size="sm" />
                        <span className="text-sm text-slate-500">
                          {(pred.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="flex justify-end mt-3">
                  <Link
                    to={`/attribution/${item.sample_id}`}
                    className="text-sm text-indigo-600 hover:text-indigo-700 font-medium"
                  >
                    View Attribution →
                  </Link>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
