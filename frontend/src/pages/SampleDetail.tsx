import { useQuery } from '@tanstack/react-query'
import { useParams, Link } from 'react-router-dom'
import { ArrowLeft, ExternalLink } from 'lucide-react'
import { api } from '../lib/api'
import ClassBadge from '../components/ClassBadge'
import PredictionComparison from '../components/PredictionComparison'
import SequenceViewer from '../components/SequenceViewer'

export default function SampleDetail() {
  const { id } = useParams<{ id: string }>()
  const sampleId = parseInt(id || '0')

  const { data: sample, isLoading: sampleLoading } = useQuery({
    queryKey: ['sample', sampleId],
    queryFn: () => api.getSample(sampleId),
    enabled: !!sampleId,
  })

  const { data: predictions, isLoading: predictionsLoading } = useQuery({
    queryKey: ['predictions', sampleId],
    queryFn: () => api.getPredictions(sampleId),
    enabled: !!sampleId,
  })

  if (sampleLoading) {
    return <div className="animate-pulse">Loading sample...</div>
  }

  if (!sample) {
    return (
      <div className="text-center py-12">
        <p className="text-slate-500">Sample not found</p>
        <Link to="/samples" className="text-green-600 hover:underline mt-2 inline-block">
          Back to samples
        </Link>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <Link
          to="/samples"
          className="flex items-center gap-1 text-slate-500 hover:text-slate-700 mb-2"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to samples
        </Link>
        <h1 className="text-3xl font-bold text-slate-900">{sample.name}</h1>
        {sample.description && (
          <p className="text-slate-600 mt-1">{sample.description}</p>
        )}
      </div>

      {/* Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white rounded-xl shadow-sm p-4">
          <p className="text-sm text-slate-500">NCBI ID</p>
          <p className="font-mono font-medium mt-1">
            {sample.ncbi_id ? (
              <a
                href={`https://www.ncbi.nlm.nih.gov/nuccore/${sample.ncbi_id}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-green-600 hover:underline flex items-center gap-1"
              >
                {sample.ncbi_id}
                <ExternalLink className="w-3 h-3" />
              </a>
            ) : (
              '-'
            )}
          </p>
        </div>
        
        <div className="bg-white rounded-xl shadow-sm p-4">
          <p className="text-sm text-slate-500">True Class</p>
          <div className="mt-1">
            {sample.true_label ? (
              <ClassBadge label={sample.true_label} />
            ) : (
              <span className="text-slate-400">Unknown</span>
            )}
          </div>
        </div>
        
        <div className="bg-white rounded-xl shadow-sm p-4">
          <p className="text-sm text-slate-500">Length</p>
          <p className="font-medium mt-1">{sample.length.toLocaleString()} bp</p>
        </div>
        
        <div className="bg-white rounded-xl shadow-sm p-4">
          <p className="text-sm text-slate-500">GC Content</p>
          <p className="font-medium mt-1">{(sample.gc_content * 100).toFixed(1)}%</p>
        </div>
      </div>

      {/* Predictions */}
      <div className="bg-white rounded-xl shadow-sm p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">
          Classification Results
        </h2>
        {predictionsLoading ? (
          <div className="animate-pulse">Loading predictions...</div>
        ) : predictions ? (
          <PredictionComparison predictions={predictions} />
        ) : (
          <p className="text-slate-500">No predictions available</p>
        )}
      </div>

      {/* Sequence Viewer */}
      <div className="bg-white rounded-xl shadow-sm p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">
          Sequence
        </h2>
        <SequenceViewer 
          sampleId={sampleId}
          sequence={sample.sequence || ''}
          totalLength={sample.length}
        />
      </div>
    </div>
  )
}
