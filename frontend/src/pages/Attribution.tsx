import { useQuery } from '@tanstack/react-query'
import { useParams, Link } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import { useState } from 'react'
import { api } from '../lib/api'
import AttributionHeatmap from '../components/AttributionHeatmap'
import MotifViewer from '../components/MotifViewer'

export default function Attribution() {
  const { id } = useParams<{ id: string }>()
  const sampleId = parseInt(id || '0')
  
  const [method, setMethod] = useState<'contrastive' | 'reinforcement'>('contrastive')
  const [regionStart, setRegionStart] = useState(0)
  const [regionEnd, setRegionEnd] = useState(500)

  const { data: sample } = useQuery({
    queryKey: ['sample', sampleId],
    queryFn: () => api.getSample(sampleId),
    enabled: !!sampleId,
  })

  const { data: saliency, isLoading: saliencyLoading } = useQuery({
    queryKey: ['saliency', sampleId, method, regionStart, regionEnd],
    queryFn: () => api.getSaliencyMap(sampleId, method, regionStart, regionEnd),
    enabled: !!sampleId,
  })

  const { data: motifs } = useQuery({
    queryKey: ['motifs', sampleId, method],
    queryFn: () => api.getMotifs(sampleId, method),
    enabled: !!sampleId,
  })

  const { data: regions } = useQuery({
    queryKey: ['regions', sampleId, method],
    queryFn: () => api.getImportantRegions(sampleId, method),
    enabled: !!sampleId,
  })

  const handleRegionClick = (start: number, end: number) => {
    setRegionStart(start)
    setRegionEnd(end)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <Link
          to={sample ? `/samples/${sampleId}` : '/samples'}
          className="flex items-center gap-1 text-slate-500 hover:text-slate-700 mb-2"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to {sample?.name || 'samples'}
        </Link>
        <h1 className="text-3xl font-bold text-slate-900">
          Attribution Analysis
        </h1>
        <p className="text-slate-600 mt-1">
          Visualize which sequence regions influence the model's prediction
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-xl shadow-sm p-4 flex items-center gap-6">
        <div>
          <label className="text-sm text-slate-500 block mb-1">Method</label>
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value as typeof method)}
            className="border border-slate-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500"
          >
            <option value="contrastive">Contrastive Learning</option>
            <option value="reinforcement">Deep RL</option>
          </select>
        </div>
        
        <div>
          <label className="text-sm text-slate-500 block mb-1">Region Start</label>
          <input
            type="number"
            value={regionStart}
            onChange={(e) => setRegionStart(parseInt(e.target.value) || 0)}
            min={0}
            max={sample?.length || 10000}
            className="w-24 border border-slate-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500"
          />
        </div>
        
        <div>
          <label className="text-sm text-slate-500 block mb-1">Region End</label>
          <input
            type="number"
            value={regionEnd}
            onChange={(e) => setRegionEnd(parseInt(e.target.value) || 500)}
            min={regionStart + 10}
            max={sample?.length || 10000}
            className="w-24 border border-slate-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500"
          />
        </div>

        <div className="ml-auto text-sm text-slate-500">
          Showing {regionEnd - regionStart} bp of {sample?.length?.toLocaleString() || '?'} bp total
        </div>
      </div>

      {/* Attribution Heatmap */}
      <div className="bg-white rounded-xl shadow-sm p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">
          Sequence Attribution Heatmap
        </h2>
        {saliencyLoading ? (
          <div className="animate-pulse h-32 bg-slate-100 rounded" />
        ) : saliency ? (
          <AttributionHeatmap
            sequence={saliency.sequence}
            scores={saliency.scores}
            start={saliency.start}
          />
        ) : (
          <p className="text-slate-500">Failed to load attribution data</p>
        )}
      </div>

      {/* Important Regions */}
      <div className="bg-white rounded-xl shadow-sm p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">
          Top Important Regions
        </h2>
        {regions ? (
          <div className="space-y-2">
            {regions.slice(0, 10).map((region, i) => (
              <button
                key={i}
                onClick={() => handleRegionClick(region.start, region.end)}
                className="w-full text-left p-3 bg-slate-50 rounded-lg hover:bg-slate-100 transition-colors flex items-center justify-between"
              >
                <div>
                  <span className="font-mono text-sm">
                    {region.start.toLocaleString()} - {region.end.toLocaleString()}
                  </span>
                  <span className="text-slate-500 text-sm ml-2">
                    ({region.end - region.start} bp)
                  </span>
                </div>
                <div className="flex items-center gap-4">
                  <div className="text-right">
                    <span className="text-sm text-slate-500">Mean Score</span>
                    <p className={`font-medium ${
                      region.mean_score > 0.5 ? 'text-green-600' :
                      region.mean_score > 0.2 ? 'text-amber-600' :
                      'text-slate-600'
                    }`}>
                      {region.mean_score.toFixed(3)}
                    </p>
                  </div>
                  <div
                    className="w-16 h-4 rounded overflow-hidden bg-slate-200"
                    title={`Score: ${region.mean_score.toFixed(3)}`}
                  >
                    <div
                      className="h-full bg-green-500"
                      style={{ width: `${Math.max(0, region.mean_score * 100)}%` }}
                    />
                  </div>
                </div>
              </button>
            ))}
          </div>
        ) : (
          <div className="animate-pulse h-32 bg-slate-100 rounded" />
        )}
      </div>

      {/* Motifs */}
      <div className="bg-white rounded-xl shadow-sm p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">
          Discovered Motifs
        </h2>
        {motifs ? (
          <MotifViewer motifs={motifs.motifs} />
        ) : (
          <div className="animate-pulse h-32 bg-slate-100 rounded" />
        )}
      </div>
    </div>
  )
}
