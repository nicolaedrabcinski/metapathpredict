import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { 
  Database, 
  Activity, 
  AlertTriangle,
  TrendingUp
} from 'lucide-react'
import { api } from '../lib/api'

export default function Dashboard() {
  const { data: stats, isLoading } = useQuery({
    queryKey: ['stats'],
    queryFn: () => api.getStats(),
  })

  if (isLoading) {
    return <div className="animate-pulse">Loading...</div>
  }

  const cards = [
    {
      title: 'Total Samples',
      value: stats?.total_samples || 0,
      icon: Database,
      color: 'bg-blue-500',
      link: '/samples',
    },
    {
      title: 'Bacteria',
      value: stats?.by_class?.bacteria || 0,
      icon: Activity,
      color: 'bg-green-500',
      link: '/samples?class=bacteria',
    },
    {
      title: 'Viruses',
      value: stats?.by_class?.virus || 0,
      icon: AlertTriangle,
      color: 'bg-red-500',
      link: '/samples?class=virus',
    },
    {
      title: 'Eukaryotic',
      value: stats?.by_class?.eukaryotic || 0,
      icon: TrendingUp,
      color: 'bg-indigo-500',
      link: '/samples?class=eukaryotic',
    },
  ]

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Dashboard</h1>
        <p className="text-slate-600 mt-1">
          ML Interpretability Dashboard for Metagenomic Sequence Classification
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {cards.map((card) => (
          <Link
            key={card.title}
            to={card.link}
            className="bg-white rounded-xl shadow-sm p-6 hover:shadow-md transition-shadow"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-500 text-sm">{card.title}</p>
                <p className="text-3xl font-bold text-slate-900 mt-1">
                  {card.value}
                </p>
              </div>
              <div className={`${card.color} p-3 rounded-lg`}>
                <card.icon className="w-6 h-6 text-white" />
              </div>
            </div>
          </Link>
        ))}
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Samples */}
        <div className="bg-white rounded-xl shadow-sm p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-4">
            Quick Actions
          </h2>
          <div className="space-y-3">
            <Link
              to="/samples"
              className="flex items-center justify-between p-3 bg-slate-50 rounded-lg hover:bg-slate-100 transition-colors"
            >
              <span className="text-slate-700">Browse Samples</span>
              <span className="text-slate-400">→</span>
            </Link>
            <Link
              to="/predictions"
              className="flex items-center justify-between p-3 bg-slate-50 rounded-lg hover:bg-slate-100 transition-colors"
            >
              <span className="text-slate-700">View Predictions</span>
              <span className="text-slate-400">→</span>
            </Link>
            <Link
              to="/predictions?disagreements=true"
              className="flex items-center justify-between p-3 bg-slate-50 rounded-lg hover:bg-slate-100 transition-colors"
            >
              <span className="text-slate-700">Method Disagreements</span>
              <span className="text-slate-400">→</span>
            </Link>
          </div>
        </div>

        {/* Dataset Info */}
        <div className="bg-white rounded-xl shadow-sm p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-4">
            Dataset Statistics
          </h2>
          <div className="space-y-4">
            <div className="flex justify-between">
              <span className="text-slate-500">Avg. Sequence Length</span>
              <span className="font-medium">
                {stats?.avg_length?.toFixed(0) || 0} bp
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Avg. GC Content</span>
              <span className="font-medium">
                {((stats?.avg_gc_content || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Sources</span>
              <span className="font-medium">
                {Object.keys(stats?.by_source || {}).length}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
