import { useQuery } from '@tanstack/react-query'
import { Link, useSearchParams } from 'react-router-dom'
import { Search, Filter, ChevronLeft, ChevronRight } from 'lucide-react'
import { useState } from 'react'
import { api } from '../lib/api'
import ClassBadge from '../components/ClassBadge'

export default function SampleList() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [search, setSearch] = useState(searchParams.get('search') || '')
  
  const page = parseInt(searchParams.get('page') || '1')
  const classFilter = searchParams.get('class') || undefined

  const { data, isLoading } = useQuery({
    queryKey: ['samples', page, classFilter, search],
    queryFn: () => api.getSamples({ page, class_filter: classFilter, search }),
  })

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    setSearchParams({ search, page: '1' })
  }

  const handlePageChange = (newPage: number) => {
    setSearchParams({ ...Object.fromEntries(searchParams), page: String(newPage) })
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Samples</h1>
          <p className="text-slate-600 mt-1">
            Browse and analyze sequence samples
          </p>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4">
        <form onSubmit={handleSearch} className="flex-1 max-w-md">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
            <input
              type="text"
              placeholder="Search by name or NCBI ID..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
            />
          </div>
        </form>

        <div className="flex items-center gap-2">
          <Filter className="w-5 h-5 text-slate-400" />
          <select
            value={classFilter || ''}
            onChange={(e) => setSearchParams({ 
              ...Object.fromEntries(searchParams),
              class: e.target.value,
              page: '1'
            })}
            className="border border-slate-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500"
          >
            <option value="">All Classes</option>
            <option value="bacteria">Bacteria</option>
            <option value="virus">Virus</option>
            <option value="eukaryotic">Eukaryotic</option>
          </select>
        </div>
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl shadow-sm overflow-hidden">
        <table className="w-full">
          <thead className="bg-slate-50 border-b border-slate-200">
            <tr>
              <th className="text-left px-6 py-3 text-sm font-medium text-slate-500">Name</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-slate-500">NCBI ID</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-slate-500">Class</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-slate-500">Length</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-slate-500">GC%</th>
              <th className="text-left px-6 py-3 text-sm font-medium text-slate-500">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {isLoading ? (
              <tr>
                <td colSpan={6} className="px-6 py-8 text-center text-slate-500">
                  Loading...
                </td>
              </tr>
            ) : data?.items.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-6 py-8 text-center text-slate-500">
                  No samples found
                </td>
              </tr>
            ) : (
              data?.items.map((sample) => (
                <tr key={sample.id} className="hover:bg-slate-50">
                  <td className="px-6 py-4">
                    <Link 
                      to={`/samples/${sample.id}`}
                      className="font-medium text-slate-900 hover:text-green-600"
                    >
                      {sample.name}
                    </Link>
                    {sample.organism && (
                      <p className="text-sm text-slate-500">{sample.organism}</p>
                    )}
                  </td>
                  <td className="px-6 py-4 text-slate-600 font-mono text-sm">
                    {sample.ncbi_id || '-'}
                  </td>
                  <td className="px-6 py-4">
                    {sample.true_label && <ClassBadge label={sample.true_label} />}
                  </td>
                  <td className="px-6 py-4 text-slate-600">
                    {sample.length.toLocaleString()} bp
                  </td>
                  <td className="px-6 py-4 text-slate-600">
                    {(sample.gc_content * 100).toFixed(1)}%
                  </td>
                  <td className="px-6 py-4">
                    <Link
                      to={`/samples/${sample.id}`}
                      className="text-green-600 hover:text-green-700 text-sm font-medium"
                    >
                      View
                    </Link>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>

        {/* Pagination */}
        {data && data.pages > 1 && (
          <div className="flex items-center justify-between px-6 py-3 border-t border-slate-200">
            <p className="text-sm text-slate-500">
              Showing {(page - 1) * 20 + 1} to {Math.min(page * 20, data.total)} of {data.total}
            </p>
            <div className="flex items-center gap-2">
              <button
                onClick={() => handlePageChange(page - 1)}
                disabled={page === 1}
                className="p-2 rounded-lg hover:bg-slate-100 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="w-5 h-5" />
              </button>
              <span className="text-sm text-slate-600">
                Page {page} of {data.pages}
              </span>
              <button
                onClick={() => handlePageChange(page + 1)}
                disabled={page === data.pages}
                className="p-2 rounded-lg hover:bg-slate-100 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronRight className="w-5 h-5" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
