interface ClassBadgeProps {
  label: string
  size?: 'sm' | 'md'
}

const classColors: Record<string, { bg: string; text: string }> = {
  bacteria: { bg: 'bg-green-100', text: 'text-green-700' },
  virus: { bg: 'bg-red-100', text: 'text-red-700' },
  eukaryotic: { bg: 'bg-blue-100', text: 'text-blue-700' },
}

export default function ClassBadge({ label, size = 'md' }: ClassBadgeProps) {
  const colors = classColors[label] || { bg: 'bg-slate-100', text: 'text-slate-700' }
  const sizeClasses = size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-3 py-1 text-sm'

  return (
    <span className={`inline-block rounded-full font-medium capitalize ${colors.bg} ${colors.text} ${sizeClasses}`}>
      {label}
    </span>
  )
}
