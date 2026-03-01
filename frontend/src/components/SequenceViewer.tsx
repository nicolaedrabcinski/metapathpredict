import { useState } from 'react'
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut } from 'lucide-react'

interface SequenceViewerProps {
  sampleId: number
  sequence: string
  totalLength: number
}

const NUCLEOTIDE_COLORS: Record<string, string> = {
  A: 'text-green-600',
  T: 'text-red-600',
  G: 'text-amber-600',
  C: 'text-blue-600',
  N: 'text-slate-400',
}

export default function SequenceViewer({ sequence, totalLength }: SequenceViewerProps) {
  const [start, setStart] = useState(0)
  const [charsPerLine, setCharsPerLine] = useState(100)
  const [visibleLines] = useState(5)

  const visibleLength = charsPerLine * visibleLines
  const visibleSequence = sequence.slice(start, start + visibleLength)

  const handlePrev = () => {
    setStart(Math.max(0, start - visibleLength))
  }

  const handleNext = () => {
    setStart(Math.min(totalLength - visibleLength, start + visibleLength))
  }

  const zoomIn = () => {
    setCharsPerLine(Math.max(50, charsPerLine - 25))
  }

  const zoomOut = () => {
    setCharsPerLine(Math.min(200, charsPerLine + 25))
  }

  // Split into lines
  const lines: string[] = []
  for (let i = 0; i < visibleSequence.length; i += charsPerLine) {
    lines.push(visibleSequence.slice(i, i + charsPerLine))
  }

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <button
            onClick={handlePrev}
            disabled={start === 0}
            className="p-2 rounded-lg hover:bg-slate-100 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>
          <span className="text-sm text-slate-600">
            Position: {start.toLocaleString()} - {Math.min(start + visibleLength, totalLength).toLocaleString()}
          </span>
          <button
            onClick={handleNext}
            disabled={start + visibleLength >= totalLength}
            className="p-2 rounded-lg hover:bg-slate-100 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ChevronRight className="w-5 h-5" />
          </button>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={zoomIn}
            className="p-2 rounded-lg hover:bg-slate-100"
            title="Zoom in (fewer chars per line)"
          >
            <ZoomIn className="w-5 h-5" />
          </button>
          <span className="text-sm text-slate-500">{charsPerLine} bp/line</span>
          <button
            onClick={zoomOut}
            className="p-2 rounded-lg hover:bg-slate-100"
            title="Zoom out (more chars per line)"
          >
            <ZoomOut className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Progress bar */}
      <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
        <div
          className="h-full bg-green-500 transition-all duration-200"
          style={{
            width: `${(visibleLength / totalLength) * 100}%`,
            marginLeft: `${(start / totalLength) * 100}%`,
          }}
        />
      </div>

      {/* Sequence display */}
      <div className="font-mono text-xs bg-slate-900 text-slate-100 p-4 rounded-lg overflow-x-auto">
        {lines.map((line, lineIndex) => {
          const lineStart = start + lineIndex * charsPerLine
          return (
            <div key={lineIndex} className="flex">
              <span className="text-slate-500 w-16 flex-shrink-0 select-none">
                {lineStart.toString().padStart(6, ' ')}
              </span>
              <span className="ml-4">
                {line.split('').map((char, charIndex) => (
                  <span
                    key={charIndex}
                    className={NUCLEOTIDE_COLORS[char.toUpperCase()] || 'text-slate-400'}
                  >
                    {char}
                  </span>
                ))}
              </span>
            </div>
          )
        })}
      </div>

      {/* Legend */}
      <div className="flex items-center gap-6 text-sm">
        <span className="text-slate-500">Legend:</span>
        <span className="text-green-600 font-mono">A - Adenine</span>
        <span className="text-red-600 font-mono">T - Thymine</span>
        <span className="text-amber-600 font-mono">G - Guanine</span>
        <span className="text-blue-600 font-mono">C - Cytosine</span>
        <span className="text-slate-400 font-mono">N - Unknown</span>
      </div>
    </div>
  )
}
