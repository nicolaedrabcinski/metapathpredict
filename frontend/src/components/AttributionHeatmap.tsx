import { useEffect, useRef } from 'react'
import * as d3 from 'd3'

interface AttributionHeatmapProps {
  sequence: string
  scores: number[]
  start: number
}

const NUCLEOTIDE_COLORS: Record<string, string> = {
  A: '#22c55e',
  T: '#ef4444',
  G: '#f59e0b',
  C: '#3b82f6',
  N: '#9ca3af',
}

export default function AttributionHeatmap({ sequence, scores, start }: AttributionHeatmapProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || !sequence || !scores.length) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const containerWidth = containerRef.current.clientWidth
    const margin = { top: 20, right: 20, bottom: 60, left: 60 }
    const width = containerWidth - margin.left - margin.right
    const height = 200 - margin.top - margin.bottom

    const charWidth = Math.max(2, width / sequence.length)
    const actualWidth = charWidth * sequence.length

    svg.attr('width', actualWidth + margin.left + margin.right)
       .attr('height', 200)

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Color scale for attribution
    const colorScale = d3.scaleLinear<string>()
      .domain([-1, 0, 1])
      .range(['#ef4444', '#f8fafc', '#22c55e'])

    // Draw attribution bars
    g.selectAll('.attribution-bar')
      .data(scores)
      .join('rect')
      .attr('class', 'attribution-bar')
      .attr('x', (_, i) => i * charWidth)
      .attr('y', d => d >= 0 ? height / 2 - Math.abs(d) * (height / 2) : height / 2)
      .attr('width', charWidth - 1)
      .attr('height', d => Math.abs(d) * (height / 2))
      .attr('fill', d => colorScale(d))
      .attr('opacity', 0.8)

    // Draw center line
    g.append('line')
      .attr('x1', 0)
      .attr('x2', actualWidth)
      .attr('y1', height / 2)
      .attr('y2', height / 2)
      .attr('stroke', '#94a3b8')
      .attr('stroke-dasharray', '4,4')

    // Draw sequence (below the chart)
    if (charWidth >= 6) {
      g.selectAll('.nucleotide')
        .data(sequence.split(''))
        .join('text')
        .attr('class', 'nucleotide')
        .attr('x', (_, i) => i * charWidth + charWidth / 2)
        .attr('y', height + 15)
        .attr('text-anchor', 'middle')
        .attr('font-family', 'monospace')
        .attr('font-size', Math.min(10, charWidth))
        .attr('fill', d => NUCLEOTIDE_COLORS[d.toUpperCase()] || '#9ca3af')
        .text(d => d)
    }

    // X-axis with positions
    const xScale = d3.scaleLinear()
      .domain([start, start + sequence.length])
      .range([0, actualWidth])

    const xAxis = d3.axisBottom(xScale)
      .tickValues(d3.range(start, start + sequence.length, Math.ceil(sequence.length / 10)))
      .tickFormat(d => d.toLocaleString())

    g.append('g')
      .attr('transform', `translate(0,${height + 25})`)
      .call(xAxis)
      .selectAll('text')
      .attr('font-size', 10)

    // Y-axis label
    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -100)
      .attr('y', 15)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('fill', '#64748b')
      .text('Attribution Score')

    // Add tooltip
    const tooltip = d3.select('body').append('div')
      .attr('class', 'tooltip')
      .style('position', 'absolute')
      .style('background', '#1e293b')
      .style('color', 'white')
      .style('padding', '8px 12px')
      .style('border-radius', '6px')
      .style('font-size', '12px')
      .style('pointer-events', 'none')
      .style('opacity', 0)
      .style('z-index', 1000)

    g.selectAll('.attribution-bar')
      .on('mouseover', function(event, d) {
        const i = scores.indexOf(d as number)
        d3.select(this).attr('opacity', 1)
        tooltip.transition().duration(100).style('opacity', 1)
        tooltip.html(`
          Position: ${(start + i).toLocaleString()}<br/>
          Nucleotide: ${sequence[i]}<br/>
          Score: ${(d as number).toFixed(4)}
        `)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px')
      })
      .on('mouseout', function() {
        d3.select(this).attr('opacity', 0.8)
        tooltip.transition().duration(100).style('opacity', 0)
      })

    return () => {
      tooltip.remove()
    }
  }, [sequence, scores, start])

  return (
    <div ref={containerRef} className="overflow-x-auto">
      <svg ref={svgRef} />
      
      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mt-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-red-500 rounded" />
          <span className="text-slate-600">Negative (suppresses)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-slate-100 rounded border" />
          <span className="text-slate-600">Neutral</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-green-500 rounded" />
          <span className="text-slate-600">Positive (activates)</span>
        </div>
      </div>
    </div>
  )
}
