import { useRef, useState } from 'react'
import { Stage, Layer, Line } from 'react-konva'
import type Konva from 'konva'

type Props = {
  imageUrl: string
  onExport: (maskDataUrl: string) => void
}

export function EditorMask({ imageUrl, onExport }: Props) {
  const [lines, setLines] = useState<number[][]>([])
  const stageRef = useRef<Konva.Stage>(null)

  const exportMask = () => {
    const stage = stageRef.current
    if (!stage) return
    const uri = stage.toDataURL({ pixelRatio: 1 })
    onExport(uri)
  }

  return (
    <div>
      <img src={imageUrl} alt="" style={{ maxWidth: 200, display: 'block', marginBottom: 8 }} />
      <Stage
        width={512}
        height={288}
        ref={stageRef}
        onMouseDown={(e) => {
          const pos = e.target.getStage()?.getPointerPosition()
          if (!pos) return
          setLines([...lines, [pos.x, pos.y]])
        }}
        onMousemove={(e) => {
          if (!e.evt.buttons) return
          const pos = e.target.getStage()?.getPointerPosition()
          if (!pos || lines.length === 0) return
          const last = lines[lines.length - 1]
          setLines([...lines.slice(0, -1), [...last, pos.x, pos.y]])
        }}
        style={{ border: '1px solid #333', background: '#111' }}
      >
        <Layer>
          {lines.map((line, i) => (
            <Line key={i} points={line} stroke="#fff" strokeWidth={16} tension={0.5} lineCap="round" globalCompositeOperation="source-over" />
          ))}
        </Layer>
      </Stage>
      <button type="button" onClick={exportMask} style={{ marginTop: 8 }}>Экспорт маски</button>
    </div>
  )
}
