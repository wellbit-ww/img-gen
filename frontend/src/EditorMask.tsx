import { useEffect, useRef, useState } from 'react'
import { Image as KonvaImage, Layer, Line, Stage } from 'react-konva'
import type Konva from 'konva'

type Props = {
  imageUrl: string
  width: number
  height: number
  onMaskReady: (blob: Blob) => void
}

function dataUrlToBlob(dataUrl: string): Blob {
  const [header, data] = dataUrl.split(',')
  const mime = header.match(/:(.*?);/)?.[1] ?? 'image/png'
  const bin = atob(data)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i)
  return new Blob([bytes], { type: mime })
}

export function EditorMask({ imageUrl, width, height, onMaskReady }: Props) {
  const [lines, setLines] = useState<number[][]>([])
  const [image, setImage] = useState<HTMLImageElement | null>(null)
  const maskStageRef = useRef<Konva.Stage>(null)

  useEffect(() => {
    const img = new window.Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => setImage(img)
    img.src = imageUrl
  }, [imageUrl])

  const exportMask = () => {
    const stage = maskStageRef.current
    if (!stage) return
    const dataUrl = stage.toDataURL({ pixelRatio: 1 })
    onMaskReady(dataUrlToBlob(dataUrl))
  }

  return (
    <div className="space-y-2">
      <Stage
        width={width}
        height={height}
        className="rounded-md border border-border bg-black"
        onMouseDown={(e) => {
          const pos = e.target.getStage()?.getPointerPosition()
          if (!pos) return
          setLines((prev) => [...prev, [pos.x, pos.y]])
        }}
        onMousemove={(e) => {
          if (!e.evt.buttons) return
          const pos = e.target.getStage()?.getPointerPosition()
          if (!pos) return
          setLines((prev) => {
            if (prev.length === 0) return prev
            const last = prev[prev.length - 1]
            return [...prev.slice(0, -1), [...last, pos.x, pos.y]]
          })
        }}
      >
        <Layer>
          {image && <KonvaImage image={image} width={width} height={height} />}
          {lines.map((line, i) => (
            <Line
              key={i}
              points={line}
              stroke="rgba(255,0,0,0.55)"
              strokeWidth={18}
              tension={0.5}
              lineCap="round"
              globalCompositeOperation="source-over"
            />
          ))}
        </Layer>
      </Stage>
      <Stage
        width={width}
        height={height}
        ref={maskStageRef}
        className="hidden"
        style={{ position: 'absolute', left: -9999, top: 0 }}
      >
        <Layer>
          <Line points={[0, 0, width, 0, width, height, 0, height, 0, 0]} fill="#000" closed />
          {lines.map((line, i) => (
            <Line
              key={i}
              points={line}
              stroke="#fff"
              strokeWidth={18}
              tension={0.5}
              lineCap="round"
            />
          ))}
        </Layer>
      </Stage>
      <button type="button" className="text-sm underline" onClick={exportMask}>
        Обновить маску для отправки
      </button>
    </div>
  )
}
