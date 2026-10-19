import { useCallback, useEffect, useState } from 'react'
import './App.css'
import { api, subscribeJobEvents } from './api/client'
import type { ImageMeta, Preset } from './api/client'
import { EditorMask } from './EditorMask'

type Tab = 'generate' | 'batch' | 'gallery' | 'editor' | 'settings' | 'brand'

const ASPECTS = ['16:9', '4:3', '1:1', '3:2', '2:3', '9:16', '3:4']
const QUALITIES = ['draft', 'standard', 'high']

export default function App() {
  const [tab, setTab] = useState<Tab>('generate')
  const [presets, setPresets] = useState<Preset[]>([])
  const [prompt, setPrompt] = useState('Фон для слайда о цифровой трансформации')
  const [presetId, setPresetId] = useState('slide-background')
  const [aspect, setAspect] = useState('16:9')
  const [quality, setQuality] = useState('standard')
  const [numImages, setNumImages] = useState(2)
  const [enhance, setEnhance] = useState(true)
  const [overlayTitle, setOverlayTitle] = useState('Наш проект')
  const [overlaySubtitle, setOverlaySubtitle] = useState('')
  const [progress, setProgress] = useState<string | null>(null)
  const [results, setResults] = useState<ImageMeta[]>([])
  const [gallery, setGallery] = useState<ImageMeta[]>([])
  const [batchText, setBatchText] = useState(
    'Введение в продукт\nАрхитектура решения\nДорожная карта',
  )
  const [brandColors, setBrandColors] = useState('#0B3D91,#FFFFFF,#1E88E5')
  const [health, setHealth] = useState('')
  const [editImageId, setEditImageId] = useState('')
  const [editPrompt, setEditPrompt] = useState('Замени фон на современный офис')

  const loadPresets = useCallback(async () => {
    const list = await api<Preset[]>('/api/presets')
    setPresets(list)
  }, [])

  const loadGallery = useCallback(async () => {
    const data = await api<{ items: ImageMeta[] }>('/api/gallery?page_size=48')
    setGallery(data.items)
  }, [])

  const loadHealth = useCallback(async () => {
    const h = await api<{ status: string; engine_mode: string }>('/api/health')
    setHealth(`${h.status} / ${h.engine_mode}`)
  }, [])

  useEffect(() => {
    loadPresets().catch(console.error)
    loadGallery().catch(console.error)
    loadHealth().catch(console.error)
  }, [loadPresets, loadGallery, loadHealth])

  const waitJob = async (jobId: string) => {
    return new Promise<string[]>((resolve, reject) => {
      const unsub = subscribeJobEvents(jobId, (ev) => {
        if (ev.type === 'progress') {
          setProgress(`Шаг ${ev.step}/${ev.total}`)
        }
        if (ev.type === 'image') {
          setResults((prev) => [
            ...prev,
            { id: String(ev.image_id), job_id: jobId, url: String(ev.url), favorite: false },
          ])
        }
        if (ev.type === 'done') {
          setProgress(null)
          unsub()
          resolve((ev.image_ids as string[]) ?? [])
        }
        if (ev.type === 'error') {
          setProgress(null)
          unsub()
          reject(new Error(String(ev.message)))
        }
      })
    })
  }

  const onGenerate = async () => {
    setResults([])
    setProgress('В очереди…')
    const body = {
      prompt,
      preset_id: presetId,
      aspect_ratio: aspect,
      quality,
      num_images: numImages,
      enhance_prompt: enhance,
      text_mode: overlayTitle ? 'overlay' : 'none',
      overlay: overlayTitle
        ? { title: overlayTitle, subtitle: overlaySubtitle, position: 'left', font: 'inter', color: '#FFFFFF' }
        : null,
    }
    const { job_id } = await api<{ job_id: string }>('/api/generate', {
      method: 'POST',
      body: JSON.stringify(body),
    })
    await waitJob(job_id)
    await loadGallery()
  }

  const onEnhanceOnly = async () => {
    const res = await api<{ prompt: string; warning?: string }>('/api/enhance-prompt', {
      method: 'POST',
      body: JSON.stringify({ prompt, preset_id: presetId, aspect_ratio: aspect }),
    })
    setPrompt(res.prompt)
    if (res.warning) alert(res.warning)
  }

  const onBatch = async () => {
    const lines = batchText
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean)
      .map((l) => ({ prompt: l, overlay_title: l }))
    setProgress('Пакет…')
    const { job_id } = await api<{ job_id: string }>('/api/batch', {
      method: 'POST',
      body: JSON.stringify({
        lines,
        preset_id: presetId,
        aspect_ratio: aspect,
        quality: 'draft',
        num_images: 1,
        text_mode: 'overlay',
      }),
    })
    await waitJob(job_id)
    setProgress(null)
    await loadGallery()
  }

  const saveSettings = async () => {
    const colors = brandColors.split(',').map((c) => c.trim()).filter(Boolean)
    await api('/api/settings', {
      method: 'PUT',
      body: JSON.stringify({ brand_colors: colors }),
    })
    alert('Сохранено')
  }

  return (
    <div className="layout">
      <header>
        <h1>Генератор для презентаций</h1>
        <span className="health">{health}</span>
      </header>
      <nav>
        {(
          [
            ['generate', 'Генерация'],
            ['batch', 'Пакет'],
            ['gallery', 'Галерея'],
            ['editor', 'Редактор'],
            ['settings', 'Настройки'],
            ['brand', 'Стиль бренда'],
          ] as [Tab, string][]
        ).map(([id, label]) => (
          <button
            key={id}
            className={tab === id ? 'active' : ''}
            onClick={() => setTab(id)}
            type="button"
          >
            {label}
          </button>
        ))}
      </nav>

      {tab === 'generate' && (
        <section className="panel">
          <label>
            Промпт (RU/EN)
            <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={4} />
          </label>
          <div className="row">
            <label>
              Пресет
              <select value={presetId} onChange={(e) => setPresetId(e.target.value)}>
                {presets.map((p) => (
                  <option key={p.id} value={p.id}>{p.name}</option>
                ))}
              </select>
            </label>
            <label>
              Формат
              <select value={aspect} onChange={(e) => setAspect(e.target.value)}>
                {ASPECTS.map((a) => (
                  <option key={a} value={a}>{a}</option>
                ))}
              </select>
            </label>
            <label>
              Качество
              <select value={quality} onChange={(e) => setQuality(e.target.value)}>
                {QUALITIES.map((q) => (
                  <option key={q} value={q}>{q}</option>
                ))}
              </select>
            </label>
            <label>
              Вариантов
              <input
                type="number"
                min={1}
                max={8}
                value={numImages}
                onChange={(e) => setNumImages(Number(e.target.value))}
              />
            </label>
          </div>
          <div className="row">
            <label>
              Заголовок (overlay)
              <input value={overlayTitle} onChange={(e) => setOverlayTitle(e.target.value)} />
            </label>
            <label>
              Подзаголовок
              <input value={overlaySubtitle} onChange={(e) => setOverlaySubtitle(e.target.value)} />
            </label>
          </div>
          <label className="check">
            <input type="checkbox" checked={enhance} onChange={(e) => setEnhance(e.target.checked)} />
            Улучшить промпт (Ollama)
          </label>
          <div className="actions">
            <button type="button" onClick={onEnhanceOnly}>Улучшить</button>
            <button type="button" className="primary" onClick={onGenerate}>Создать (Ctrl+Enter)</button>
          </div>
          {progress && <p className="progress">{progress}</p>}
          <div className="grid">
            {results.map((img) => (
              <figure key={img.id}>
                <img src={img.url} alt="" />
              </figure>
            ))}
          </div>
        </section>
      )}

      {tab === 'batch' && (
        <section className="panel">
          <label>
            Темы слайдов (по строке)
            <textarea value={batchText} onChange={(e) => setBatchText(e.target.value)} rows={8} />
          </label>
          <button type="button" className="primary" onClick={onBatch}>Запустить пакет</button>
          {progress && <p className="progress">{progress}</p>}
        </section>
      )}

      {tab === 'editor' && (
        <section className="panel">
          <label>
            ID изображения из галереи
            <input value={editImageId} onChange={(e) => setEditImageId(e.target.value)} />
          </label>
          <label>
            Инструкция
            <input value={editPrompt} onChange={(e) => setEditPrompt(e.target.value)} />
          </label>
          {editImageId && (
            <EditorMask
              imageUrl={`/api/images/${editImageId}`}
              onExport={() => {
                /* mask upload wired via multipart in future; edit uses full image */
              }}
            />
          )}
          <button
            type="button"
            className="primary"
            disabled={!editImageId}
            onClick={async () => {
              const { job_id } = await api<{ job_id: string }>('/api/edit', {
                method: 'POST',
                body: JSON.stringify({
                  image_id: editImageId,
                  prompt: editPrompt,
                  quality: 'standard',
                }),
              })
              await waitJob(job_id)
              await loadGallery()
            }}
          >
            Применить редактирование
          </button>
        </section>
      )}

      {tab === 'gallery' && (
        <section className="panel">
          <button type="button" onClick={loadGallery}>Обновить</button>
          <div className="grid">
            {gallery.map((img) => (
              <figure key={img.id}>
                <img src={img.url} alt={img.prompt ?? ''} />
                <figcaption>{img.prompt?.slice(0, 80)}</figcaption>
              </figure>
            ))}
          </div>
        </section>
      )}

      {tab === 'settings' && (
        <section className="panel">
          <label>
            Палитра бренда (HEX через запятую)
            <input value={brandColors} onChange={(e) => setBrandColors(e.target.value)} />
          </label>
          <button type="button" onClick={saveSettings}>Сохранить</button>
        </section>
      )}

      {tab === 'brand' && (
        <section className="panel">
          <p>
            Загрузите 15–40 изображений фирменного стиля через API{' '}
            <code>POST /api/lora/datasets</code> или используйте UI после подключения формы загрузки.
            Обучение LoRA: <code>POST /api/lora/train</code> (профиль Docker <code>training</code>).
          </p>
        </section>
      )}
    </div>
  )
}
