import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Moon, Sun } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'

import { api, ApiError, subscribeJobEvents, type ImageMeta, type Preset } from './api/client'
import { LoginGate } from './components/LoginGate'
import { Button } from './components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card'
import { Input } from './components/ui/input'
import { EditorMask } from './EditorMask'
import { useEditorStore } from './stores/editorStore'
import { useThemeStore } from './stores/themeStore'

type Tab = 'generate' | 'batch' | 'gallery' | 'editor' | 'settings' | 'brand'

const ASPECTS = ['16:9', '4:3', '1:1', '3:2', '2:3', '9:16', '3:4']
const QUALITIES = ['draft', 'standard', 'high']

function useJobWaiter() {
  const [progress, setProgress] = useState<string | null>(null)

  const waitJob = useCallback(async (jobId: string, onImage?: (img: ImageMeta) => void) => {
    return new Promise<string[]>((resolve, reject) => {
      const unsub = subscribeJobEvents(jobId, (ev) => {
        if (ev.type === 'progress') {
          setProgress(`Шаг ${ev.step}/${ev.total}`)
        }
        if (ev.type === 'image') {
          onImage?.({
            id: String(ev.image_id),
            job_id: jobId,
            url: String(ev.url),
            favorite: false,
            has_overlay: false,
            created_at: new Date().toISOString(),
          })
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
  }, [])

  return { progress, setProgress, waitJob }
}

export default function App() {
  const [tab, setTab] = useState<Tab>('generate')
  const [authed, setAuthed] = useState<boolean | null>(null)
  const [prompt, setPrompt] = useState('Фон для слайда о цифровой трансформации')
  const [presetId, setPresetId] = useState('slide-background')
  const [aspect, setAspect] = useState('16:9')
  const [quality, setQuality] = useState('standard')
  const [numImages, setNumImages] = useState(4)
  const [enhance, setEnhance] = useState(true)
  const [overlayTitle, setOverlayTitle] = useState('Наш проект')
  const [overlaySubtitle, setOverlaySubtitle] = useState('')
  const [results, setResults] = useState<ImageMeta[]>([])
  const [batchText, setBatchText] = useState(
    'Введение в продукт\nАрхитектура решения\nДорожная карта',
  )
  const [brandColors, setBrandColors] = useState('#0B3D91,#FFFFFF,#1E88E5')
  const [editImageId, setEditImageId] = useState('')
  const [editPrompt, setEditPrompt] = useState('Замени фон на современный офис')
  const [editQuality, setEditQuality] = useState('standard')
  const [enhanceNotice, setEnhanceNotice] = useState<string | null>(null)

  const theme = useThemeStore((s) => s.theme)
  const toggleTheme = useThemeStore((s) => s.toggle)
  const maskBlob = useEditorStore((s) => s.maskBlob)
  const referenceFiles = useEditorStore((s) => s.referenceFiles)
  const setMaskBlob = useEditorStore((s) => s.setMaskBlob)
  const addReferenceFiles = useEditorStore((s) => s.addReferenceFiles)
  const clearReferences = useEditorStore((s) => s.clearReferences)

  const queryClient = useQueryClient()
  const { progress, setProgress, waitJob } = useJobWaiter()

  const healthQuery = useQuery({
    queryKey: ['health'],
    queryFn: () => api<{ status: string; engine_mode: string }>('/api/health'),
    retry: false,
  })

  const presetsQuery = useQuery({
    queryKey: ['presets'],
    queryFn: () => api<Preset[]>('/api/presets'),
    enabled: authed === true,
    retry: false,
  })

  const galleryQuery = useQuery({
    queryKey: ['gallery'],
    queryFn: () => api<{ items: ImageMeta[] }>('/api/gallery?page_size=48'),
    enabled: authed === true,
  })

  useEffect(() => {
    api<Preset[]>('/api/presets')
      .then(() => setAuthed(true))
      .catch((err) => {
        setAuthed(!(err instanceof ApiError && err.status === 401))
      })
  }, [])

  const generateMutation = useMutation({
    mutationFn: async () => {
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
          ? {
              title: overlayTitle,
              subtitle: overlaySubtitle,
              position: 'left',
              font: 'inter',
              color: '#FFFFFF',
            }
          : null,
      }
      const { job_id } = await api<{ job_id: string }>('/api/generate', {
        method: 'POST',
        body: JSON.stringify(body),
      })
      await waitJob(job_id, (img) => setResults((prev) => [...prev, img]))
      await queryClient.invalidateQueries({ queryKey: ['gallery'] })
    },
  })

  const enhanceMutation = useMutation({
    mutationFn: async () => {
      const res = await api<{ prompt: string; warning?: string | null; notes?: string | null }>(
        '/api/enhance-prompt',
        {
          method: 'POST',
          body: JSON.stringify({ prompt, preset_id: presetId, aspect_ratio: aspect }),
        },
      )
      setPrompt(res.prompt)
      setEnhanceNotice([res.warning, res.notes].filter(Boolean).join(' ') || null)
    },
  })

  const batchMutation = useMutation({
    mutationFn: async () => {
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
      await queryClient.invalidateQueries({ queryKey: ['gallery'] })
    },
  })

  const editMutation = useMutation({
    mutationFn: async () => {
      const fd = new FormData()
      fd.append('prompt', editPrompt)
      fd.append('quality', editQuality)
      fd.append('image_id', editImageId)
      if (maskBlob) fd.append('mask', maskBlob, 'mask.png')
      for (const file of referenceFiles) {
        fd.append('references', file, file.name)
      }
      setProgress('Редактирование…')
      const { job_id } = await api<{ job_id: string }>('/api/edit', { method: 'POST', body: fd })
      await waitJob(job_id)
      await queryClient.invalidateQueries({ queryKey: ['gallery'] })
    },
  })

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && tab === 'generate') {
        e.preventDefault()
        generateMutation.mutate()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [tab, generateMutation])

  const upscaleMutation = useMutation({
    mutationFn: async (imageId: string) => {
      setProgress('Апскейл…')
      const { job_id } = await api<{ job_id: string }>('/api/upscale', {
        method: 'POST',
        body: JSON.stringify({ image_id: imageId, target: '4k' }),
      })
      await waitJob(job_id)
      await queryClient.invalidateQueries({ queryKey: ['gallery'] })
    },
  })

  const saveSettings = async () => {
    const colors = brandColors.split(',').map((c) => c.trim()).filter(Boolean)
    await api('/api/settings', {
      method: 'PUT',
      body: JSON.stringify({ brand_colors: colors }),
    })
    alert('Сохранено')
  }

  if (authed === false) {
    return <LoginGate onSuccess={() => setAuthed(true)} />
  }

  const presets = presetsQuery.data ?? []
  const gallery = galleryQuery.data?.items ?? []

  return (
    <div className="mx-auto flex min-h-screen max-w-6xl flex-col gap-4 p-4">
      <header className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-2xl font-semibold">Генератор для презентаций</h1>
          <p className="text-sm text-muted-foreground">
            {healthQuery.data ? `${healthQuery.data.status} / ${healthQuery.data.engine_mode}` : '…'}
          </p>
        </div>
        <Button type="button" variant="outline" size="sm" onClick={toggleTheme}>
          {theme === 'dark' ? <Sun className="size-4" /> : <Moon className="size-4" />}
          {theme === 'dark' ? 'Светлая' : 'Тёмная'}
        </Button>
      </header>

      <nav className="flex flex-wrap gap-2">
        {(
          [
            ['generate', 'Генерация'],
            ['batch', 'Пакет'],
            ['gallery', 'Галерея'],
            ['editor', 'Редактор'],
            ['settings', 'Настройки'],
            ['brand', 'LoRA'],
          ] as [Tab, string][]
        ).map(([id, label]) => (
          <Button
            key={id}
            type="button"
            variant={tab === id ? 'default' : 'secondary'}
            size="sm"
            onClick={() => setTab(id)}
          >
            {label}
          </Button>
        ))}
      </nav>

      {progress && <p className="text-sm text-primary">{progress}</p>}

      {tab === 'generate' && (
        <Card>
          <CardHeader>
            <CardTitle>Генерация</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <label className="flex flex-col gap-1 text-sm">
              Промпт (RU/EN)
              <textarea
                className="min-h-24 rounded-md border border-input bg-transparent px-3 py-2"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
              />
            </label>
            <div className="grid gap-3 md:grid-cols-4">
              <label className="flex flex-col gap-1 text-sm">
                Пресет
                <select
                  className="h-9 rounded-md border border-input bg-transparent px-2"
                  value={presetId}
                  onChange={(e) => setPresetId(e.target.value)}
                >
                  {presets.map((p) => (
                    <option key={p.id} value={p.id}>{p.name}</option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-1 text-sm">
                Формат
                <select
                  className="h-9 rounded-md border border-input bg-transparent px-2"
                  value={aspect}
                  onChange={(e) => setAspect(e.target.value)}
                >
                  {ASPECTS.map((a) => (
                    <option key={a} value={a}>{a}</option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-1 text-sm">
                Качество
                <select
                  className="h-9 rounded-md border border-input bg-transparent px-2"
                  value={quality}
                  onChange={(e) => setQuality(e.target.value)}
                >
                  {QUALITIES.map((q) => (
                    <option key={q} value={q}>{q}</option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-1 text-sm">
                Вариантов
                <Input
                  type="number"
                  min={1}
                  max={8}
                  value={numImages}
                  onChange={(e) => setNumImages(Number(e.target.value))}
                />
              </label>
            </div>
            <div className="grid gap-3 md:grid-cols-2">
              <label className="flex flex-col gap-1 text-sm">
                Заголовок (overlay)
                <Input value={overlayTitle} onChange={(e) => setOverlayTitle(e.target.value)} />
              </label>
              <label className="flex flex-col gap-1 text-sm">
                Подзаголовок
                <Input value={overlaySubtitle} onChange={(e) => setOverlaySubtitle(e.target.value)} />
              </label>
            </div>
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={enhance} onChange={(e) => setEnhance(e.target.checked)} />
              Улучшить промпт (Ollama)
            </label>
            <div className="flex flex-wrap gap-2">
              <Button type="button" variant="secondary" onClick={() => enhanceMutation.mutate()}>
                Улучшить
              </Button>
              <Button type="button" onClick={() => generateMutation.mutate()}>
                Создать (Ctrl+Enter)
              </Button>
            </div>
            {enhanceNotice && <p className="text-sm text-muted-foreground">{enhanceNotice}</p>}
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {results.map((img) => (
                <figure key={img.id} className="overflow-hidden rounded-lg border border-border">
                  <img src={img.url} alt="" className="w-full object-cover" />
                </figure>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {tab === 'batch' && (
        <Card>
          <CardHeader>
            <CardTitle>Пакетная генерация</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <textarea
              className="min-h-48 w-full rounded-md border border-input bg-transparent px-3 py-2"
              value={batchText}
              onChange={(e) => setBatchText(e.target.value)}
            />
            <Button type="button" onClick={() => batchMutation.mutate()}>Запустить пакет</Button>
          </CardContent>
        </Card>
      )}

      {tab === 'editor' && (
        <Card>
          <CardHeader>
            <CardTitle>Редактор</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-3 md:grid-cols-2">
              <label className="flex flex-col gap-1 text-sm">
                ID изображения
                <Input value={editImageId} onChange={(e) => setEditImageId(e.target.value)} />
              </label>
              <label className="flex flex-col gap-1 text-sm">
                Качество
                <select
                  className="h-9 rounded-md border border-input bg-transparent px-2"
                  value={editQuality}
                  onChange={(e) => setEditQuality(e.target.value)}
                >
                  {QUALITIES.map((q) => (
                    <option key={q} value={q}>{q}</option>
                  ))}
                </select>
              </label>
            </div>
            <label className="flex flex-col gap-1 text-sm">
              Инструкция
              <Input value={editPrompt} onChange={(e) => setEditPrompt(e.target.value)} />
            </label>
            <label className="flex flex-col gap-1 text-sm">
              Референсы (до 10)
              <input
                type="file"
                accept="image/*"
                multiple
                onChange={(e) => e.target.files && addReferenceFiles(e.target.files)}
              />
              {referenceFiles.length > 0 && (
                <Button type="button" variant="ghost" size="sm" onClick={clearReferences}>
                  Очистить референсы ({referenceFiles.length})
                </Button>
              )}
            </label>
            {editImageId && (
              <EditorMask
                imageUrl={`/api/images/${editImageId}`}
                width={768}
                height={432}
                onMaskReady={setMaskBlob}
              />
            )}
            {maskBlob && <p className="text-xs text-muted-foreground">Маска готова к отправке</p>}
            <Button
              type="button"
              disabled={!editImageId || editMutation.isPending}
              onClick={() => editMutation.mutate()}
            >
              Применить редактирование
            </Button>
          </CardContent>
        </Card>
      )}

      {tab === 'gallery' && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Галерея</CardTitle>
            <Button type="button" variant="secondary" size="sm" onClick={() => galleryQuery.refetch()}>
              Обновить
            </Button>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {gallery.map((img) => (
                <figure key={img.id} className="space-y-2 rounded-lg border border-border p-2">
                  <img src={img.url} alt={img.prompt ?? ''} className="w-full rounded-md object-cover" />
                  <figcaption className="text-xs text-muted-foreground line-clamp-2">
                    {img.prompt?.slice(0, 120)}
                  </figcaption>
                  <div className="flex flex-wrap gap-2">
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      onClick={() => {
                        setEditImageId(img.id)
                        setTab('editor')
                      }}
                    >
                      Редактировать
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant="secondary"
                      onClick={() => upscaleMutation.mutate(img.id)}
                    >
                      4K
                    </Button>
                    <a
                      className="inline-flex h-8 items-center rounded-md border border-input px-3 text-xs"
                      href={`/api/images/${img.id}/export`}
                      download
                    >
                      Скачать
                    </a>
                  </div>
                </figure>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {tab === 'settings' && (
        <Card>
          <CardHeader>
            <CardTitle>Настройки</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <label className="flex flex-col gap-1 text-sm">
              Палитра бренда (HEX через запятую)
              <Input value={brandColors} onChange={(e) => setBrandColors(e.target.value)} />
            </label>
            <Button type="button" onClick={saveSettings}>Сохранить</Button>
          </CardContent>
        </Card>
      )}

      {tab === 'brand' && (
        <Card>
          <CardHeader>
            <CardTitle>LoRA / фирменный стиль</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            Загрузка датасета: <code>POST /api/lora/datasets</code>. Обучение:{' '}
            <code>POST /api/lora/train</code> (Docker profile training).
          </CardContent>
        </Card>
      )}
    </div>
  )
}
