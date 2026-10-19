const API_KEY = import.meta.env.VITE_API_KEY ?? ''

function headers(): HeadersInit {
  const h: HeadersInit = { 'Content-Type': 'application/json' }
  if (API_KEY) h['X-API-Key'] = API_KEY
  return h
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { ...headers(), ...(init?.headers ?? {}) },
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || res.statusText)
  }
  return res.json() as Promise<T>
}

export type Preset = {
  id: string
  name: string
  default_aspect: string
  transparent: boolean
  text_mode: string
  description: string
}

export type ImageMeta = {
  id: string
  job_id: string
  url: string
  prompt?: string
  favorite: boolean
  width?: number
  height?: number
}

export function subscribeJobEvents(
  jobId: string,
  onEvent: (data: Record<string, unknown>) => void,
): () => void {
  const es = new EventSource(`/api/jobs/${jobId}/events`)
  es.onmessage = (ev) => {
    try {
      onEvent(JSON.parse(ev.data))
    } catch {
      /* ignore */
    }
  }
  return () => es.close()
}
