import type { components } from './schema'

export type Preset = components['schemas']['PresetInfo']
export type ImageMeta = components['schemas']['ImageMeta']

export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  const headers = new Headers(init?.headers ?? {})
  if (!(init?.body instanceof FormData)) {
    if (!headers.has('Content-Type') && init?.body) {
      headers.set('Content-Type', 'application/json')
    }
    if (!headers.has('Accept')) {
      headers.set('Accept', 'application/json')
    }
  }
  const res = await fetch(path, {
    credentials: 'include',
    ...init,
    headers,
  })
  return res
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await apiFetch(path, init)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || res.statusText)
  }
  if (res.status === 204) {
    return undefined as T
  }
  return res.json() as Promise<T>
}

export async function login(password: string): Promise<void> {
  const body = new FormData()
  body.append('password', password)
  await api('/api/login', { method: 'POST', body })
}

export function subscribeJobEvents(
  jobId: string,
  onEvent: (data: Record<string, unknown>) => void,
): () => void {
  const es = new EventSource(`/api/jobs/${jobId}/events`, { withCredentials: true })
  es.onmessage = (ev) => {
    try {
      onEvent(JSON.parse(ev.data))
    } catch {
      /* ignore */
    }
  }
  return () => es.close()
}
